#!/usr/bin/env python3
"""
Lattice Planner for F1TENTH
============================
Two-layer planning architecture:
  Global layer : pre-computed raceline from raceline_generator.py (path_v.yaml)
  Local  layer : this node — samples N lateral candidate paths, checks collision
                 via lidar, scores them, tracks the best with pure pursuit

Architecture:
  /scan  ─────────────────────────────┐
  /ego_racecar/odom  ─────────────────┤─► [lattice_planner] ──► /drive
  path_v.yaml (global raceline)  ───────┘
                                        └► /lattice/candidates  (RViz markers)
                                        └► /lattice/selected    (RViz marker)

Algorithm per odom tick:
  1. Localise  – find nearest waypoint on global path
  2. Window    – extract next PLAN_HORIZON metres of global path
  3. Generate  – shift window by each lateral offset → N candidate paths
  4. Collide   – transform lidar points to map frame, reject any candidate
                 whose points come within SAFETY_RADIUS of a lidar hit
  5. Score     – cost = w_dev * |offset| + w_smooth * curvature_change
  6. Track     – pure-pursuit lookahead on the winning candidate
  7. Publish   – AckermannDriveStamped on /drive
"""

import math
import os

import numpy as np
import rclpy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class LatticePlanner(Node):

    def __init__(self):
        super().__init__("lattice_planner")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("plan_horizon",   2.5)   # m ahead to plan (shorter = better corners)
        self.declare_parameter("num_offsets",    11)    # candidate count (more = finer steps)
        self.declare_parameter("max_offset",     1.0)   # m lateral each side
        self.declare_parameter("safety_radius",  0.55)  # m around each path pt (larger = earlier detection)
        self.declare_parameter("w_deviation",    1.0)   # penalty: stray from global
        self.declare_parameter("w_smooth",       0.3)   # penalty: path curvature
        self.declare_parameter("lookahead",      1.2)   # m pure-pursuit lookahead
        self.declare_parameter("min_lookahead",  0.5)
        self.declare_parameter("max_lookahead",  1.8)   # smaller max = tighter corner tracking
        self.declare_parameter("speed_gain",     0.35)  # lookahead = speed * gain
        self.declare_parameter("steer_gain",     1.2)   # P gain on pure pursuit
        self.declare_parameter("steer_limit",    0.41)  # rad (~23°)

        p = lambda name: self.get_parameter(name).value
        self.plan_horizon   = p("plan_horizon")
        self.num_offsets    = p("num_offsets")
        self.max_offset     = p("max_offset")
        self.safety_radius  = p("safety_radius")
        self.w_deviation    = p("w_deviation")
        self.w_smooth       = p("w_smooth")
        self.min_lookahead  = p("min_lookahead")
        self.max_lookahead  = p("max_lookahead")
        self.speed_gain     = p("speed_gain")
        self.steer_gain     = p("steer_gain")
        self.steer_limit    = p("steer_limit")

        # ── Waypoints ─────────────────────────────────────────────────────────
        self.waypoints, self.wp_velocities = self._load_waypoints()
        self.num_pts = len(self.waypoints)
        self.normals = self._calc_normals()
        self.get_logger().info(f"Loaded {self.num_pts} global waypoints")

        # ── State ─────────────────────────────────────────────────────────────
        self.scan_pts_car    = None
        self.prev_nearest    = 0
        self.current_speed   = 0.0
        self.committed_offset  = 0.0
        self.prev_pos          = None
        self.dist_since_check  = 0.0
        self.prev_steer        = 0.0  # low-pass filter state

        # ── Lateral offsets to sample ──────────────────────────────────────
        n = self.num_offsets
        self.offsets = np.linspace(-self.max_offset, self.max_offset, n)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.scan_sub  = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, 10)
        self.odom_sub  = self.create_subscription(
            Odometry, "/ego_racecar/odom", self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self.cand_pub  = self.create_publisher(
            MarkerArray, "/lattice/candidates", 10)
        self.sel_pub   = self.create_publisher(
            Marker, "/lattice/selected", 10)

        self.get_logger().info("Lattice planner ready.")

    # ── Waypoint loading ─────────────────────────────────────────────────────

    def _load_waypoints(self):
        path = str(self.get_parameter("waypoints_path").value)
        if not path:
            pkg = get_package_share_directory("f1tenth_controller")
            path = os.path.join(pkg, "path", "path_v.yaml")
        self.get_logger().info(f"Loading waypoints: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        wps = data["waypoints"]
        xy = np.array([[w["x"], w["y"]] for w in wps])
        vel = np.array([w.get("v", 2.0) for w in wps])
        return xy, vel

    def _calc_normals(self):
        """Left-pointing unit normals at each waypoint."""
        n = self.num_pts
        normals = np.zeros((n, 2))
        for i in range(n):
            fwd = self.waypoints[(i + 1) % n] - self.waypoints[(i - 1) % n]
            normals[i] = np.array([-fwd[1], fwd[0]])
            nrm = np.linalg.norm(normals[i])
            if nrm > 1e-9:
                normals[i] /= nrm
        return normals

    # ── Scan callback ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        angles = np.linspace(msg.angle_min, msg.angle_max,
                             len(msg.ranges))
        r = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        r = r[valid];  a = angles[valid]
        self.scan_pts_car = np.column_stack([r * np.cos(a), r * np.sin(a)])

    # ── Odom callback (main loop) ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        if self.scan_pts_car is None:
            return

        # 1. Car pose
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        self.current_speed = msg.twist.twist.linear.x
        curr_pos = np.array([px, py])

        # 2. Nearest waypoint (local window search)
        nearest = self._find_nearest(curr_pos)
        self.prev_nearest = nearest

        # 3. Extract global path window
        window_idx = self._window_indices(nearest)
        global_window = self.waypoints[window_idx]  # (M, 2)
        window_normals = self.normals[window_idx]   # (M, 2)

        # 4. Transform lidar to map frame.
        #    Two filtered sets:
        #      scan_map_fwd — only points AHEAD of the car (x > 0 in car frame)
        #                     used for path-clear checks and candidate scoring.
        #                     Once the car passes an obstacle it disappears from
        #                     this set, allowing return to centerline.
        #      scan_map_all — all points, kept only for the all-blocked fallback.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        # Forward filter: only points with x > 0.1 m in the car frame
        fwd_mask = self.scan_pts_car[:, 0] > 0.1
        scan_fwd_car = self.scan_pts_car[fwd_mask]
        scan_map_fwd_raw = (R @ scan_fwd_car.T).T + curr_pos if len(scan_fwd_car) > 0 \
                           else np.zeros((0, 2))

        # Corridor filter on forward points only
        corridor_width = self.max_offset + self.safety_radius + 0.3
        if len(scan_map_fwd_raw) > 0 and len(global_window) > 0:
            diff_sw = scan_map_fwd_raw[:, np.newaxis, :] - global_window[np.newaxis, :, :]
            dist_to_path = np.linalg.norm(diff_sw, axis=2).min(axis=1)
            scan_map = scan_map_fwd_raw[dist_to_path <= corridor_width]
        else:
            scan_map = scan_map_fwd_raw

        # 5. Commitment state machine — only re-plan when necessary.
        #
        # States:
        #   FOLLOWING  committed_offset == 0   — tracking global raceline
        #   AVOIDING   committed_offset != 0   — locked on avoidance path
        #
        # Transitions:
        #   FOLLOWING → AVOIDING : global path blocked → pick best clear offset
        #   AVOIDING  → FOLLOWING: global path clear for CLEAR_HOLD ticks in a row
        #   AVOIDING  → AVOIDING : committed path also blocked → re-plan once
        # ── Distance-based replanning ────────────────────────────────────────
        # Check interval: every CHECK_DIST metres driven (velocity-independent).
        #
        # FOLLOWING (offset=0):
        #   Each check: is global path blocked?
        #     YES → pick best candidate, lock offset → AVOIDING
        #     NO  → stay on global path
        #
        # AVOIDING (offset!=0):
        #   Each check: is locked path still safe?
        #     NO  → re-plan, lock new candidate → stay AVOIDING
        #     YES → is global path (centerline) also safe?
        #             YES → return to global path → FOLLOWING
        #             NO  → obstacle still present, keep locked offset
        DETECT_DIST = 3.0   # m — how far to drive before first checking for obstacle
        REPLAN_DIST = 3.0   # m — how far to drive before re-evaluating locked path (AVOIDING)

        # Accumulate distance since last check
        if self.prev_pos is not None:
            self.dist_since_check += float(np.linalg.norm(curr_pos - self.prev_pos))
        self.prev_pos = curr_pos.copy()

        all_blocked = False
        check_dist  = DETECT_DIST if self.committed_offset == 0.0 else REPLAN_DIST
        do_check    = self.dist_since_check >= check_dist

        if self.committed_offset == 0.0:
            # FOLLOWING — check every DETECT_DIST metres
            if do_check:
                self.dist_since_check = 0.0
                if not self._path_clear(global_window, scan_map):
                    _, new_offset, all_blocked = self._evaluate_candidates(
                        global_window, window_normals, window_idx, scan_map)
                    self.committed_offset = new_offset
                    self.get_logger().info(
                        f"Obstacle — locked offset {new_offset:+.2f} m",
                        throttle_duration_sec=0.5)
        else:
            # AVOIDING — re-evaluate only every REPLAN_DIST metres
            if do_check:
                self.dist_since_check = 0.0
                # 1. Check centerline FIRST — if clear, return immediately.
                #    Never re-plan to an opposite offset when the centerline
                #    is already the right answer.
                if self._path_clear(global_window, scan_map):
                    self.committed_offset = 0.0
                    self.get_logger().info("Cleared — back to centerline")
                else:
                    # Obstacle still blocking centerline — check committed path
                    committed_path = global_window + window_normals * self.committed_offset
                    if not self._path_clear(committed_path, scan_map):
                        # Committed path also blocked → re-plan
                        _, new_offset, all_blocked = self._evaluate_candidates(
                            global_window, window_normals, window_idx, scan_map)
                        self.committed_offset = new_offset
                        self.get_logger().info(
                            f"Re-plan — new offset {new_offset:+.2f} m",
                            throttle_duration_sec=0.5)
                    # else: committed path still safe, keep it

        best_offset = self.committed_offset
        best_path   = global_window + window_normals * best_offset

        # Anchor path start to car position so pure pursuit always has a
        # reachable target when the car is slightly off the committed path.
        if len(best_path) >= 2:
            blend_pts = min(5, len(best_path))
            for k in range(blend_pts):
                t = (k + 1) / (blend_pts + 1)
                best_path[k] = (1.0 - t) * curr_pos + t * best_path[k]

        # 6. Pure pursuit on best candidate
        steer, speed = self._pure_pursuit(curr_pos, yaw, best_path, nearest)

        # Smooth steering with a low-pass filter to remove tick-to-tick jitter.
        # alpha = how fast steering responds: 0.0 = frozen, 1.0 = instant.
        # 0.3 keeps ~30% of the new value + 70% of previous → smooth transitions.
        STEER_ALPHA = 0.3
        steer = STEER_ALPHA * steer + (1.0 - STEER_ALPHA) * self.prev_steer
        self.prev_steer = steer

        # 7. Speed control
        # a) Actively avoiding: small speed reduction only
        if best_offset != 0.0:
            deviation_ratio = abs(best_offset) / max(self.max_offset, 1e-6)
            speed *= max(0.7, 1.0 - 0.3 * deviation_ratio)

        # b) All paths blocked: stop almost completely
        if all_blocked:
            speed *= 0.15
            self.get_logger().warn("All candidates blocked — braking",
                                   throttle_duration_sec=1.0)

        # c) Distance-to-nearest-forward-obstacle: primary speed limiter.
        SLOW_START = 2.0
        STOP_DIST  = 0.5
        MIN_SPEED  = 0.4
        if len(scan_fwd_car) > 0:
            min_fwd = float(np.linalg.norm(scan_fwd_car, axis=1).min())
            if min_fwd < SLOW_START:
                t = max(0.0, (min_fwd - STOP_DIST) / (SLOW_START - STOP_DIST))
                speed *= max(MIN_SPEED, t)

        # 8. Publish drive
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.drive.speed           = float(max(speed, 0.0))
        drive.drive.steering_angle  = float(steer)
        self.drive_pub.publish(drive)

        # 8. Visualise
        self._publish_markers(global_window, window_normals, best_path, nearest)

    # ── Nearest waypoint ─────────────────────────────────────────────────────

    def _find_nearest(self, pos: np.ndarray) -> int:
        window = 60
        start  = self.prev_nearest
        idxs   = [(start + i) % self.num_pts for i in range(-5, window)]
        pts    = self.waypoints[idxs]
        local  = np.argmin(np.linalg.norm(pts - pos, axis=1))
        return idxs[local]

    # ── Window of upcoming waypoints ─────────────────────────────────────────

    def _window_indices(self, start: int):
        horizon = self.plan_horizon
        idxs = []
        arc  = 0.0
        i    = start
        for _ in range(self.num_pts):
            idxs.append(i)
            nxt = (i + 1) % self.num_pts
            arc += np.linalg.norm(self.waypoints[nxt] - self.waypoints[i])
            i   = nxt
            if arc >= horizon:
                break
        return np.array(idxs)

    # ── Candidate generation and scoring ─────────────────────────────────────

    def _evaluate_candidates(self, global_win, win_normals, win_idx,
                              scan_map: np.ndarray):
        """
        Returns (best_path [M,2], best_offset float, blocked bool).

        If ALL candidates are blocked: returns the least-bad one (smallest
        min-clearance) so the car can at least slow down gracefully instead
        of crashing at full speed.
        """
        best_path      = global_win.copy()
        best_offset    = 0.0
        best_score     = float("inf")
        all_blocked    = True

        fallback_path      = global_win.copy()
        fallback_offset    = 0.0        # BUG FIX: track offset alongside path
        fallback_clearance = 0.0

        for offset in self.offsets:
            candidate = global_win + win_normals * offset

            min_clearance = float("inf")
            if len(scan_map) > 0:
                diff  = candidate[:, np.newaxis, :] - scan_map[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_clearance = float(dists.min())

            if min_clearance < self.safety_radius:
                if min_clearance > fallback_clearance:
                    fallback_clearance = min_clearance
                    fallback_path      = candidate
                    fallback_offset    = offset   # BUG FIX
                continue

            all_blocked = False
            dev_cost    = self.w_deviation * abs(offset)
            smooth_cost = self.w_smooth * self._curvature_cost(candidate)
            # Right-side bias: normals are left-pointing so negative offset = right.
            # Small penalty for left (positive) over right (negative) as tiebreaker.
            side_bias   = 0.05 * max(0.0, offset)
            score       = dev_cost + smooth_cost + side_bias

            if score < best_score:
                best_score  = score
                best_path   = candidate
                best_offset = offset

        if all_blocked:
            return fallback_path, fallback_offset, True   # BUG FIX

        return best_path, best_offset, False

    def _path_clear(self, path: np.ndarray, scan_map: np.ndarray) -> bool:
        """Return True if no lidar point is within safety_radius of any path point."""
        if len(scan_map) == 0:
            return True
        diff  = path[:, np.newaxis, :] - scan_map[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return float(dists.min()) >= self.safety_radius

    def _curvature_cost(self, path: np.ndarray) -> float:
        """Sum of squared curvatures along the candidate path."""
        n = len(path)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(1, n - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ac = np.linalg.norm(c - a)
            cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            denom = ab * bc * ac
            kappa = 2.0 * cross / denom if denom > 1e-9 else 0.0
            total += kappa ** 2
        return total

    # ── Pure pursuit on local path ────────────────────────────────────────────

    def _pure_pursuit(self, pos, yaw, path, nearest_idx):
        # Adaptive lookahead
        L = np.clip(abs(self.current_speed) * self.speed_gain,
                    self.min_lookahead, self.max_lookahead)

        # Walk forward along local path until arc >= L
        arc = 0.0
        target = path[-1]
        for i in range(len(path) - 1):
            arc += np.linalg.norm(path[i + 1] - path[i])
            if arc >= L:
                target = path[i + 1]
                break

        # Transform target to car frame
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        cos_y = math.cos(-yaw);  sin_y = math.sin(-yaw)
        local_x =  cos_y * dx - sin_y * dy
        local_y =  sin_y * dx + cos_y * dy

        # Pure pursuit steering
        L_actual = max(math.hypot(local_x, local_y), 1e-6)
        steer = self.steer_gain * (2.0 * local_y) / (L_actual ** 2)
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        # Speed from global path at nearest index (speed profile)
        speed = float(self.wp_velocities[nearest_idx])

        return steer, speed

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_markers(self, global_win, win_normals, best_path, nearest_idx):
        stamp = self.get_clock().now().to_msg()

        # All candidates as thin grey lines
        ma = MarkerArray()
        for j, offset in enumerate(self.offsets):
            cand = global_win + win_normals * offset
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp    = stamp
            m.ns = "lattice_candidates";  m.id = j
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color.r = 0.5;  m.color.g = 0.5;  m.color.b = 0.5;  m.color.a = 0.4
            for pt in cand:
                from geometry_msgs.msg import Point
                p = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = 0.05
                m.points.append(p)
            ma.markers.append(m)
        self.cand_pub.publish(ma)

        # Selected path as bright green line
        sel = Marker()
        sel.header.frame_id = "map"
        sel.header.stamp    = stamp
        sel.ns = "lattice_selected";  sel.id = 0
        sel.type = Marker.LINE_STRIP
        sel.action = Marker.ADD
        sel.scale.x = 0.08
        sel.color.r = 0.0;  sel.color.g = 1.0;  sel.color.b = 0.2;  sel.color.a = 0.9
        for pt in best_path:
            from geometry_msgs.msg import Point
            p = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = 0.1
            sel.points.append(p)
        self.sel_pub.publish(sel)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LatticePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
