#!/usr/bin/env python3
"""
Lattice Planner for F1TENTH
============================
Generates offset paths from a global raceline and selects the best one
based on obstacle clearance.

Key design decisions
--------------------
  • Lidar is filtered to ±135° (front + both sides) so an obstacle beside
    the car is still counted — prevents the false "Cleared" that occurred
    when the obstacle left the forward-only FOV while still blocking the path.
  • Obstacle points = lidar returns within track_half_width of the global path.
    Points further away are static walls and ignored for avoidance.
  • Side-lock: once committed to a lateral side, only same-sign offsets are
    considered on re-plan to prevent left↔right oscillation mid-pass.
  • Clearing only fires when BOTH the committed path AND the centerline are
    free in the ±135° scan — guaranteeing the obstacle is truly past.
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
        # Physical / track
        self.declare_parameter("waypoints_path",        "")
        self.declare_parameter("max_offset",            1.0)   # m — max lateral shift
        self.declare_parameter("safety_radius",         0.50)  # car half-width + buffer (merged clearance_margin)
        self.declare_parameter("track_half_width",      0.70)  # m — obs vs wall split
        # Planning
        self.declare_parameter("plan_horizon",          2.5)   # m ahead
        self.declare_parameter("num_offsets",           11)
        # Scoring (deviation & continuity are the useful knobs; smooth/clearance hardcoded)
        self.declare_parameter("w_deviation",           1.0)   # prefer staying near raceline
        self.declare_parameter("w_continuity",          1.5)   # penalise offset changes
        # Pure pursuit
        self.declare_parameter("min_lookahead",         0.5)
        self.declare_parameter("max_lookahead",         1.8)
        self.declare_parameter("speed_gain",            0.35)
        self.declare_parameter("steer_gain",            1.2)
        self.declare_parameter("steer_limit",           0.41)
        # Safety / speed
        self.declare_parameter("imminent_dist",         0.40)  # m — hard brake threshold
        self.declare_parameter("avoidance_speed_scale", 0.85)  # speed fraction when swerving
        self.declare_parameter("replan_hold_ticks",     20)
        self.declare_parameter("clear_hold_ticks",      15)

        p = lambda name: self.get_parameter(name).value
        self.max_offset          = p("max_offset")
        self.safety_radius       = p("safety_radius")
        self.track_half_width    = p("track_half_width")
        self.plan_horizon        = p("plan_horizon")
        self.num_offsets         = p("num_offsets")
        self.w_deviation         = p("w_deviation")
        self.w_continuity        = p("w_continuity")
        self.min_lookahead       = p("min_lookahead")
        self.max_lookahead       = p("max_lookahead")
        self.speed_gain          = p("speed_gain")
        self.steer_gain          = p("steer_gain")
        self.steer_limit         = p("steer_limit")
        self.imminent_dist       = p("imminent_dist")
        self.avoidance_speed_scale = p("avoidance_speed_scale")
        self.replan_hold_ticks   = int(p("replan_hold_ticks"))
        self.clear_hold_ticks    = int(p("clear_hold_ticks"))

        # Derived constants (not exposed — change safety_radius/imminent_dist instead)
        self._required_clearance    = self.safety_radius
        self._trigger_clearance     = self.safety_radius + 0.15  # earlier state-transition trigger
        self._lateral_check_band    = self.safety_radius + 0.15  # match trigger so wider obs are seen
        self._obs_approach_start    = 4.0 * self.imminent_dist   # begin taper at 4× hard-brake dist
        self._obs_approach_min_frac = 0.5                        # floor speed fraction at approach
        self._max_lat_accel         = 3.0   # m/s² — speed cap from path curvature (hairpin safety)

        # ── Waypoints ─────────────────────────────────────────────────────────
        self.waypoints, self.wp_velocities = self._load_waypoints()
        self.num_pts = len(self.waypoints)
        self.normals = self._calc_normals()
        self.get_logger().info(f"Loaded {self.num_pts} global waypoints")

        # ── State ─────────────────────────────────────────────────────────────
        self.scan_pts_car     = None
        self.prev_nearest     = 0
        self.current_speed    = 0.0
        self.committed_offset = 0.0
        self.clear_ticks      = 0
        self.replan_ticks     = 0
        self.prev_steer       = 0.0

        self.offsets = np.linspace(-self.max_offset, self.max_offset, self.num_offsets)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.scan_sub  = self.create_subscription(LaserScan, "/scan",             self._scan_cb, 10)
        self.odom_sub  = self.create_subscription(Odometry,  "/ego_racecar/odom", self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive",               10)
        self.cand_pub  = self.create_publisher(MarkerArray,  "/lattice/candidates",           10)
        self.sel_pub   = self.create_publisher(Marker,       "/lattice/selected",             10)
        self.obs_pub   = self.create_publisher(MarkerArray,  "/lattice/obstacles",            10)

        self.get_logger().info("Lattice planner ready.")

    # ── Waypoint loading ─────────────────────────────────────────────────────

    def _load_waypoints(self):
        path = str(self.get_parameter("waypoints_path").value)
        if not path:
            pkg  = get_package_share_directory("f1tenth_controller")
            path = os.path.join(pkg, "path", "path_v_centerline.yaml")
        self.get_logger().info(f"Loading waypoints: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        wps = data["waypoints"]
        xy  = np.array([[w["x"], w["y"]] for w in wps])
        vel = np.array([w.get("v", 2.0) for w in wps])
        return xy, vel

    def _calc_normals(self):
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
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        r      = np.array(msg.ranges, dtype=np.float32)
        valid  = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        r, a   = r[valid], angles[valid]
        self.scan_pts_car = np.column_stack([r * np.cos(a), r * np.sin(a)])

    # ── Obstacle map ──────────────────────────────────────────────────────────

    def _build_obs_and_wall_maps(self, R, curr_pos, global_window):
        """
        Split ±135° lidar into obstacle and wall sets:
          • obs_map  = within track_half_width of the raceline (on-track obstacles)
          • wall_map = beyond track_half_width but within +1.5 m (track edges)
        Walls feed the candidate clearance check so wide offsets that crash the
        wall are rejected, but they don't drive the FOLLOWING/AVOIDING state
        transitions (those should react to obstacles only).
        """
        angles    = np.arctan2(self.scan_pts_car[:, 1], self.scan_pts_car[:, 0])
        side_mask = np.abs(angles) <= (3.0 * math.pi / 4.0)   # ±135°
        pts_car   = self.scan_pts_car[side_mask]
        if len(pts_car) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pts_map = (R @ pts_car.T).T + curr_pos
        diff    = pts_map[:, np.newaxis, :] - global_window[np.newaxis, :, :]
        dist    = np.linalg.norm(diff, axis=2).min(axis=1)
        obs     = pts_map[dist <= self.track_half_width]
        walls   = pts_map[(dist > self.track_half_width) &
                          (dist < self.track_half_width + 1.5)]
        return obs, walls

    def _candidate_obstacles(self, candidate, obs_map):
        """Keep obs_map points within lateral_check_band of a candidate path."""
        if len(obs_map) == 0:
            return obs_map
        diff   = obs_map[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        d_cand = np.linalg.norm(diff, axis=2).min(axis=1)
        return obs_map[d_cand <= self._lateral_check_band]

    # ── Main odom callback ────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        if self.scan_pts_car is None:
            return

        px  = msg.pose.pose.position.x
        py  = msg.pose.pose.position.y
        qz  = msg.pose.pose.orientation.z
        qw  = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        self.current_speed = msg.twist.twist.linear.x
        curr_pos = np.array([px, py])

        nearest       = self._find_nearest(curr_pos)
        self.prev_nearest = nearest
        window_idx    = self._window_indices(nearest)
        global_window = self.waypoints[window_idx]
        win_normals   = self.normals[window_idx]

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        obs_map, wall_map = self._build_obs_and_wall_maps(R, curr_pos, global_window)

        # Narrow ±30° forward cone for imminent brake and approach taper.
        # A wider x>0.1 filter picks up track walls at corners (40-60° off-heading)
        # and false-triggers both checks. The cone angle excludes those.
        fwd_mask     = self.scan_pts_car[:, 0] > 0.05
        scan_fwd_car = self.scan_pts_car[fwd_mask] if fwd_mask.any() else np.zeros((0, 2))
        if len(scan_fwd_car) > 0:
            cone_ang     = np.abs(np.arctan2(scan_fwd_car[:, 1], scan_fwd_car[:, 0]))
            scan_fwd_car = scan_fwd_car[cone_ang <= (math.pi / 6)]   # ±30°
        imminent = (len(scan_fwd_car) > 0 and
                    float(np.linalg.norm(scan_fwd_car, axis=1).min()) < self.imminent_dist)

        # ── State machine ─────────────────────────────────────────────────────
        all_blocked = False

        # State-transition checks use _trigger_clearance (larger than safety_radius)
        # so avoidance commits earlier, giving the car more distance to swerve.
        # Candidate validity inside _evaluate_candidates still uses _required_clearance
        # so we don't reject otherwise-fine candidates.
        if self.committed_offset == 0.0:
            # FOLLOWING — trigger avoidance if centerline is blocked
            global_anchored = self._anchor_path(global_window, curr_pos)
            global_obs      = self._candidate_obstacles(global_anchored, obs_map)
            if not self._path_clear(global_anchored, global_obs, self._trigger_clearance):
                _, new_offset, all_blocked = self._evaluate_candidates(
                    global_window, win_normals, obs_map, wall_map, curr_pos)
                if not all_blocked:
                    self.committed_offset = new_offset
                    self.clear_ticks  = 0
                    self.replan_ticks = 0
                    self.get_logger().info(
                        f"Obstacle — locked offset {new_offset:+.2f} m",
                        throttle_duration_sec=0.3)
        else:
            # AVOIDING
            self.replan_ticks += 1

            committed_path     = global_window + win_normals * self.committed_offset
            committed_anchored = self._anchor_path(committed_path, curr_pos)
            committed_obs      = self._candidate_obstacles(committed_anchored, obs_map)

            if not self._path_clear(committed_anchored, committed_obs, self._trigger_clearance):
                # Re-plan after hold (prevents rapid oscillation)
                if self.replan_ticks >= self.replan_hold_ticks:
                    _, new_offset, all_blocked = self._evaluate_candidates(
                        global_window, win_normals, obs_map, wall_map, curr_pos)
                    if not all_blocked:
                        self.committed_offset = new_offset
                        self.clear_ticks  = 0
                        self.replan_ticks = 0
                        self.get_logger().info(
                            f"Re-plan — new offset {new_offset:+.2f} m",
                            throttle_duration_sec=0.3)
            else:
                # Committed path free — also require centerline free before clearing.
                # With ±135° scan, this correctly keeps the offset while the obstacle
                # is still beside the car (it shows up in obs_map even when not ahead).
                global_anchored = self._anchor_path(global_window, curr_pos)
                global_obs      = self._candidate_obstacles(global_anchored, obs_map)
                if self._path_clear(global_anchored, global_obs, self._trigger_clearance):
                    self.clear_ticks += 1
                    if self.clear_ticks >= self.clear_hold_ticks:
                        self.committed_offset = 0.0
                        self.clear_ticks  = 0
                        self.replan_ticks = 0
                        self.get_logger().info("Cleared — back to centerline")
                else:
                    self.clear_ticks = 0

        # ── Drive command ─────────────────────────────────────────────────────
        best_path = self._anchor_path(
            global_window + win_normals * self.committed_offset, curr_pos)
        steer, speed = self._pure_pursuit(curr_pos, yaw, best_path, nearest)

        # Steering damping. 0.5 leaves visible heading shake at speed; 0.4 is
        # the sweet spot between hairpin responsiveness and straight-line stability.
        alpha = 0.3
        steer = alpha * steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = steer

        # Scale speed down proportional to lateral deviation from raceline
        if self.committed_offset != 0.0:
            ratio = abs(self.committed_offset) / max(self.max_offset, 1e-6)
            speed *= 1.0 - (1.0 - self.avoidance_speed_scale) * ratio

        # Emergency: all same-side candidates blocked
        if all_blocked:
            speed *= 0.15
            self.get_logger().warn("All candidates blocked — braking",
                                   throttle_duration_sec=1.0)

        # Curvature speed cap — only fires for genuinely tight curves
        # (radius < 5 m). Below that threshold, kappa fluctuates tick-to-tick
        # as the window slides forward, which jitters speed → lookahead →
        # steering target → heading. The raceline velocities already handle
        # gentle corners, so let the cap only protect against hairpins.
        kappa = self._path_max_kappa(best_path)
        if kappa > 0.15:
            v_curve = math.sqrt(self._max_lat_accel / kappa)
            speed = min(speed, v_curve)

        # Hard cap when any forward point is very close
        if imminent:
            speed = min(speed, 0.3)

        # Smooth taper as forward obstacle approaches
        if len(scan_fwd_car) > 0:
            min_fwd = float(np.linalg.norm(scan_fwd_car, axis=1).min())
            if min_fwd < self._obs_approach_start:
                t     = max(0.0, (min_fwd - self.imminent_dist) /
                            (self._obs_approach_start - self.imminent_dist))
                frac  = self._obs_approach_min_frac + (1.0 - self._obs_approach_min_frac) * t
                speed *= frac

        drive = AckermannDriveStamped()
        drive.header.stamp         = self.get_clock().now().to_msg()
        drive.drive.speed          = float(max(speed, 0.0))
        drive.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive)

        self._publish_markers(global_window, win_normals, best_path, nearest, obs_map)

    # ── Nearest waypoint ─────────────────────────────────────────────────────

    def _find_nearest(self, pos):
        start = self.prev_nearest
        idxs  = [(start + i) % self.num_pts for i in range(-5, 60)]
        pts   = self.waypoints[idxs]
        return idxs[np.argmin(np.linalg.norm(pts - pos, axis=1))]

    def _window_indices(self, start):
        idxs = []
        arc, i = 0.0, start
        for _ in range(self.num_pts):
            idxs.append(i)
            nxt = (i + 1) % self.num_pts
            arc += np.linalg.norm(self.waypoints[nxt] - self.waypoints[i])
            i = nxt
            if arc >= self.plan_horizon:
                break
        return np.array(idxs)

    # ── Candidate scoring ─────────────────────────────────────────────────────

    def _evaluate_candidates(self, global_win, win_normals, obs_map, wall_map, curr_pos):
        # Side-lock: once committed to a side, only consider same-sign offsets.
        # Prevents left↔right oscillation that overshoots through the obstacle.
        committed_sign = (1 if self.committed_offset > 0
                          else -1 if self.committed_offset < 0 else 0)
        offsets_to_try = (list(self.offsets) if committed_sign == 0
                          else [o for o in self.offsets if o * committed_sign >= 0])

        best_path, best_offset, best_score = global_win.copy(), 0.0, float("inf")
        all_blocked = True
        fallback_path, fallback_offset, fallback_clr = global_win.copy(), 0.0, 0.0

        for offset in offsets_to_try:
            candidate = global_win + win_normals * offset
            anchored  = self._anchor_path(candidate, curr_pos)
            cand_obs  = self._candidate_obstacles(anchored, obs_map)
            cand_walls = self._candidate_obstacles(anchored, wall_map)

            obs_clr  = self._path_min_clearance(anchored, cand_obs)
            wall_clr = self._path_min_clearance(anchored, cand_walls)
            clearance = min(obs_clr, wall_clr)

            # Hard reject: candidate too close to either an obstacle or a wall.
            if clearance < self._required_clearance:
                if clearance > fallback_clr:
                    fallback_clr, fallback_path, fallback_offset = clearance, anchored, offset
                continue

            all_blocked = False
            surplus = obs_clr - self._required_clearance
            # Soft wall penalty: scales with how short of safety_radius the
            # actual lidar-detected wall is. wall_clr=safety_radius → 0 penalty.
            wall_short = max(0.0, self.safety_radius + 0.10 - wall_clr)
            score = (self.w_deviation  * abs(offset)
                   + 0.3               * self._curvature_cost(candidate)
                   + 4.0               * max(0.0, 0.6 - surplus) ** 2
                   + 15.0              * wall_short ** 2
                   + self.w_continuity * abs(offset - self.committed_offset))
            if score < best_score:
                best_score, best_path, best_offset = score, anchored, offset

        if all_blocked:
            self.get_logger().warn(
                f"All {self.num_offsets} candidates blocked. "
                f"Best fallback clearance: {fallback_clr:.3f} m. "
                f"Tune: reduce safety_radius ({self.safety_radius:.2f} m) or "
                f"increase track_half_width ({self.track_half_width:.2f} m).",
                throttle_duration_sec=2.0)
            return fallback_path, fallback_offset, True

        return best_path, best_offset, False

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _anchor_path(self, path, curr_pos):
        # Light blend (2 points) — heavy blending (5 pts) flattened the start of
        # the offset path, so pure pursuit's lookahead landed in the straightened
        # zone instead of seeing the planned swerve. Result: corner-cutting at
        # hairpin avoidance. Two points is enough to avoid steering snap.
        anchored  = path.copy()
        blend_pts = min(2, len(anchored))
        for k in range(blend_pts):
            t = (k + 1) / (blend_pts + 1)
            anchored[k] = (1.0 - t) * curr_pos + t * anchored[k]
        return anchored

    def _densify_path(self, path, max_step=0.08):
        if len(path) < 2:
            return path
        samples = [path[0]]
        for i in range(len(path) - 1):
            a, b  = path[i], path[i + 1]
            steps = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / max_step)))
            for j in range(1, steps + 1):
                t = j / steps
                samples.append((1.0 - t) * a + t * b)
                if len(samples) >= 200:
                    return np.asarray(samples)
        return np.asarray(samples)

    def _path_min_clearance(self, path, obs):
        if len(obs) == 0:
            return float("inf")
        dense = self._densify_path(path)
        diff  = dense[:, np.newaxis, :] - obs[np.newaxis, :, :]
        return float(np.linalg.norm(diff, axis=2).min())

    def _path_clear(self, path, obs, clearance=None):
        if clearance is None:
            clearance = self._required_clearance
        return self._path_min_clearance(path, obs) >= clearance

    def _curvature_cost(self, path):
        n, total = len(path), 0.0
        for i in range(1, n - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            ab, bc, ac = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(c-a)
            cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            denom = ab * bc * ac
            if denom > 1e-9:
                total += (2.0 * cross / denom) ** 2
        return total

    def _path_max_kappa(self, path, start=5, n_check=10):
        """
        Max menger curvature in path segments [start, start+n_check).
        Skips the first `start` points because _anchor_path blends them toward
        the car position, producing artificial high curvature in the transition
        zone. We want the curvature of the actual planned arc, not the blend.
        """
        kappa = 0.0
        end = min(start + n_check, len(path) - 1)
        for i in range(max(start, 1), end):
            a, b, c = path[i - 1], path[i], path[i + 1]
            ab, bc, ac = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(c-a)
            denom = ab * bc * ac
            if denom < 1e-9:
                continue
            cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            k = 2.0 * cross / denom
            if k > kappa:
                kappa = k
        return kappa

    # ── Pure pursuit ──────────────────────────────────────────────────────────

    def _pure_pursuit(self, pos, yaw, path, nearest_idx):
        L = np.clip(abs(self.current_speed) * self.speed_gain,
                    self.min_lookahead, self.max_lookahead)
        cy, sy = math.cos(-yaw), math.sin(-yaw)

        # Find first path point past arc-length L that is also in front of the
        # car (local_x > 0). At a hairpin the raceline wraps, so a naive
        # "first point past L" can land beside or behind the car and produce
        # garbage steering.
        target = path[-1]
        arc = 0.0
        for i in range(len(path) - 1):
            arc += np.linalg.norm(path[i + 1] - path[i])
            pt = path[i + 1]
            dx, dy = pt[0] - pos[0], pt[1] - pos[1]
            if arc >= L and (cy*dx - sy*dy) > 0.05:
                target = pt
                break

        dx, dy = target[0] - pos[0], target[1] - pos[1]
        local_x, local_y = cy*dx - sy*dy, sy*dx + cy*dy
        L_act = max(math.hypot(local_x, local_y), 1e-6)
        steer = float(np.clip(self.steer_gain * 2.0 * local_y / L_act**2,
                               -self.steer_limit, self.steer_limit))
        return steer, float(self.wp_velocities[nearest_idx])

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_markers(self, global_win, win_normals, best_path, nearest_idx, obs_map):
        from geometry_msgs.msg import Point
        stamp = self.get_clock().now().to_msg()

        ma = MarkerArray()
        for j, offset in enumerate(self.offsets):
            cand = global_win + win_normals * offset
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns = "lattice_candidates"; m.id = j
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = 0.03
            m.color.r = m.color.g = m.color.b = 0.5; m.color.a = 0.4
            for pt in cand:
                p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
                m.points.append(p)
            ma.markers.append(m)
        self.cand_pub.publish(ma)

        sel = Marker()
        sel.header.frame_id = "map"; sel.header.stamp = stamp
        sel.ns = "lattice_selected"; sel.id = 0
        sel.type = Marker.LINE_STRIP; sel.action = Marker.ADD
        sel.scale.x = 0.08
        sel.color.r = 0.0; sel.color.g = 1.0; sel.color.b = 0.2; sel.color.a = 0.9
        for pt in best_path:
            p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
            sel.points.append(p)
        self.sel_pub.publish(sel)

        obs_ma = MarkerArray()
        clr = Marker(); clr.header.frame_id = "map"; clr.header.stamp = stamp
        clr.ns = "lattice_obstacles"; clr.id = 0; clr.action = Marker.DELETEALL
        obs_ma.markers.append(clr)
        for k, pt in enumerate(obs_map[:60]):
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns = "lattice_obstacles"; m.id = k + 1
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                float(pt[0]), float(pt[1]), 0.15
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color.r = 1.0; m.color.g = 0.1; m.color.b = 0.1; m.color.a = 0.85
            obs_ma.markers.append(m)
        self.obs_pub.publish(obs_ma)


# ── Entry point ───────────────────────────────────────────────────────────────

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
