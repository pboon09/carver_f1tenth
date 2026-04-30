#!/usr/bin/env python3
"""
Lattice Planner for F1TENTH  — v3
===================================
Root-cause fix for "all candidates blocked" when walls are on both sides
------------------------------------------------------------------------

The previous versions computed _path_min_clearance as the minimum distance
from ANY path point to ANY lidar point.  On a narrow track the walls are
ALWAYS within safety_radius of some offset candidates, so every candidate
was rejected and the planner braked every lap.

Fix: per-candidate clearance checks use ONLY lidar points that are closer
than track_half_width to the global centerline.  Wall hits that are
laterally BEYOND the candidate never count against it.

Architecture change
-------------------
  Old: score(candidate) uses global scan_map (all forward points in corridor)
  New:
    • obs_map  — lidar pts whose distance from the global path ≤ track_half_width
                  These are obstacles ON the track.  Only these block candidates.
    • Wall hits (beyond track_half_width) are ignored for avoidance.
    • Per-candidate: also filter obs_map to points within lateral_check_band of
                     the candidate, so a left-shifted path is not penalised by
                     the right wall.

Parameters added
----------------
  track_half_width  : half the usable track width (m).  Points beyond this from
                      the centerline are classified as walls and ignored.
                      Set to ~90% of the actual half-width.
                      e.g. 1.0 m wide track → set 0.45
  lateral_check_band: how far either side of a candidate to include obstacle
                      points in its clearance check.
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
        self.declare_parameter("waypoints_path",     "")
        self.declare_parameter("plan_horizon",       2.5)
        self.declare_parameter("num_offsets",        11)
        self.declare_parameter("max_offset",         1.0)
        self.declare_parameter("safety_radius",      0.35)   # car half-width + margin
        self.declare_parameter("clearance_margin",   0.15)
        # KEY PARAM — measure your track and set to ~90% of half-width
        # e.g. track 1.0 m wide → half = 0.5 m → set 0.45
        self.declare_parameter("track_half_width",   0.70)
        # lateral_check_band: keep narrow — just enough to catch the obstacle
        # near the candidate, but not so wide it catches the opposite wall.
        # Rule of thumb: safety_radius + clearance_margin + 0.05
        self.declare_parameter("lateral_check_band", 0.25)   # ↓ tightened
        self.declare_parameter("w_deviation",        1.0)
        self.declare_parameter("w_smooth",           0.3)
        self.declare_parameter("w_clearance",        2.0)
        self.declare_parameter("w_continuity",       0.8)    # ↑ stronger anti-creep
        self.declare_parameter("min_lookahead",      0.5)
        self.declare_parameter("max_lookahead",      1.8)
        self.declare_parameter("speed_gain",         0.35)
        self.declare_parameter("steer_gain",         1.2)
        self.declare_parameter("steer_limit",        0.41)
        self.declare_parameter("clear_hold_ticks",   15)     # ↑ more hysteresis
        self.declare_parameter("imminent_dist",      0.55)
        # Minimum ticks between successive re-plans while AVOIDING.
        # Stops the -0.40 → -0.60 → -0.80 creep that happens when the car
        # hasn't physically moved far enough to re-evaluate yet.
        self.declare_parameter("replan_hold_ticks",  20)

        # ── Speed tuning knobs ───────────────────────────────────────────────
        # 1. How much to slow when actively swerving around an obstacle.
        #    1.0 = no slowdown, 0.5 = half speed.  Applied as a flat cap
        #    on top of the per-waypoint speed profile.
        #    Tune this first — it is the simplest lever.
        self.declare_parameter("avoidance_speed_scale", 0.80)

        # 2. Distance-based slow-down: starts at obs_approach_slow_start metres
        #    and reaches obs_approach_min_speed fraction at imminent_dist.
        #    These apply whenever an obstacle point is in front, regardless of
        #    whether the planner has committed to an offset yet.
        self.declare_parameter("obs_approach_slow_start", 2.0)  # m — begin braking
        self.declare_parameter("obs_approach_min_speed",  0.50) # fraction of wp speed

        p = lambda name: self.get_parameter(name).value
        self.plan_horizon       = p("plan_horizon")
        self.num_offsets        = p("num_offsets")
        self.max_offset         = p("max_offset")
        self.safety_radius      = p("safety_radius")
        self.clearance_margin   = p("clearance_margin")
        self.track_half_width   = p("track_half_width")
        self.lateral_check_band = p("lateral_check_band")
        self.w_deviation        = p("w_deviation")
        self.w_smooth           = p("w_smooth")
        self.w_clearance        = p("w_clearance")
        self.w_continuity       = p("w_continuity")
        self.min_lookahead      = p("min_lookahead")
        self.max_lookahead      = p("max_lookahead")
        self.speed_gain         = p("speed_gain")
        self.steer_gain         = p("steer_gain")
        self.steer_limit        = p("steer_limit")
        self.clear_hold_ticks        = int(p("clear_hold_ticks"))
        self.imminent_dist           = p("imminent_dist")
        self.replan_hold_ticks       = int(p("replan_hold_ticks"))
        self.avoidance_speed_scale   = p("avoidance_speed_scale")
        self.obs_approach_slow_start = p("obs_approach_slow_start")
        self.obs_approach_min_speed  = p("obs_approach_min_speed")

        # Single shared clearance threshold
        self._required_clearance = self.safety_radius + self.clearance_margin

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
        self.replan_ticks     = 0   # ticks since last re-plan while AVOIDING
        self.prev_steer       = 0.0

        self.offsets = np.linspace(-self.max_offset, self.max_offset, self.num_offsets)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.scan_sub  = self.create_subscription(LaserScan,  "/scan",             self._scan_cb, 10)
        self.odom_sub  = self.create_subscription(Odometry,   "/ego_racecar/odom", self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive",               10)
        self.cand_pub  = self.create_publisher(MarkerArray,  "/lattice/candidates",           10)
        self.sel_pub   = self.create_publisher(Marker,       "/lattice/selected",             10)
        self.obs_pub   = self.create_publisher(MarkerArray,  "/lattice/obstacles",            10)

        self.get_logger().info("Lattice planner v3 ready.")

    # ── Waypoint loading ─────────────────────────────────────────────────────

    def _load_waypoints(self):
        path = str(self.get_parameter("waypoints_path").value)
        if not path:
            pkg  = get_package_share_directory("f1tenth_controller")
            path = os.path.join(pkg, "path", "path_v.yaml")
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

    # ── Wall/obstacle split ───────────────────────────────────────────────────

    def _build_obstacle_map(self, scan_map_fwd: np.ndarray,
                             global_window: np.ndarray) -> np.ndarray:
        """
        Keep only lidar points within track_half_width of the centerline.
        Points further away are wall returns and must not trigger avoidance.

        If the warn fires on an empty straight, track_half_width is too small.
        """
        if len(scan_map_fwd) == 0 or len(global_window) == 0:
            return np.zeros((0, 2))

        diff       = scan_map_fwd[:, np.newaxis, :] - global_window[np.newaxis, :, :]
        dist_path  = np.linalg.norm(diff, axis=2).min(axis=1)
        on_track   = dist_path <= self.track_half_width

        self.get_logger().debug(
            f"lidar: {on_track.sum()} obstacle pts, {(~on_track).sum()} wall pts "
            f"(track_half_width={self.track_half_width:.2f} m)",
            throttle_duration_sec=1.0)

        return scan_map_fwd[on_track]

    def _candidate_obstacles(self, candidate: np.ndarray,
                              obs_map: np.ndarray) -> np.ndarray:
        """Keep only obs_map points within lateral_check_band of candidate."""
        if len(obs_map) == 0:
            return obs_map
        diff   = obs_map[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        d_cand = np.linalg.norm(diff, axis=2).min(axis=1)
        return obs_map[d_cand <= self.lateral_check_band]

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

        nearest = self._find_nearest(curr_pos)
        self.prev_nearest = nearest

        window_idx     = self._window_indices(nearest)
        global_window  = self.waypoints[window_idx]
        window_normals = self.normals[window_idx]

        # Lidar → map frame (forward only)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        fwd_mask     = self.scan_pts_car[:, 0] > 0.1
        scan_fwd_car = self.scan_pts_car[fwd_mask]

        if len(scan_fwd_car) > 0:
            scan_map_fwd = (R @ scan_fwd_car.T).T + curr_pos
        else:
            scan_fwd_car = np.zeros((0, 2))
            scan_map_fwd = np.zeros((0, 2))

        # Split into wall and obstacle points
        obs_map = self._build_obstacle_map(scan_map_fwd, global_window)

        # Imminent-danger brake uses raw car-frame distances (walls included)
        imminent = False
        if len(scan_fwd_car) > 0:
            if float(np.linalg.norm(scan_fwd_car, axis=1).min()) < self.imminent_dist:
                imminent = True

        # ── State machine (runs every tick) ───────────────────────────────────
        all_blocked = False

        if self.committed_offset == 0.0:
            # FOLLOWING — detect obstacle on centerline
            global_anchored = self._anchor_path(global_window, curr_pos)
            global_obs      = self._candidate_obstacles(global_anchored, obs_map)
            if not self._path_clear(global_anchored, global_obs):
                _, new_offset, all_blocked = self._evaluate_candidates(
                    global_window, window_normals, obs_map, curr_pos)
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

            committed_path     = global_window + window_normals * self.committed_offset
            committed_anchored = self._anchor_path(committed_path, curr_pos)
            committed_obs      = self._candidate_obstacles(committed_anchored, obs_map)

            if not self._path_clear(committed_anchored, committed_obs):
                # Committed path blocked — re-plan only after replan_hold_ticks.
                # This prevents the -0.40→-0.60→-0.80 creep: the car needs time
                # to actually move toward the new offset before we check again.
                if self.replan_ticks >= self.replan_hold_ticks:
                    _, new_offset, all_blocked = self._evaluate_candidates(
                        global_window, window_normals, obs_map, curr_pos)
                    if not all_blocked:
                        self.committed_offset = new_offset
                        self.clear_ticks  = 0
                        self.replan_ticks = 0
                        self.get_logger().info(
                            f"Re-plan — new offset {new_offset:+.2f} m",
                            throttle_duration_sec=0.3)
                # else: hold current offset, give it time to work
            else:
                # Committed path safe — check if centerline is also clear
                global_anchored = self._anchor_path(global_window, curr_pos)
                global_obs      = self._candidate_obstacles(global_anchored, obs_map)
                if self._path_clear(global_anchored, global_obs):
                    self.clear_ticks += 1
                    if self.clear_ticks >= self.clear_hold_ticks:
                        self.committed_offset = 0.0
                        self.clear_ticks  = 0
                        self.replan_ticks = 0
                        self.get_logger().info("Cleared — back to centerline")
                else:
                    self.clear_ticks = 0

        best_offset   = self.committed_offset
        best_path_raw = global_window + window_normals * best_offset
        best_path     = self._anchor_path(best_path_raw, curr_pos)

        steer, speed = self._pure_pursuit(curr_pos, yaw, best_path, nearest)

        alpha = 0.6 if best_offset != 0.0 else 0.3
        steer = alpha * steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = steer

        if best_offset != 0.0:
            # ── Layer 1: avoidance lateral-deviation scale ─────────────────
            # Mild reduction proportional to how far we've shifted.
            # At max_offset: speed *= avoidance_speed_scale (e.g. 0.80).
            # At zero offset: no reduction (multiplier = 1.0).
            # Tune avoidance_speed_scale (0.5–1.0):
            #   1.0 → full speed while swerving (may overshoot)
            #   0.6 → noticeably slower through the avoidance arc
            deviation_ratio = abs(best_offset) / max(self.max_offset, 1e-6)
            avoidance_mult  = 1.0 - (1.0 - self.avoidance_speed_scale) * deviation_ratio
            speed *= avoidance_mult

        if all_blocked:
            speed *= 0.15
            self.get_logger().warn("All on-track candidates blocked — braking",
                                   throttle_duration_sec=1.0)

        # ── Layer 2: imminent hard cap ─────────────────────────────────────
        # Any forward lidar point within imminent_dist → cap speed to 0.3 m/s.
        # Tune imminent_dist (default 0.55 m): larger = brakes earlier.
        if imminent:
            speed = min(speed, 0.3)

        # ── Layer 3: smooth distance-based taper ───────────────────────────
        # Begins at obs_approach_slow_start metres from nearest forward point.
        # At that distance: no reduction.  At imminent_dist: speed reduced to
        # obs_approach_min_speed fraction.
        #
        # Tune obs_approach_slow_start (default 2.5 m):
        #   larger → braking begins earlier, more gentle deceleration
        #   smaller → braking begins late, sharper deceleration
        #
        # Tune obs_approach_min_speed (default 0.40):
        #   0.20 → very slow when close (nearly stopped before imminent kicks in)
        #   0.60 → barely slows until imminent_dist takes over
        #
        # This layer is active even on a clear path if something is in front —
        # it is NOT gated on committed_offset.
        if len(scan_fwd_car) > 0:
            min_fwd   = float(np.linalg.norm(scan_fwd_car, axis=1).min())
            slow_start = self.obs_approach_slow_start
            stop_dist  = self.imminent_dist
            if min_fwd < slow_start:
                # t=1.0 at slow_start (no reduction), t=0.0 at stop_dist (min speed)
                t        = max(0.0, (min_fwd - stop_dist) / (slow_start - stop_dist))
                min_frac = self.obs_approach_min_speed
                # Lerp from min_frac to 1.0 as t goes 0→1
                frac     = min_frac + (1.0 - min_frac) * t
                speed    = min(speed, frac * speed)

        drive = AckermannDriveStamped()
        drive.header.stamp         = self.get_clock().now().to_msg()
        drive.drive.speed          = float(max(speed, 0.0))
        drive.drive.steering_angle = float(steer)
        self.drive_pub.publish(drive)

        self._publish_markers(global_window, window_normals, best_path,
                              nearest, obs_map)

    # ── Nearest waypoint ─────────────────────────────────────────────────────

    def _find_nearest(self, pos: np.ndarray) -> int:
        window = 60
        start  = self.prev_nearest
        idxs   = [(start + i) % self.num_pts for i in range(-5, window)]
        pts    = self.waypoints[idxs]
        local  = np.argmin(np.linalg.norm(pts - pos, axis=1))
        return idxs[local]

    def _window_indices(self, start: int):
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

    def _evaluate_candidates(self, global_win, win_normals, obs_map, curr_pos):
        best_path   = global_win.copy()
        best_offset = 0.0
        best_score  = float("inf")
        all_blocked = True
        fallback_path, fallback_offset, fallback_clearance = global_win.copy(), 0.0, 0.0

        for offset in self.offsets:
            candidate          = global_win + win_normals * offset
            candidate_anchored = self._anchor_path(candidate, curr_pos)
            cand_obs           = self._candidate_obstacles(candidate_anchored, obs_map)
            min_clearance      = self._path_min_clearance(candidate_anchored, cand_obs)

            if min_clearance < self._required_clearance:
                if min_clearance > fallback_clearance:
                    fallback_clearance = min_clearance
                    fallback_path      = candidate_anchored
                    fallback_offset    = offset
                continue

            all_blocked = False
            surplus         = min_clearance - self._required_clearance
            score = (self.w_deviation  * abs(offset)
                   + self.w_smooth     * self._curvature_cost(candidate)
                   + self.w_clearance  * max(0.0, 0.4 - surplus) ** 2
                   + self.w_continuity * abs(offset - self.committed_offset)
                   + 0.05 * max(0.0, offset))

            if score < best_score:
                best_score, best_path, best_offset = score, candidate_anchored, offset

        if all_blocked:
            self.get_logger().warn(
                f"All {self.num_offsets} candidates blocked. "
                f"Best fallback clearance: {fallback_clearance:.3f} m. "
                f"If no real obstacle is present, reduce safety_radius or "
                f"increase track_half_width (currently {self.track_half_width:.2f} m).",
                throttle_duration_sec=2.0)
            return fallback_path, fallback_offset, True

        return best_path, best_offset, False

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _anchor_path(self, path: np.ndarray, curr_pos: np.ndarray) -> np.ndarray:
        anchored  = path.copy()
        blend_pts = min(5, len(anchored))
        for k in range(blend_pts):
            t = (k + 1) / (blend_pts + 1)
            anchored[k] = (1.0 - t) * curr_pos + t * anchored[k]
        return anchored

    def _densify_path(self, path: np.ndarray, max_step: float = 0.08) -> np.ndarray:
        if len(path) < 2:
            return path
        MAX_PTS = 200
        samples = [path[0]]
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            steps = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / max_step)))
            for j in range(1, steps + 1):
                t = j / steps
                samples.append((1.0 - t) * a + t * b)
                if len(samples) >= MAX_PTS:
                    return np.asarray(samples)
        return np.asarray(samples)

    def _path_min_clearance(self, path: np.ndarray, obs: np.ndarray) -> float:
        if len(obs) == 0:
            return float("inf")
        dense = self._densify_path(path)
        diff  = dense[:, np.newaxis, :] - obs[np.newaxis, :, :]
        return float(np.linalg.norm(diff, axis=2).min())

    def _path_clear(self, path: np.ndarray, obs: np.ndarray) -> bool:
        return self._path_min_clearance(path, obs) >= self._required_clearance

    def _curvature_cost(self, path: np.ndarray) -> float:
        n = len(path)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(1, n - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            ab, bc, ac = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(c-a)
            cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            denom = ab * bc * ac
            total += (2.0 * cross / denom) ** 2 if denom > 1e-9 else 0.0
        return total

    # ── Pure pursuit ──────────────────────────────────────────────────────────

    def _pure_pursuit(self, pos, yaw, path, nearest_idx):
        L   = np.clip(abs(self.current_speed) * self.speed_gain,
                      self.min_lookahead, self.max_lookahead)
        arc = 0.0
        target = path[-1]
        for i in range(len(path) - 1):
            arc += np.linalg.norm(path[i + 1] - path[i])
            if arc >= L:
                target = path[i + 1]
                break

        dx, dy = target[0] - pos[0], target[1] - pos[1]
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        local_x, local_y = cy*dx - sy*dy, sy*dx + cy*dy

        L_act = max(math.hypot(local_x, local_y), 1e-6)
        steer = float(np.clip(self.steer_gain * 2.0 * local_y / L_act**2,
                               -self.steer_limit, self.steer_limit))
        return steer, float(self.wp_velocities[nearest_idx])

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_markers(self, global_win, win_normals, best_path,
                         nearest_idx, obs_map):
        from geometry_msgs.msg import Point
        stamp = self.get_clock().now().to_msg()

        # Candidate lines
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

        # Selected path
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

        # Obstacle points (red) — add /lattice/obstacles to RViz to debug
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