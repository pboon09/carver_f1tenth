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
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       DurabilityPolicy, ReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray


class LatticePlanner(Node):

    def __init__(self):
        super().__init__("lattice_planner")

        # ── Parameters ───────────────────────────────────────────────────────
        # ── F1TENTH mechanical limits (match gap_follow.py / pure_pursuit.py
        # so all three controllers share one source of truth for the robot).
        # car_width:           0.33 m  — physical track width of the car
        # wheelbase:           0.30 m  — distance between front & rear axles
        # max_steering_angle:  0.4189 rad (24°) — steering rack mechanical limit
        self.declare_parameter("car_width",             0.33)
        self.declare_parameter("wheelbase",             0.30)

        # Physical / track
        self.declare_parameter("waypoints_path",        "")
        # On-car defaults — /dedicate_odom is the pose stream in the map
        # frame from trajectory_publisher (slam_toolbox + EKF). /odometry/filtered
        # is the EKF state for the speed signal (since /dedicate_odom has
        # zero twist). For the gym sim, override to /ego_racecar/odom and
        # leave twist_topic empty so twist comes from the same message.
        self.declare_parameter("odom_topic",   "/dedicate_odom")
        self.declare_parameter("twist_topic",  "/odometry/filtered")
        # Safety clamp on commanded speed (applied after the path + obstacle
        # taper). 0.5 m/s matches the raceline_generator --vmax setting.
        self.declare_parameter("min_speed",    0.0)
        self.declare_parameter("max_speed",    0.5)
        # Max lateral shift from raceline for candidate paths. Set to the
        # WIDE-section ceiling (not the min), and let the per-candidate
        # wall_clr penalty reject the wide-swing options on tight sections.
        # Beam: right-wall ray-cast hits 0.75 m, so 0.40 m gives the planner
        # real room to choose on the wide curves while staying safe at the
        # 0.38 m narrows (those candidates get rejected for low clearance).
        self.declare_parameter("max_offset",            0.45)
        # Extra clearance beyond the car's physical edge. The effective
        # safety_radius used by collision checks is car_width/2 + safety_buffer.
        # 0.34 m worked on wide sim tracks but is impossible inside a 0.5 m
        # half-corridor — set it to a small physical margin (5 cm) here.
        self.declare_parameter("safety_buffer",         0.045)
        # Lidar returns within this distance of the raceline are classified
        # as on-track OBSTACLES; further returns are walls. Must be SMALLER
        # than the corridor half-width or every wall return becomes an
        # obstacle and the planner brakes constantly. But too small misses
        # obstacles that sit just off the line, classifying them as walls
        # (much weaker score penalty) so the planner picks the centerline
        # path right next to the obstacle. 0.40 m works for Nine_clean
        # (corridor half-width ≈ 0.6 m). Drop to 0.25 m on the tighter beam
        # corridor (half-width ≈ 0.5 m) via `ros2 param set` if needed.
        self.declare_parameter("track_half_width",      0.40)
        # Planning
        self.declare_parameter("plan_horizon",          1.5)   # m ahead
        self.declare_parameter("num_offsets",           9)     # 10 cm spacing across ±0.40 m
        # Scoring (deviation & continuity are the useful knobs; smooth/clearance hardcoded)
        self.declare_parameter("w_deviation",           1.0)   # prefer staying near raceline
        self.declare_parameter("w_continuity",          1.5)   # penalise offset changes
        # Pure pursuit
        self.declare_parameter("min_lookahead",         0.8)
        self.declare_parameter("max_lookahead",         1.5)
        self.declare_parameter("speed_gain",            0.35)
        # Bicycle-model responsiveness multiplier — see _pure_pursuit() below.
        # 1.0 = physical truth (steer = atan(wheelbase × κ), maxes out at 21°
        # on this map's κ=1.3 rad/m corner — just inside the 24° limit).
        # Higher values saturate the steering ("bang-bang") through corners.
        # 1.2 gives a small responsiveness boost without saturating.
        self.declare_parameter("steer_gain",            1.2)
        self.declare_parameter("steer_limit",           0.4189)  # F1TENTH mechanical limit = 24°
        # Safety / speed
        # Hard-stop threshold (forward cone lidar). 0.30 m suits a ~1 m wide
        # SLAM track — wider (e.g. 0.40) routinely catches the side walls
        # of curving corridors and deadlocks the car at startup.
        self.declare_parameter("imminent_dist",         0.30)
        # Half-angle (rad) of the forward cone used by imminent + approach
        # taper. π/9 ≈ 20°. Narrower than ±30° so side-walls in a curving
        # corridor don't trigger the brake.
        self.declare_parameter("imminent_cone_rad",     math.pi / 9)
        self.declare_parameter("avoidance_speed_scale", 0.85)  # speed fraction when swerving
        self.declare_parameter("replan_hold_ticks",     20)
        self.declare_parameter("clear_hold_ticks",      15)

        # Steering low-pass (matches pure_pursuit): higher = more responsive,
        # lower = smoother. 0.4 is a good balance for slow tight-corner driving.
        self.declare_parameter("steer_smooth_weight",   0.4)

        # ── One-lap auto-stop (mirrors pure_pursuit) ──
        # Subscribes to /initialpose (RViz "2D Pose Estimate"). Once the car
        # has driven >lap_far_threshold m away from that point and returned
        # within lap_close_threshold m, commanded speed is forced to 0.
        # Republishing /initialpose resets the detector.
        self.declare_parameter("lap_topic",             "/initialpose")
        self.declare_parameter("lap_far_threshold",     2.0)
        self.declare_parameter("lap_close_threshold",   0.6)
        # Grace period after first re-entering close zone before lap_done
        # fires — gives the car a clean overshoot past the start point.
        self.declare_parameter("lap_dwell_seconds",     3.0)

        p = lambda name: self.get_parameter(name).value
        self.car_width           = float(p("car_width"))
        self.wheelbase           = float(p("wheelbase"))
        self.odom_topic          = str(p("odom_topic"))
        self.twist_topic         = str(p("twist_topic"))
        self.min_speed_cap       = float(p("min_speed"))
        self.max_speed_cap       = float(p("max_speed"))
        self.max_offset          = p("max_offset")
        self.safety_buffer       = float(p("safety_buffer"))
        # Effective collision-clearance = car physical half-width + tunable buffer.
        # Wires car_width into the algorithm so changing the mechanical constant
        # automatically tightens/loosens obstacle checks.
        self.safety_radius       = self.car_width / 2.0 + self.safety_buffer
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
        self.imminent_cone_rad   = float(p("imminent_cone_rad"))
        self.avoidance_speed_scale = p("avoidance_speed_scale")
        self.replan_hold_ticks   = int(p("replan_hold_ticks"))
        self.clear_hold_ticks    = int(p("clear_hold_ticks"))
        self.steer_smooth_weight = float(p("steer_smooth_weight"))
        self.lap_topic           = str(p("lap_topic"))
        self.lap_far_threshold   = float(p("lap_far_threshold"))
        self.lap_close_threshold = float(p("lap_close_threshold"))
        self.lap_dwell_seconds   = float(p("lap_dwell_seconds"))
        # Lap-stop and direction-detect state (matches pure_pursuit).
        self.lap_start: object   = None    # np.ndarray when set
        self.lap_far             = False
        self.lap_done            = False
        # ROS clock time when the car first re-entered the close zone after
        # passing the far threshold. lap_done fires lap_dwell_seconds later.
        self.lap_complete_at     = None
        self._direction_checked  = False
        # External pause: set by the launcher via /controller_pause topic.
        # Path/markers keep publishing; only the commanded speed is forced 0.
        self.paused              = False

        # Derived constants (not exposed — change safety_radius/imminent_dist instead)
        # Trigger/check bands sit slightly outside the required clearance so
        # the planner reacts a bit early. Margin = max(safety_buffer, 0.05)
        # so it shrinks together with safety_buffer on narrow tracks (a
        # fixed +0.15 m left no room inside a 0.25 m half-corridor margin).
        trigger_margin              = max(self.safety_buffer, 0.05)
        self._required_clearance    = self.safety_radius
        self._trigger_clearance     = self.safety_radius + trigger_margin
        self._lateral_check_band    = self.safety_radius + trigger_margin
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
        # /scan from RPLidar driver is BEST_EFFORT — must match here or
        # this subscription receives zero messages.
        self.scan_sub  = self.create_subscription(LaserScan, "/scan",             self._scan_cb, qos_profile_sensor_data)
        self.odom_sub  = self.create_subscription(Odometry,  self.odom_topic,     self._odom_cb, 10)
        if self.twist_topic:
            self.twist_sub = self.create_subscription(
                Odometry, self.twist_topic, self._twist_cb, 10
            )
            self.get_logger().info(
                f"speed source: {self.twist_topic} (twist on {self.odom_topic} ignored)"
            )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive",               10)
        self.cand_pub  = self.create_publisher(MarkerArray,  "/lattice/candidates",           10)
        self.sel_pub   = self.create_publisher(Marker,       "/lattice/selected",             10)
        self.obs_pub   = self.create_publisher(MarkerArray,  "/lattice/obstacles",            10)

        if self.lap_far_threshold > 0.0:
            self.lap_sub = self.create_subscription(
                PoseWithCovarianceStamped, self.lap_topic,
                self._lap_cb, 1
            )
            self.get_logger().info(
                f"lap stop: set start with RViz '2D Pose Estimate' on "
                f"{self.lap_topic} (need to travel >{self.lap_far_threshold:.1f} m "
                f"away, return within {self.lap_close_threshold:.1f} m)"
            )

        # External pause from the launcher button (TRANSIENT_LOCAL so we
        # immediately get the latest state on subscribe — even if the
        # launcher published it before we started).
        pause_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pause_sub = self.create_subscription(
            Bool, "/controller_pause", self._pause_cb, pause_qos
        )

        # Register live-tunable params (safety margins, candidate fan, etc.).
        # Changing these via `ros2 param set` triggers _on_param_change which
        # re-derives safety_radius and the offset array immediately.
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info("Lattice planner ready.")

    def _on_param_change(self, params):
        """Recompute derived safety values when car_width or safety_buffer
        are set live via `ros2 param set`. Without this, the cached
        self.safety_radius is frozen at __init__ time so live tweaks of
        wall-collision margin don't take effect."""
        from rcl_interfaces.msg import SetParametersResult
        changed = False
        for p in params:
            if p.name == "car_width":
                self.car_width = float(p.value)
                changed = True
            elif p.name == "safety_buffer":
                self.safety_buffer = float(p.value)
                changed = True
            elif p.name == "track_half_width":
                self.track_half_width = float(p.value)
            elif p.name == "imminent_dist":
                self.imminent_dist = float(p.value)
                self._obs_approach_start = 4.0 * self.imminent_dist
            elif p.name == "imminent_cone_rad":
                self.imminent_cone_rad = float(p.value)
            elif p.name == "max_offset":
                self.max_offset = float(p.value)
                self.offsets = np.linspace(-self.max_offset,
                                            self.max_offset, self.num_offsets)
            elif p.name == "num_offsets":
                self.num_offsets = int(p.value)
                self.offsets = np.linspace(-self.max_offset,
                                            self.max_offset, self.num_offsets)
        if changed:
            self.safety_radius = self.car_width / 2.0 + self.safety_buffer
            trigger_margin = max(self.safety_buffer, 0.05)
            self._required_clearance = self.safety_radius
            self._trigger_clearance = self.safety_radius + trigger_margin
            self._lateral_check_band = self.safety_radius + trigger_margin
            self.get_logger().info(
                f"safety_radius recomputed: {self.safety_radius:.3f} m "
                f"(car_width/2={self.car_width/2:.3f} + buffer={self.safety_buffer:.3f})"
            )
        return SetParametersResult(successful=True)

    def _pause_cb(self, msg: Bool):
        # Toggle external pause. While paused, the drive callback forces
        # commanded speed to 0 but path/marker publishers keep running so
        # the GUI/RViz still shows the planned trajectory.
        if self.paused != msg.data:
            self.get_logger().info(f"pause = {msg.data}")
        self.paused = bool(msg.data)

    def _lap_cb(self, msg: PoseWithCovarianceStamped):
        # Each /initialpose message defines a NEW lap-start point and resets
        # the lap-complete detector. Republish the arrow to start another lap.
        p = msg.pose.pose.position
        self.lap_start = np.array([p.x, p.y])
        self.lap_far = False
        self.lap_done = False
        self.lap_complete_at = None
        self.get_logger().info(
            f"lap start ← ({p.x:+.2f}, {p.y:+.2f})  "
            f"(need: travel >{self.lap_far_threshold:.1f} m away, "
            f"return within {self.lap_close_threshold:.1f} m, "
            f"then {self.lap_dwell_seconds:.1f} s grace → STOP)"
        )

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
        # Use a densified centerline so the obs/wall split doesn't change
        # discontinuously at waypoint boundaries (a point right between two
        # raw waypoints can otherwise be measured 0.15 m farther than it
        # actually is from the path line).
        dense_window = self._densify_path(global_window)
        diff    = pts_map[:, np.newaxis, :] - dense_window[np.newaxis, :, :]
        dist    = np.linalg.norm(diff, axis=2).min(axis=1)
        obs     = pts_map[dist <= self.track_half_width]
        walls   = pts_map[(dist > self.track_half_width) &
                          (dist < self.track_half_width + 1.5)]
        return obs, walls

    def _candidate_obstacles(self, candidate, obs_map):
        """Keep obs_map points within lateral_check_band of a candidate path.
        Densifies the candidate before measuring — at the raw 0.3 m waypoint
        spacing an obstacle sitting between two waypoints could land outside
        the band on the min-to-waypoint check while the actual path-line
        passes right through it. The downstream clearance check uses the
        densified path too, so this keeps the two consistent."""
        if len(obs_map) == 0:
            return obs_map
        dense  = self._densify_path(candidate)
        diff   = obs_map[:, np.newaxis, :] - dense[np.newaxis, :, :]
        d_cand = np.linalg.norm(diff, axis=2).min(axis=1)
        return obs_map[d_cand <= self._lateral_check_band]

    # ── Main odom callback ────────────────────────────────────────────────────

    def _twist_cb(self, msg: Odometry):
        # Separate twist source — used when odom_topic carries pose but no
        # twist (e.g. /dedicate_odom in map frame).
        self.current_speed = msg.twist.twist.linear.x

    def _odom_cb(self, msg: Odometry):
        if self.scan_pts_car is None:
            return

        px  = msg.pose.pose.position.x
        py  = msg.pose.pose.position.y
        qz  = msg.pose.pose.orientation.z
        qw  = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        if not self.twist_topic:
            self.current_speed = msg.twist.twist.linear.x
        curr_pos = np.array([px, py])

        # Path direction auto-detect (one-shot, mirrors pure_pursuit). If the
        # car is facing opposite the loop's stored ordering, flip the path.
        if not self._direction_checked:
            n_now = int(np.argmin(np.linalg.norm(self.waypoints - curr_pos, axis=1)))
            nxt = (n_now + 1) % self.num_pts
            pv = self.waypoints[nxt] - self.waypoints[n_now]
            path_yaw = math.atan2(pv[1], pv[0])
            yd = (yaw - path_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yd) > math.pi / 2:
                self.waypoints = self.waypoints[::-1]
                self.wp_velocities = self.wp_velocities[::-1]
                self.normals = self._calc_normals()
                self.prev_nearest = 0
                self.get_logger().warn(
                    f"Path direction reversed at startup "
                    f"(Δ={math.degrees(yd):+.0f}°)"
                )
            else:
                self.get_logger().info(
                    f"Path direction matches car heading "
                    f"(Δ={math.degrees(yd):+.0f}°)"
                )
            self._direction_checked = True

        # Auto-record lap start on first odom message so the detector works
        # even if the user doesn't manually click '2D Pose Estimate' first.
        # /initialpose still overrides this (resets lap_start, lap_far, lap_done).
        if self.lap_start is None and self.lap_far_threshold > 0.0:
            self.lap_start = curr_pos.copy()
            self.get_logger().info(
                f"lap start (auto) ← ({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f})  "
                f"— set 2D Pose Estimate in RViz to override"
            )

        # Lap-complete detector with overshoot grace period.
        if self.lap_start is not None and not self.lap_done:
            d = float(np.linalg.norm(curr_pos - self.lap_start))
            if not self.lap_far and d > self.lap_far_threshold:
                self.lap_far = True
                self.get_logger().info(
                    f"lap: passed far threshold (d={d:.2f} m), watching for return"
                )
            elif self.lap_far and self.lap_complete_at is None \
                    and d < self.lap_close_threshold:
                self.lap_complete_at = self.get_clock().now()
                self.get_logger().info(
                    f"lap: returned within {d:.2f} m of start — "
                    f"stopping in {self.lap_dwell_seconds:.1f} s"
                )
            elif self.lap_complete_at is not None:
                elapsed = (self.get_clock().now()
                           - self.lap_complete_at).nanoseconds / 1e9
                if elapsed >= self.lap_dwell_seconds:
                    self.lap_done = True
                    self.get_logger().warn(
                        f"LAP COMPLETE — stopping "
                        f"(grace {elapsed:.1f} s after close detect)"
                    )

        nearest       = self._find_nearest(curr_pos)
        self.prev_nearest = nearest
        window_idx    = self._window_indices(nearest)
        global_window = self.waypoints[window_idx]
        win_normals   = self.normals[window_idx]

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        obs_map, wall_map = self._build_obs_and_wall_maps(R, curr_pos, global_window)

        # Forward cone (configurable; default ±20°) for imminent brake and
        # approach taper. A wider cone picks up the side walls of a curving
        # corridor and deadlocks the car at startup — keep it tight.
        fwd_mask     = self.scan_pts_car[:, 0] > 0.05
        scan_fwd_car = self.scan_pts_car[fwd_mask] if fwd_mask.any() else np.zeros((0, 2))
        if len(scan_fwd_car) > 0:
            cone_ang     = np.abs(np.arctan2(scan_fwd_car[:, 1], scan_fwd_car[:, 0]))
            scan_fwd_car = scan_fwd_car[cone_ang <= self.imminent_cone_rad]
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

        # Steering low-pass (tuneable via steer_smooth_weight — matches
        # pure_pursuit). Higher = more responsive, lower = smoother. 0.4
        # is the sweet spot for slow tight-corner driving.
        alpha = self.steer_smooth_weight
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

        # Hard stop when any forward-cone (±30°) lidar return is closer than
        # imminent_dist (default 0.40 m). Matches the pure_pursuit safety
        # stop — full brake until the path clears, not a slowdown. The
        # `imminent` flag is recomputed each tick from current lidar, so
        # the car auto-resumes when the obstacle is gone.
        if imminent:
            speed = 0.0
            self.get_logger().warn(
                f"EMERGENCY STOP — obstacle within {self.imminent_dist:.2f} m",
                throttle_duration_sec=1.0,
            )

        # Smooth taper as forward obstacle approaches
        if len(scan_fwd_car) > 0:
            min_fwd = float(np.linalg.norm(scan_fwd_car, axis=1).min())
            if min_fwd < self._obs_approach_start:
                t     = max(0.0, (min_fwd - self.imminent_dist) /
                            (self._obs_approach_start - self.imminent_dist))
                frac  = self._obs_approach_min_frac + (1.0 - self._obs_approach_min_frac) * t
                speed *= frac

        # Lap-complete / external-pause override: hold the wheel steady at
        # current steering but command zero speed. Lap-done clears only on
        # next /initialpose; pause clears when launcher publishes False.
        if self.lap_done or self.paused:
            speed = 0.0

        drive = AckermannDriveStamped()
        drive.header.stamp         = self.get_clock().now().to_msg()
        drive.drive.speed          = float(np.clip(speed, self.min_speed_cap,
                                                          self.max_speed_cap))
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
            # Use the same threshold for both — on a narrow corridor, a stricter
            # obstacle requirement causes every candidate to be rejected.
            if clearance < self._required_clearance:
                if clearance > fallback_clr:
                    fallback_clr, fallback_path, fallback_offset = clearance, anchored, offset
                continue

            all_blocked = False
            surplus = obs_clr - self._required_clearance
            # Soft wall penalty: scales with how short of safety_radius the
            # actual lidar-detected wall is. wall_clr=safety_radius → 0 penalty.
            # Penalty kicks in when wall_clr drops within safety_radius + 0.20
            # (was +0.10) — gives the planner more headroom to react to
            # close walls / mis-classified obstacles before they're imminent.
            wall_short = max(0.0, self.safety_radius + 0.20 - wall_clr)
            # Strong obstacle-clearance penalty — pushes the planner to pick a
            # candidate that gives the obstacle a wide berth from the FIRST
            # decision instead of picking marginal clearance and re-correcting.
            # 25 × (0.8 − surplus)² grows fast: surplus=0.3 → penalty=6.25,
            # surplus=0.6 → penalty=1.0, surplus≥0.8 → penalty=0.
            score = (self.w_deviation  * abs(offset)
                   + 0.3               * self._curvature_cost(candidate)
                   + 25.0              * max(0.0, 0.8 - surplus) ** 2
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

        # Find first path segment whose accumulated arc exceeds L, then
        # interpolate INSIDE that segment to land exactly L metres ahead.
        # Snapping to the next discrete waypoint (the previous behaviour)
        # introduced 0.3 m step-changes in the target → step-changes in
        # steering → "bang-bang" feel on tight corners. Also require the
        # landed target to be in front of the car (local_x > 0.05) so a
        # path that wraps at a hairpin doesn't pick a point beside/behind.
        target = path[-1]
        arc = 0.0
        for i in range(len(path) - 1):
            seg = path[i + 1] - path[i]
            seg_len = float(np.linalg.norm(seg))
            if arc + seg_len >= L and seg_len > 1e-9:
                t = (L - arc) / seg_len
                candidate = path[i] + t * seg
                dx, dy = candidate[0] - pos[0], candidate[1] - pos[1]
                if (cy*dx - sy*dy) > 0.05:
                    target = candidate
                    break
            arc += seg_len

        dx, dy = target[0] - pos[0], target[1] - pos[1]
        local_x, local_y = cy*dx - sy*dy, sy*dx + cy*dy
        L_act = max(math.hypot(local_x, local_y), 1e-6)
        # Bicycle-model pure pursuit: steer = atan(L * κ) where κ = 2y/Ld².
        # steer_gain is a responsiveness multiplier on top of the physical
        # bicycle output (1.0 = pure model truth, higher = aggressive). The
        # default (steer_gain=4.0 × wheelbase=0.30 = 1.2) preserves the
        # behaviour of the previous `1.2 × κ` formula.
        curvature = 2.0 * local_y / (L_act ** 2)
        steer = float(np.clip(
            self.steer_gain * math.atan(self.wheelbase * curvature),
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
