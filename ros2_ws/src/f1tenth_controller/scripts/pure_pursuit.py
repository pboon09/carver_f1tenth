#!/usr/bin/python3

import math
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       DurabilityPolicy, ReliabilityPolicy)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        # Pure pursuit tracks waypoints; it has no notion of car geometry
        # since it doesn't do obstacle avoidance or clearance checks. The
        # mechanical steering limit IS enforced (see `steering_limit`
        # below = 0.4189 rad = 24°, matching gap_follow + lattice).
        # car_width / wheelbase are NOT declared here on purpose — they
        # don't apply to this algorithm.
        self.declare_parameter("waypoints_path", "")
        # On-car defaults — /dedicate_odom is the pose stream in the map
        # frame from trajectory_publisher (slam_toolbox + EKF). /odometry/filtered
        # is the EKF state for the speed signal (since /dedicate_odom has
        # zero twist). For the gym sim, override to /ego_racecar/odom and
        # leave twist_topic empty so twist comes from the same message.
        self.declare_parameter("odom_topic",  "/dedicate_odom")
        self.declare_parameter("twist_topic", "/odometry/filtered")
        self.declare_parameter("drive_topic", "/drive")

        # Speed
        self.declare_parameter("velocity", 0.5)
        # Safety clamp on the commanded speed (applied after the path-yaml
        # speed profile is sampled). 0.5 m/s matches raceline_generator --vmax.
        self.declare_parameter("min_speed", 0.0)
        self.declare_parameter("max_speed", 0.5)

        # Lookahead — scales linearly with speed between min and max.
        # min_lookahead=0.6 m gives the controller ~2 waypoints of warning
        # before each corner so steering can ramp instead of stepping.
        self.declare_parameter("min_lookahead", 0.6)
        self.declare_parameter("max_lookahead", 1.5)
        self.declare_parameter("min_lookahead_speed", 0.0)
        self.declare_parameter("max_lookahead_speed", 7.0)

        # Steering P-gain — multiplies path curvature κ to get a steering
        # angle command. At κ=1.3 rad/m (the tightest corner on this track)
        # a gain of 0.30 (≈ wheelbase) produces ~21° which lives inside the
        # 24° steering limit → no saturation. Higher gains pin the steering
        # at the limit through every corner ("bang-bang"). Scheduled lower
        # at high speed to suppress oscillation in dynamics.
        self.declare_parameter("min_gain", 0.3)
        self.declare_parameter("max_gain", 0.45)
        self.declare_parameter("gain_speed_scale", 7.0)

        # Derivative gain for damping
        self.declare_parameter("D", 2.0)

        # Steering low-pass: steer = smooth_weight*new + (1-smooth_weight)*prev.
        # Higher = more responsive but jitterier; lower = smoother but laggy.
        # 0.4 is a balance for slow tight-corner driving — enough smoothing
        # to hide the per-waypoint discretization noise without lagging the
        # corner entry.
        self.declare_parameter("steer_smooth_weight", 0.4)

        self.declare_parameter("steering_limit", 24.0)   # degrees (= 0.4189 rad, mechanical limit)
        self.declare_parameter("lookahead_window", 50)   # waypoints to search ahead

        # ── Safety: lidar-based emergency stop ─────────────────────────────
        # If any lidar return inside the forward arc is closer than
        # safety_stop_distance, publish speed=0 until clear. Disable by
        # setting safety_stop_distance <= 0.
        self.declare_parameter("scan_topic",            "/scan")
        self.declare_parameter("safety_stop_distance",  0.35)  # m
        self.declare_parameter("safety_arc_deg",        90.0)  # ± half this around forward
        self.declare_parameter("safety_min_range",      0.05)  # ignore returns below this (sensor noise)

        # ── One-lap auto-stop ─────────────────────────────────────────────
        # Subscribes to /initialpose (RViz "2D Pose Estimate" arrow). Once
        # the car has driven at least lap_far_threshold metres away from
        # that point AND returned within lap_close_threshold of it, the
        # commanded speed is forced to 0. Republishing /initialpose resets
        # the detector so the next lap can begin. Disable by setting
        # lap_far_threshold <= 0.
        self.declare_parameter("lap_topic",            "/initialpose")
        self.declare_parameter("lap_far_threshold",    2.0)   # m
        self.declare_parameter("lap_close_threshold",  0.6)   # m
        # Grace period AFTER the car first re-enters the close zone before
        # the stop fires. Lets the car cross past the start point cleanly
        # (at 0.5 m/s × 3 s ≈ 1.5 m overshoot) instead of slamming brakes
        # the instant it touches the close radius.
        self.declare_parameter("lap_dwell_seconds",    3.0)

        # Read parameters
        self.odom_topic  = str(self.get_parameter("odom_topic").value)
        self.twist_topic = str(self.get_parameter("twist_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.velocity    = float(self.get_parameter("velocity").value)
        self.min_speed   = float(self.get_parameter("min_speed").value)
        self.max_speed   = float(self.get_parameter("max_speed").value)

        self.min_lookahead       = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead       = float(self.get_parameter("max_lookahead").value)
        self.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)

        self.min_gain        = float(self.get_parameter("min_gain").value)
        self.max_gain        = float(self.get_parameter("max_gain").value)
        self.gain_speed_scale = float(self.get_parameter("gain_speed_scale").value)
        self.D               = float(self.get_parameter("D").value)

        self.steering_limit       = float(self.get_parameter("steering_limit").value)
        self.lookahead_window     = int(self.get_parameter("lookahead_window").value)
        self.steer_smooth_weight  = float(self.get_parameter("steer_smooth_weight").value)

        self.scan_topic            = str(self.get_parameter("scan_topic").value)
        self.safety_stop_distance  = float(self.get_parameter("safety_stop_distance").value)
        self.safety_arc_deg        = float(self.get_parameter("safety_arc_deg").value)
        self.safety_min_range      = float(self.get_parameter("safety_min_range").value)
        self.emergency_stop        = False

        self.lap_topic             = str(self.get_parameter("lap_topic").value)
        self.lap_far_threshold     = float(self.get_parameter("lap_far_threshold").value)
        self.lap_close_threshold   = float(self.get_parameter("lap_close_threshold").value)
        self.lap_dwell_seconds     = float(self.get_parameter("lap_dwell_seconds").value)
        self.lap_start             = None    # np.ndarray when set
        self.lap_far               = False
        self.lap_done              = False
        # ROS clock time when the car first re-entered the close zone after
        # passing the far threshold. lap_done triggers lap_dwell_seconds
        # later — gives the car a small overshoot past the start point.
        self.lap_complete_at       = None

        # External pause flag (set by /controller_pause topic, published by
        # the launcher's pause/resume button). Path markers keep publishing
        # while paused; only the commanded drive speed is forced to 0.
        self.paused                = False

        # Load waypoints (also sets self.waypoint_velocities)
        self.waypoint_velocities = None
        self.waypoints = self._load_waypoints()
        self.num_pts   = len(self.waypoints)
        self.get_logger().info(f"Loaded {self.num_pts} waypoints")

        # State
        self.current_speed    = 0.0
        self.prev_nearest_idx = None
        self.prev_steer_error = 0.0
        self.prev_d_item      = 0.0
        self.prev_steer       = 0.0
        self.target_point     = None
        # Set True after the first odom message; used to one-shot reverse
        # the path if the car is facing opposite the loop direction at start
        # (raceline_generator picks an arbitrary CW/CCW orientation).
        self._direction_checked = False

        # Subscriptions / publications
        self.odom_sub  = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        if self.twist_topic:
            self.twist_sub = self.create_subscription(
                Odometry, self.twist_topic, self._twist_callback, 1
            )
            self.get_logger().info(
                f"speed source: {self.twist_topic} (twist on {self.odom_topic} ignored)"
            )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.target_pub = self.create_publisher(Marker, "/pure_pursuit/target", 10)
        self.path_pub   = self.create_publisher(Marker, "/viz/path", 10)

        if self.safety_stop_distance > 0.0:
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self._scan_callback,
                qos_profile_sensor_data
            )
            self.get_logger().info(
                f"safety: stop if forward-arc(±{self.safety_arc_deg/2:.0f}°) "
                f"lidar < {self.safety_stop_distance:.2f} m"
            )

        if self.lap_far_threshold > 0.0:
            self.lap_sub = self.create_subscription(
                PoseWithCovarianceStamped, self.lap_topic,
                self._lap_callback, 1
            )
            self.get_logger().info(
                f"lap stop: set start with RViz '2D Pose Estimate' on "
                f"{self.lap_topic} (need to travel >{self.lap_far_threshold:.1f} m "
                f"away, return within {self.lap_close_threshold:.1f} m)"
            )

        # External pause: launcher publishes Bool on /controller_pause with
        # TRANSIENT_LOCAL durability so late-joining subscribers (us, just
        # launched) immediately get the latest pause state.
        pause_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pause_sub = self.create_subscription(
            Bool, "/controller_pause", self._pause_callback, pause_qos
        )

        # Publish path immediately and then twice a second so RViz picks
        # it up within the first frame after the controller starts, and
        # late-joining subscribers still see it.
        self._publish_path()
        self.create_timer(0.5, self._publish_path)

    # ------------------------------------------------------------------
    # Waypoint loading
    # ------------------------------------------------------------------

    def _load_waypoints(self):
        path = str(self.get_parameter("waypoints_path").value)
        if not path:
            pkg_share = get_package_share_directory("f1tenth_controller")
            path = os.path.join(pkg_share, "path", "path_v_centerline.yaml")

        self.get_logger().info(f"Loading waypoints from: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        wp_list = data["waypoints"]
        xy = np.array([[wp["x"], wp["y"]] for wp in wp_list])
        # Load per-waypoint velocity if present, otherwise use fixed param
        if "v" in wp_list[0]:
            self.waypoint_velocities = np.array([wp["v"] for wp in wp_list])
            self.get_logger().info("Using per-waypoint speed profile from path_v_centerline.yaml")
        else:
            self.waypoint_velocities = None
        return xy

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _pause_callback(self, msg: Bool):
        # Toggle external pause. While paused, the drive callback forces
        # commanded speed to 0 but path/target markers keep publishing so
        # RViz still shows the planned line.
        if self.paused != msg.data:
            self.get_logger().info(f"pause = {msg.data}")
        self.paused = bool(msg.data)

    def _twist_callback(self, msg: Odometry):
        # Separate twist source — used when odom_topic carries pose but no
        # twist (e.g. /dedicate_odom in map frame).
        self.current_speed = msg.twist.twist.linear.x

    def _lap_callback(self, msg: PoseWithCovarianceStamped):
        # Each /initialpose message defines a NEW lap-start point and resets
        # the lap-complete detector. Lets the user start another lap by
        # re-publishing the arrow in RViz after the car has stopped.
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

    def _scan_callback(self, msg: LaserScan):
        # Trigger emergency stop if the closest lidar return inside the
        # forward arc is nearer than safety_stop_distance.
        n = len(msg.ranges)
        if n == 0:
            return
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        arc_half = math.radians(self.safety_arc_deg / 2.0)
        front = np.abs(angles) <= arc_half
        ranges = np.asarray(msg.ranges)[front]
        valid = np.isfinite(ranges) & (ranges > self.safety_min_range)
        if not valid.any():
            return
        min_dist = float(ranges[valid].min())
        stop_now = min_dist < self.safety_stop_distance
        if stop_now and not self.emergency_stop:
            self.get_logger().warn(
                f"EMERGENCY STOP — obstacle {min_dist:.2f} m ahead "
                f"(threshold {self.safety_stop_distance:.2f} m)"
            )
        elif not stop_now and self.emergency_stop:
            self.get_logger().info(
                f"path clear ({min_dist:.2f} m), resuming"
            )
        self.emergency_stop = stop_now

    def odom_callback(self, msg: Odometry):
        pose  = msg.pose.pose
        twist = msg.twist.twist

        curr_x   = pose.position.x
        curr_y   = pose.position.y
        curr_pos = np.array([curr_x, curr_y])
        if not self.twist_topic:
            self.current_speed = twist.linear.x

        q = pose.orientation
        curr_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

        # On the first odom message, check whether the loop's CW/CCW
        # ordering matches the car's current heading. raceline_generator
        # picks an arbitrary direction; if the car is placed facing the
        # opposite way, every lookahead target lands BEHIND the car and
        # pure pursuit dives off the path. Flipping the waypoint order
        # fixes this once at startup.
        if not self._direction_checked:
            nearest_now = int(np.argmin(
                np.linalg.norm(self.waypoints - curr_pos, axis=1)))
            nxt = (nearest_now + 1) % self.num_pts
            path_vec = self.waypoints[nxt] - self.waypoints[nearest_now]
            path_yaw = math.atan2(path_vec[1], path_vec[0])
            yaw_diff = (curr_yaw - path_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_diff) > math.pi / 2:
                self.waypoints = self.waypoints[::-1]
                if self.waypoint_velocities is not None:
                    self.waypoint_velocities = self.waypoint_velocities[::-1]
                self.prev_nearest_idx = None  # rebuild nearest with new order
                self.get_logger().warn(
                    f"Path direction reversed at startup: car_yaw="
                    f"{math.degrees(curr_yaw):+.0f}°, path_yaw="
                    f"{math.degrees(path_yaw):+.0f}° "
                    f"(Δ={math.degrees(yaw_diff):+.0f}°)"
                )
            else:
                self.get_logger().info(
                    f"Path direction matches car heading "
                    f"(Δ={math.degrees(yaw_diff):+.0f}°)"
                )
            self._direction_checked = True

        # 1. Find nearest waypoint using local window
        nearest_idx = self._find_nearest(curr_pos)

        # 2. Walk forward along arc to find lookahead point
        L = self._get_lookahead(self.current_speed)
        target_global = self._find_lookahead_point(curr_pos, nearest_idx, L)
        if target_global is None:
            return
        self.target_point = target_global

        # 3. Transform target to vehicle frame
        R_mat = np.array([
            [ np.cos(curr_yaw), np.sin(curr_yaw)],
            [-np.sin(curr_yaw), np.cos(curr_yaw)]
        ])
        local = R_mat @ (target_global - curr_pos)
        target_y = local[1]

        # 4. Pure pursuit steering
        L_actual = max(np.linalg.norm(curr_pos - target_global), 1e-6)
        error    = (2.0 * target_y) / (L_actual ** 2)
        steer    = self._get_steering(self.current_speed, error)
        w        = self.steer_smooth_weight
        steer    = w * steer + (1.0 - w) * self.prev_steer
        self.prev_steer = steer

        # 5. Publish
        # Use per-waypoint speed profile if available, else fixed velocity
        if self.waypoint_velocities is not None:
            target_speed = float(self.waypoint_velocities[nearest_idx])
        else:
            target_speed = self.velocity
        target_speed = float(np.clip(target_speed, self.min_speed, self.max_speed))

        # Auto-record lap start on first odom message so the detector works
        # even if the user doesn't manually click '2D Pose Estimate' first.
        # /initialpose still overrides this (resets lap_start, lap_far, lap_done).
        if self.lap_start is None and self.lap_far_threshold > 0.0:
            self.lap_start = curr_pos.copy()
            self.get_logger().info(
                f"lap start (auto) ← ({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f})  "
                f"— set 2D Pose Estimate in RViz to override"
            )

        # Lap-complete detector — needs both a "far away" excursion and a
        # "back near start" return so the car can't trigger before moving.
        # After re-entering the close zone, an additional lap_dwell_seconds
        # grace period runs so the car overshoots the start point cleanly.
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

        # Emergency stop, lap complete, or external pause overrides the
        # commanded speed but keeps current steering so the car holds its
        # line while braking.
        if self.emergency_stop or self.lap_done or self.paused:
            target_speed = 0.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed           = target_speed
        drive_msg.drive.steering_angle  = steer
        self.drive_pub.publish(drive_msg)

        # Throttled diagnostic so the user can confirm pure_pursuit IS
        # publishing commands and what they are. One line per second.
        self.get_logger().info(
            f"cmd: v={target_speed:.2f} m/s  δ={math.degrees(steer):+.1f}°  "
            f"target=({target_global[0]:.2f},{target_global[1]:.2f})  "
            f"speed_now={self.current_speed:.2f}  "
            f"{'[E-STOP]' if self.emergency_stop else ''}",
            throttle_duration_sec=1.0,
        )

        self._draw_marker(self.target_point, self.target_pub, color="yellow")

    # ------------------------------------------------------------------
    # Nearest waypoint (local window to avoid confusion on closed loops)
    # ------------------------------------------------------------------

    def _find_nearest(self, curr_pos):
        if self.prev_nearest_idx is None:
            idx = int(np.argmin(np.linalg.norm(self.waypoints - curr_pos, axis=1)))
        else:
            window_idxs = [(self.prev_nearest_idx + i) % self.num_pts
                           for i in range(self.lookahead_window)]
            dists = np.linalg.norm(self.waypoints[window_idxs] - curr_pos, axis=1)
            idx   = window_idxs[int(np.argmin(dists))]
        self.prev_nearest_idx = idx
        return idx

    # ------------------------------------------------------------------
    # Arc-length lookahead (handles sharp turns correctly)
    # ------------------------------------------------------------------

    def _find_lookahead_point(self, curr_pos, nearest_idx, L):
        # Walk arc-length along the discrete waypoints. When the target arc
        # length L lands INSIDE a segment, linearly interpolate between the
        # two waypoints so the lookahead point is a continuous function of
        # progress — avoids the step-change in steering when L hops to the
        # next waypoint (the main cause of "jagged" tracking on tight turns).
        arc = 0.0
        idx = nearest_idx
        for _ in range(self.num_pts):
            next_idx = (idx + 1) % self.num_pts
            seg = self.waypoints[next_idx] - self.waypoints[idx]
            seg_len = float(np.linalg.norm(seg))
            if arc + seg_len >= L and seg_len > 1e-9:
                t = (L - arc) / seg_len
                return self.waypoints[idx] + t * seg
            arc += seg_len
            idx = next_idx
        return self.waypoints[idx].copy()

    # ------------------------------------------------------------------
    # Speed-dependent lookahead distance
    # ------------------------------------------------------------------

    def _get_lookahead(self, speed):
        t = (speed - self.min_lookahead_speed) / max(
            self.max_lookahead_speed - self.min_lookahead_speed, 1e-6
        )
        t = np.clip(t, 0.0, 1.0)
        return self.min_lookahead + t * (self.max_lookahead - self.min_lookahead)

    # ------------------------------------------------------------------
    # Speed-dependent PD steering
    # ------------------------------------------------------------------

    def _get_steering(self, speed, error):
        t = speed / max(self.gain_speed_scale, 1e-6)
        P = self.max_gain - np.clip(t, 0.0, 1.0) * (self.max_gain - self.min_gain)

        d_error = error - self.prev_steer_error
        if d_error == 0.0:
            d_error = self.prev_d_item
        else:
            self.prev_d_item      = d_error
            self.prev_steer_error = error

        steer = P * error + self.D * d_error
        return float(np.clip(steer, -np.radians(self.steering_limit),
                                     np.radians(self.steering_limit)))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _publish_path(self):
        from geometry_msgs.msg import Point
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns     = "pure_pursuit_path"
        marker.id     = 0
        marker.type   = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 0.8
        marker.color.g = 1.0
        marker.pose.orientation.w = 1.0

        for wp in np.vstack([self.waypoints, self.waypoints[:1]]):  # close loop
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            marker.points.append(p)

        self.path_pub.publish(marker)

    def _draw_marker(self, position, publisher, color="red"):
        if position is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id     = 0
        marker.ns     = "pure_pursuit_target"
        marker.type   = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.a = 1.0
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "yellow":
            marker.color.r = 1.0
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime.nanosec = int(1e8)
        publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
