#!/usr/bin/python3

import math
import os
import signal
import time

import numpy as np
import yaml
from scipy import signal as scipy_signal
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       DurabilityPolicy, ReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory


class StanleyAvoidance(Node):
    def __init__(self):
        super().__init__("stanley_avoidance_node")

        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/dedicate_odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("velocity", 0.5)
        self.declare_parameter("K_E", 0.5)
        self.declare_parameter("K_H", 0.4)
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("K_p_obstacle", 0.3)
        self.declare_parameter("max_shift_cells", 6)
        self.declare_parameter("avoidance_steer_limit", 0.524)  # [ADAPT-8] 30°
        self.declare_parameter("min_lookahead", 0.3)
        self.declare_parameter("max_lookahead", 0.8)
        self.declare_parameter("min_lookahead_speed", 0.5)
        self.declare_parameter("max_lookahead_speed", 0.7)
        self.declare_parameter("velocity_percentage", 1.0)
        self.declare_parameter("velocity_min", 0.5)
        self.declare_parameter("velocity_max", 0.7)
        self.declare_parameter("steering_limit", 0.4189)  # [ADAPT-2] radians
        self.declare_parameter("grid_width_meters", 6.0)
        self.declare_parameter("cells_per_meter", 20)
        self.declare_parameter("wheelbase", 0.30)  # [ADAPT-1] from URDF
        # [ADAPT-9] E-stop
        self.declare_parameter("estop_dist", 0.10)
        self.declare_parameter("estop_half_arc_deg", 15.0)
        self.declare_parameter("lidar_to_nose", 0.28)
        self.declare_parameter("car_half_width", 0.16)
        self.declare_parameter("safety_buffer", 0.0)
        self.declare_parameter("obstacle_detect_lookahead", 2.5)  # [ADAPT-5]

        # ── One-lap auto-stop (mirrors pure_pursuit / lattice). Set the
        # 2D Pose Estimate in RViz (publishes /initialpose) to mark a lap
        # start, or just rely on the first odom auto-record. After the car
        # has gone >lap_far_threshold m away and returned within
        # lap_close_threshold m, wait lap_dwell_seconds, then force stop.
        self.declare_parameter("lap_topic",            "/initialpose")
        self.declare_parameter("lap_far_threshold",    2.0)
        self.declare_parameter("lap_close_threshold",  0.6)
        self.declare_parameter("lap_dwell_seconds",    3.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.max_shift_cells = int(self.get_parameter("max_shift_cells").value)
        self.avoidance_steer_limit = float(self.get_parameter("avoidance_steer_limit").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.grid_width_meters = float(self.get_parameter("grid_width_meters").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.base_velocity = float(self.get_parameter("velocity").value)
        self.estop_dist = float(self.get_parameter("estop_dist").value)
        self.estop_half_arc = math.radians(float(self.get_parameter("estop_half_arc_deg").value))
        self.lidar_to_nose = float(self.get_parameter("lidar_to_nose").value)
        self.car_half_width = float(self.get_parameter("car_half_width").value)
        self.safety_buffer = float(self.get_parameter("safety_buffer").value)
        self.obstacle_detect_lookahead = float(self.get_parameter("obstacle_detect_lookahead").value)
        self.inflate_radius = self.car_half_width + self.safety_buffer

        # Lap-stop state (same scheme as pure_pursuit / lattice).
        self.lap_topic           = str(self.get_parameter("lap_topic").value)
        self.lap_far_threshold   = float(self.get_parameter("lap_far_threshold").value)
        self.lap_close_threshold = float(self.get_parameter("lap_close_threshold").value)
        self.lap_dwell_seconds   = float(self.get_parameter("lap_dwell_seconds").value)
        self.lap_start           = None     # np.ndarray when set
        self.lap_far             = False
        self.lap_done            = False
        self.lap_complete_at     = None

        self.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)
        self.L = self.max_lookahead

        self.waypoints_world, self.velocities = self._load_waypoints()
        self.get_logger().info(f"Loaded {len(self.waypoints_world)} waypoints")

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        # /scan is BEST_EFFORT — QoS must match or subscription drops everything.
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data)
        # External pause flag from the launcher (matches pure_pursuit + lattice).
        # While paused, drive callbacks override commanded speed to 0; path
        # markers keep streaming so RViz still shows the planned line.
        self.paused = False
        pause_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pause_sub = self.create_subscription(
            Bool, "/controller_pause", self._pause_cb, pause_qos
        )
        # /initialpose for the lap-stop detector (RViz 2D Pose Estimate).
        # If the user never publishes, the first odom message auto-records
        # the start point — see odom_callback below.
        if self.lap_far_threshold > 0.0:
            self.lap_sub = self.create_subscription(
                PoseWithCovarianceStamped, self.lap_topic,
                self._lap_cb, 1
            )
            self.get_logger().info(
                f"lap stop: set start with RViz '2D Pose Estimate' on "
                f"{self.lap_topic} (need to travel >{self.lap_far_threshold:.1f} m "
                f"away, return within {self.lap_close_threshold:.1f} m, "
                f"then {self.lap_dwell_seconds:.1f} s grace → STOP)"
            )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.target_pub = self.create_publisher(Marker, "/viz/drive_target", 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)

        self.grid_height = int(self.obstacle_detect_lookahead * self.CELLS_PER_METER)
        self.grid_width = int(self.grid_width_meters * self.CELLS_PER_METER)
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0, dtype=int)

        self.IS_OCCUPIED = 100
        self.IS_FREE = 0

        self.current_pose = None
        self.current_pose_wheelbase_front = None
        self.closest_wheelbase_rear_point = None
        self.goal_pos = None
        self.target_velocity = 0.0
        self.obstacle_detected = False
        self.obstacle_distance = 999.0
        self.velocity_index = 0
        self.index = 0
        self.odom_frame = "map"
        self._direction_checked = False

        # Detection lock + confirm/clear counters
        self.obstacle_side_lock = None
        self.obstacle_block_counter = 0
        self.obstacle_block_threshold = 3
        self.obstacle_clear_counter = 0
        self.obstacle_clear_threshold = 10
        self.last_avoid_target = None

        # [ADAPT-15] local-window waypoint search
        self._wp_idx = None
        self.waypoint_window = 12

        # [ADAPT-16] lateral offset on stanley's cte
        self.declare_parameter("avoidance_offset_m", 0.30)
        self.declare_parameter("max_avoidance_offset_m", 0.60)  # cap so we don't drive into walls
        self.declare_parameter("offset_ramp_per_tick", 0.04)
        self.avoidance_offset_m = float(self.get_parameter("avoidance_offset_m").value)
        self.max_avoidance_offset_m = float(self.get_parameter("max_avoidance_offset_m").value)
        self.offset_ramp_per_tick = float(self.get_parameter("offset_ramp_per_tick").value)
        self.target_offset = 0.0
        self.avoidance_offset = 0.0
        self.obstacle_world_xy = None
        self.declare_parameter("obstacle_passed_thresh", -0.10)
        self.obstacle_passed_thresh = float(self.get_parameter("obstacle_passed_thresh").value)

    def _load_waypoints(self):
        waypoints_path = str(self.get_parameter("waypoints_path").value)

        if not waypoints_path:
            pkg_share = get_package_share_directory("f1tenth_controller")
            waypoints_path = os.path.join(pkg_share, "path", "path.yaml")

        self.get_logger().info(f"Loading waypoints from: {waypoints_path}")

        with open(waypoints_path, "r") as f:
            data = yaml.safe_load(f)

        wp_list = data["waypoints"]
        points = np.array([[wp["x"], wp["y"], 0.0] for wp in wp_list])
        velocities = np.full(len(wp_list), self.base_velocity)
        # [ADAPT-4] store per-waypoint theta for direct use by stanley
        self.thetas = np.array([wp.get("theta", 0.0) for wp in wp_list])

        return points, velocities

    def _transform_waypoints(self, waypoints, position, pose):
        translated = waypoints - np.array(position)
        quaternion = np.array([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ])
        return R.inv(R.from_quat(quaternion)).apply(translated)

    def _local_nearest(self, distances):
        """[ADAPT-15] argmin restricted to ±waypoint_window around the
        last known index, with wrap-around. Prevents jumps to the
        opposite side of a closed-loop path."""
        n = len(distances)
        if self._wp_idx is None:
            return int(np.argmin(distances))
        w = self.waypoint_window
        candidates = [(self._wp_idx + di) % n for di in range(-w, w + 1)]
        return candidates[int(np.argmin(distances[candidates]))]

    def _get_closest_waypoint_with_velocity(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        self.velocity_index = self._local_nearest(distances)
        self._wp_idx = self.velocity_index   # canonical "where we are"
        return self.waypoints_world[self.velocity_index], self.velocities[self.velocity_index]

    def _get_waypoint(self, pose, target_velocity):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)

        # self.L stays pinned at max_lookahead; dynamic_L only picks the goal
        dynamic_L = min(
            max(
                self.min_lookahead,
                self.min_lookahead
                + (self.max_lookahead - self.min_lookahead)
                * (target_velocity - self.min_lookahead_speed)
                / max(self.max_lookahead_speed - self.min_lookahead_speed, 0.01),
            ),
            self.max_lookahead,
        )

        # [ADAPT-15] local window
        n = len(self.waypoints_world)
        w = self.waypoint_window
        if self._wp_idx is not None:
            window_idx = np.array([(self._wp_idx + di) % n
                                   for di in range(-w, w + 1)])
        else:
            window_idx = np.arange(n)

        forward_mask = waypoints_car[window_idx][:, 0] > 0
        in_range_mask = distances[window_idx] < dynamic_L
        valid_mask = in_range_mask & forward_mask

        if valid_mask.any():
            # farthest in-range, forward
            cands = window_idx[valid_mask]
            idx = int(cands[int(np.argmax(distances[cands]))])
            self.index = idx
            return waypoints_car[idx], self.waypoints_world[idx]

        if forward_mask.any():
            # no in-range forward → pick closest forward in window
            cands = window_idx[forward_mask]
            idx = int(cands[int(np.argmin(distances[cands]))])
            self.index = idx
            return waypoints_car[idx], self.waypoints_world[idx]

        return None, None

    def _path_blocked(self, max_distance, margin):
        """[ADAPT-5] Path-walking obstacle detection.

        Walk along path waypoints IN PATH-INDEX ORDER from the car's
        tracked position, up to max_distance of accumulated arc length.
        Walking by index (not by straight-line distance) prevents chord
        crossings through internal walls on closed-loop paths.

        Returns (blocked, blocking_info_or_None) where blocking_info is:
            {'prev_cell': (i,j),  # segment start (closer to car)
             'next_cell': (i,j),  # segment end (further forward)
             'obstacle_cell': (i,j)}  # the actual occupied cell that triggered
        """
        position = (self.current_pose.position.x, self.current_pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, self.current_pose)
        N = len(waypoints_car)

        # pick starting waypoint in path order: advance from _wp_idx until forward in car frame
        if self._wp_idx is not None:
            start = self._wp_idx
            for _ in range(N):
                if waypoints_car[start, 0] > 0.05:
                    break
                start = (start + 1) % N
            else:
                return False, None
        else:
            forward_mask = waypoints_car[:, 0] > 0.05
            if not forward_mask.any():
                return False, None
            distances = np.linalg.norm(waypoints_car, axis=1)
            cand = np.where(forward_mask)[0]
            start = int(cand[np.argmin(distances[cand])])

        prev_cell = self._to_grid(0.0, 0.0)
        prev_world = np.array([self.current_pose.position.x, self.current_pose.position.y])
        cumulative = 0.0
        idx = start
        for _ in range(N):
            wp_world = self.waypoints_world[idx][:2]
            cumulative += float(np.linalg.norm(wp_world - prev_world))
            if cumulative > max_distance:
                break
            wp = waypoints_car[idx]
            cell = self._to_grid(float(wp[0]), float(wp[1]))
            collision, obs_cell = self._check_collision_with_obstacle(prev_cell, cell, margin=margin)
            if collision:
                return True, {'prev_cell': prev_cell, 'next_cell': cell, 'obstacle_cell': obs_cell}
            prev_cell = cell
            prev_world = wp_world
            idx = (idx + 1) % N
        return False, None

    def _get_waypoint_stanley(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        index = self._local_nearest(distances)   # [ADAPT-15] local window
        self._wp_idx = index
        return waypoints_car[index], self.waypoints_world[index], self.thetas[index], index

    def _pause_cb(self, msg: Bool):
        # External pause from the launcher button. While True, the drive
        # publishes below force speed=0. Path/marker publishes keep running.
        if self.paused != msg.data:
            self.get_logger().info(f"pause = {msg.data}")
        self.paused = bool(msg.data)

    def _lap_cb(self, msg: PoseWithCovarianceStamped):
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

    def _update_lap_state(self, curr_pos):
        """Auto-record lap start on first odom message, then run the
        far-then-close detector with a grace-period overshoot."""
        if self.lap_start is None and self.lap_far_threshold > 0.0:
            self.lap_start = np.asarray(curr_pos, dtype=float)
            self.get_logger().info(
                f"lap start (auto) ← ({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f})  "
                f"— set 2D Pose Estimate in RViz to override"
            )
            return
        if self.lap_start is None or self.lap_done:
            return
        d = float(np.linalg.norm(np.asarray(curr_pos) - self.lap_start))
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

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.odom_frame = msg.header.frame_id
        self._update_lap_state(
            (msg.pose.pose.position.x, msg.pose.pose.position.y))

        current_pose_quaternion = np.array([
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w,
        ])

        # [ADAPT-3] one-shot startup direction check
        if not self._direction_checked:
            curr_yaw = R.from_quat(current_pose_quaternion).as_euler("xyz")[2]
            curr_pos = np.array([self.current_pose.position.x,
                                 self.current_pose.position.y, 0.0])
            nearest = int(np.argmin(np.linalg.norm(self.waypoints_world - curr_pos, axis=1)))
            nxt = (nearest + 1) % len(self.waypoints_world)
            pv = self.waypoints_world[nxt] - self.waypoints_world[nearest]
            path_yaw = math.atan2(pv[1], pv[0])
            yaw_diff = math.atan2(math.sin(curr_yaw - path_yaw),
                                  math.cos(curr_yaw - path_yaw))
            if abs(yaw_diff) > math.pi / 2:
                self.waypoints_world = self.waypoints_world[::-1]
                self.velocities = self.velocities[::-1]
                # Flipping order also flips tangent direction, so theta += pi
                self.thetas = (self.thetas[::-1] + math.pi
                               + math.pi) % (2 * math.pi) - math.pi
                self._wp_idx = None  # invalidate after reversal
                self.get_logger().warn(
                    f"Path direction reversed at startup: car_yaw="
                    f"{math.degrees(curr_yaw):+.0f}°, path_yaw="
                    f"{math.degrees(path_yaw):+.0f}° "
                    f"(Δ={math.degrees(yaw_diff):+.0f}°)"
                )
            else:
                self.get_logger().info(
                    f"Path direction matches car heading (Δ={math.degrees(yaw_diff):+.0f}°)"
                )
            self._direction_checked = True

        self.current_pose_wheelbase_front = Pose()
        current_pose_xyz = R.from_quat(current_pose_quaternion).apply((self.wheelbase, 0, 0)) + (
            self.current_pose.position.x,
            self.current_pose.position.y,
            0,
        )
        self.current_pose_wheelbase_front.position.x = current_pose_xyz[0]
        self.current_pose_wheelbase_front.position.y = current_pose_xyz[1]
        self.current_pose_wheelbase_front.position.z = current_pose_xyz[2]
        self.current_pose_wheelbase_front.orientation = self.current_pose.orientation

        self.closest_wheelbase_rear_point, self.target_velocity = self._get_closest_waypoint_with_velocity(
            self.current_pose
        )

        self.goal_pos, goal_pos_world = self._get_waypoint(self.current_pose, self.target_velocity)

        if goal_pos_world is None:
            self.get_logger().warn(f"No lookahead waypoint found! L={self.L:.2f} vel={self.target_velocity:.2f}")

    def drive_to_target(self, point, K_p):
        L = np.linalg.norm(point)
        y = point[1]
        angle = K_p * (2 * y) / (L ** 2)
        # [ADAPT-8] tighter cap during avoidance
        angle = np.clip(angle, -self.avoidance_steer_limit, self.avoidance_steer_limit)
        # [ADAPT-7] velocity = cruise during avoidance (e-stop is the safety floor)
        velocity = self.target_velocity * self.velocity_percentage
        if self.paused or self.lap_done:
            velocity = 0.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(
            f"AVOID target=({point[0]:+.2f},{point[1]:+.2f}) "
            f"cmd={math.degrees(angle):+.1f}° v={velocity:.2f} "
            f"obs_dist={self.obstacle_distance:.2f}m "
            f"vel_wp={self.velocity_index} goal_wp={self.index}",
            throttle_duration_sec=0.3,
        )
        return angle, velocity

    def drive_to_target_stanley(self):
        # [ADAPT-4] path_heading from yaml theta (4th return is wp index)
        closest_wheelbase_front_point_car, _, path_heading, stanley_wp = self._get_waypoint_stanley(
            self.current_pose_wheelbase_front
        )

        current_heading = math.atan2(
            self.current_pose_wheelbase_front.position.y - self.current_pose.position.y,
            self.current_pose_wheelbase_front.position.x - self.current_pose.position.x,
        )

        # [ADAPT-16] cte gets the lateral offset; heading stays parallel to path
        y_lateral = closest_wheelbase_front_point_car[1]
        y_effective = y_lateral + self.avoidance_offset
        crosstrack_error = math.atan2(self.K_E * y_effective, self.target_velocity)
        heading_error = path_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        heading_error *= self.K_H

        angle = heading_error + crosstrack_error
        angle = np.clip(angle, -self.steering_limit, self.steering_limit)

        velocity = self.target_velocity * self.velocity_percentage
        if self.paused or self.lap_done:
            velocity = 0.0

        self.get_logger().info(
            f"cte={y_lateral:+.2f}m off={self.avoidance_offset:+.2f}m "
            f"head_err={math.degrees(heading_error / self.K_H):+.0f}° "
            f"cmd={math.degrees(angle):+.0f}° v={velocity:.2f} "
            f"stanley_wp={stanley_wp} vel_wp={self.velocity_index} goal_wp={self.index}",
            throttle_duration_sec=0.5,
        )

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)
        return angle, velocity

    def scan_callback(self, msg):
        if self.current_pose is None or self.goal_pos is None:
            return

        # [ADAPT-9] E-stop in nose-relative meters (lidar_range − lidar_to_nose)
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        forward = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) < self.estop_half_arc
        rng = ranges[forward]
        rng = rng[np.isfinite(rng) & (rng > 0.05)]
        if rng.size > 0:
            nose_dist = rng.min() - self.lidar_to_nose
            if nose_dist < self.estop_dist:
                stop = AckermannDriveStamped()
                stop.drive.speed = 0.0
                stop.drive.steering_angle = 0.0
                self.drive_pub.publish(stop)
                self.get_logger().warn(
                    f"E-STOP: obstacle {nose_dist:.2f}m from nose "
                    f"(lidar range {rng.min():.2f}m, lidar_to_nose {self.lidar_to_nose:.2f}m)",
                    throttle_duration_sec=1.0,
                )
                return

        self._populate_grid(msg.ranges, msg.angle_increment, msg.angle_min)
        self._convolve_grid()
        self._publish_grid(msg.header.frame_id, msg.header.stamp)

        current_pos = np.array(self._to_grid(0, 0))
        goal_pos = np.array(self._to_grid(self.goal_pos[0], self.goal_pos[1]))
        target = None
        MARGIN = int(self.CELLS_PER_METER * self.inflate_radius)

        # [ADAPT-5] path-walking detection
        blocked, block_info = self._path_blocked(
            max_distance=self.obstacle_detect_lookahead, margin=MARGIN
        )
        self.get_logger().info(
            f"path_blocked={blocked} car_grid=({current_pos[0]},{current_pos[1]}) "
            f"goal_grid=({goal_pos[0]},{goal_pos[1]}) "
            f"H/W={self.grid_height}x{self.grid_width} MARGIN={MARGIN}",
            throttle_duration_sec=0.5,
        )

        # [ADAPT-16] check if we've passed the locked obstacle in WORLD frame
        if self.obstacle_world_xy is not None:
            wx, wy = self.obstacle_world_xy
            quat = np.array([self.current_pose.orientation.x, self.current_pose.orientation.y,
                             self.current_pose.orientation.z, self.current_pose.orientation.w])
            inv = R.inv(R.from_quat(quat))
            rel = inv.apply([wx - self.current_pose.position.x,
                             wy - self.current_pose.position.y, 0.0])
            obs_x_car = float(rel[0])
            obs_y_car = float(rel[1])
            self.get_logger().info(
                f"  locked obs world=({wx:+.2f},{wy:+.2f}) "
                f"car_pose=({self.current_pose.position.x:+.2f},{self.current_pose.position.y:+.2f}) "
                f"obs_in_car=({obs_x_car:+.2f},{obs_y_car:+.2f})",
                throttle_duration_sec=0.3,
            )
            if obs_x_car < self.obstacle_passed_thresh:
                self.get_logger().info(
                    f"  PASSED obstacle (obs_x_car={obs_x_car:+.2f}m) — ramping offset back"
                )
                self.target_offset = 0.0
                self.obstacle_world_xy = None
                self.obstacle_side_lock = None
                self.obstacle_block_counter = 0

        if blocked:
            self.obstacle_distance = (current_pos[0] - block_info['obstacle_cell'][0]) / self.CELLS_PER_METER
            self.obstacle_block_counter += 1
            self.obstacle_clear_counter = 0

            if self.obstacle_side_lock is None and self.obstacle_block_counter < self.obstacle_block_threshold:
                pass  # not yet confirmed
            else:
                self.obstacle_detected = True
                # [ADAPT-6] side relative to path tangent
                measured_on_right = self._obstacle_on_right_of_path(
                    block_info['prev_cell'],
                    block_info['next_cell'],
                    block_info['obstacle_cell'],
                )

                # obstacle's perpendicular distance from blocked segment
                p = block_info['prev_cell']
                q = block_info['next_cell']
                o = block_info['obstacle_cell']
                di = float(q[0] - p[0])
                dj = float(q[1] - p[1])
                pmag = math.sqrt(di * di + dj * dj)
                if pmag > 0.01:
                    pi = -dj / pmag
                    pj = -di / pmag
                    mi = (p[0] + q[0]) / 2.0
                    mj = (p[1] + q[1]) / 2.0
                    perp_cells = abs((o[0] - mi) * pi + (o[1] - mj) * pj)
                    perp_m = perp_cells / self.CELLS_PER_METER
                else:
                    perp_m = 0.0

                if self.obstacle_side_lock is None:
                    self.obstacle_side_lock = measured_on_right
                    obs_car_xy = self._from_grid(block_info['obstacle_cell'])
                    quat = np.array([self.current_pose.orientation.x, self.current_pose.orientation.y,
                                     self.current_pose.orientation.z, self.current_pose.orientation.w])
                    rotw = R.from_quat(quat)
                    world_offset = rotw.apply([obs_car_xy[0], obs_car_xy[1], 0.0])
                    self.obstacle_world_xy = (
                        self.current_pose.position.x + float(world_offset[0]),
                        self.current_pose.position.y + float(world_offset[1]),
                    )
                    self.get_logger().info(
                        f"  LOCKED obstacle_side={'RIGHT' if measured_on_right else 'LEFT'} "
                        f"obstacle_cell={block_info['obstacle_cell']} "
                        f"obstacle_world=({self.obstacle_world_xy[0]:+.2f},{self.obstacle_world_xy[1]:+.2f}) "
                        f"perp_from_path={perp_m:.2f}m "
                        f"(after {self.obstacle_block_counter} confirmed ticks)"
                    )

                # size offset to actually clear the obstacle, capped at max
                required_m = perp_m + self.car_half_width + 0.05
                mag = max(self.avoidance_offset_m, required_m)
                mag = min(mag, self.max_avoidance_offset_m)
                # obstacle RIGHT → +offset (shift LEFT); LEFT → -offset
                self.target_offset = +mag if self.obstacle_side_lock else -mag
        else:
            self.obstacle_detected = False
            self.obstacle_distance = 999.0
            self.obstacle_block_counter = 0
            self.obstacle_clear_counter += 1

        # ramp offset toward target_offset (smooth)
        delta = self.target_offset - self.avoidance_offset
        if abs(delta) <= self.offset_ramp_per_tick:
            self.avoidance_offset = self.target_offset
        else:
            self.avoidance_offset += math.copysign(self.offset_ramp_per_tick, delta)

        # always stanley — avoidance is just the offset on cte
        self._publish_target(msg.header.frame_id, msg.header.stamp,
                             self.goal_pos, avoiding=(abs(self.avoidance_offset) > 0.01))
        self.drive_to_target_stanley()

    def _to_grid(self, x, y):
        i = int(x * -self.CELLS_PER_METER + (self.grid_height - 1))
        j = int(y * self.CELLS_PER_METER + self.CELL_Y_OFFSET)
        return (i, j)

    def _from_grid(self, point):
        x = (point[0] - (self.grid_height - 1)) / -self.CELLS_PER_METER
        y = (point[1] - self.CELL_Y_OFFSET) / self.CELLS_PER_METER
        return (x, y)

    def _populate_grid(self, ranges, angle_increment, angle_min):
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), self.IS_FREE, dtype=int)
        ranges = np.array(ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = 0.0
        indices = np.arange(len(ranges))
        thetas = angle_min + indices * angle_increment
        xs = ranges * np.cos(thetas)
        ys = ranges * np.sin(thetas)
        forward = (xs > 0.2) & (xs < self.obstacle_detect_lookahead)
        half_w = self.grid_width_meters / 2
        lateral = (ys > -half_w) & (ys < half_w)
        in_range = forward & lateral
        i = np.round(xs * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
        j = np.round(ys * self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)
        valid = in_range & (i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width)
        self.occupancy_grid[i[valid], j[valid]] = self.IS_OCCUPIED

    def _convolve_grid(self):
        kernel = np.ones((2, 2))
        self.occupancy_grid = scipy_signal.convolve2d(
            self.occupancy_grid.astype("int"), kernel.astype("int"),
            boundary="symm", mode="same",
        )
        self.occupancy_grid = np.clip(self.occupancy_grid, -1, 100)

    def _publish_grid(self, frame_id, stamp):
        oc = OccupancyGrid()
        oc.header.frame_id = frame_id
        oc.header.stamp = stamp
        oc.info.origin.position.y -= ((self.grid_width / 2) + 1) / self.CELLS_PER_METER
        oc.info.width = self.grid_height
        oc.info.height = self.grid_width
        oc.info.resolution = 1 / self.CELLS_PER_METER
        oc.data = np.rot90(self.occupancy_grid, k=1).flatten().tolist()
        self.grid_pub.publish(oc)

    def _check_area(self, center, radius):
        """Check if any cell in a square region around center is occupied."""
        ci, cj = int(center[0]), int(center[1])
        i_min = max(ci - radius, 0)
        i_max = min(ci + radius + 1, self.grid_height)
        j_min = max(cj - radius, 0)
        j_max = min(cj + radius + 1, self.grid_width)
        if i_min >= i_max or j_min >= j_max:
            return True
        return np.any(self.occupancy_grid[i_min:i_max, j_min:j_max] >= self.IS_OCCUPIED)

    def _check_collision(self, cell_a, cell_b, margin=0):
        for i in range(-margin, margin + 1):
            a = (cell_a[0], cell_a[1] + i)
            b = (cell_b[0], cell_b[1] + i)
            for cell in self._traverse_grid(a, b):
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.grid_height or cell[1] >= self.grid_width:
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True
                except:
                    return True
        return False

    def _obstacle_on_right_of_path(self, prev_cell, next_cell, obstacle_cell):
        """[ADAPT-6] Side detection relative to PATH direction (not car frame).

        Given the path segment that was blocked AND the actual obstacle
        cell that triggered the block, return True if the obstacle is on
        the RIGHT side of the path's direction of travel.

        Grid convention: i decreases as we go forward, j increases as we
        go LEFT. So if the path segment goes from prev (high i) to next
        (lower i), forward direction in grid is (di, dj) where di < 0.

        The LEFT perpendicular of (di, dj) in this grid is (-dj, -di).
        Sign of (obstacle - midpoint) · perp_left > 0 means obstacle on LEFT.
        """
        di = float(next_cell[0] - prev_cell[0])
        dj = float(next_cell[1] - prev_cell[1])
        if di == 0.0 and dj == 0.0:
            # Degenerate (path went nowhere) — fall back to car-frame compare
            return obstacle_cell[1] < (self.grid_width // 2)
        mid_i = (prev_cell[0] + next_cell[0]) / 2.0
        mid_j = (prev_cell[1] + next_cell[1]) / 2.0
        d_i = float(obstacle_cell[0]) - mid_i
        d_j = float(obstacle_cell[1]) - mid_j
        # Perpendicular pointing LEFT relative to path direction (in grid):
        perp_i = -dj
        perp_j = -di
        signed = d_i * perp_i + d_j * perp_j
        # signed > 0 => obstacle is LEFT of path; signed < 0 => RIGHT
        return signed < 0.0

    def _check_collision_with_obstacle(self, cell_a, cell_b, margin=0):
        """Same as _check_collision but returns (collision, obstacle_cell).
        obstacle_cell is the first occupied cell encountered, in grid coords."""
        for i in range(-margin, margin + 1):
            a = (cell_a[0], cell_a[1] + i)
            b = (cell_b[0], cell_b[1] + i)
            for cell in self._traverse_grid(a, b):
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.grid_height or cell[1] >= self.grid_width:
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True, cell
                except:
                    return True, cell
        return False, None

    def _check_collision_loose(self, cell_a, cell_b, margin=0):
        for i in range(-margin, margin + 1):
            mid_a = (int((cell_a[0] + cell_b[0]) / 2), int((cell_a[1] + cell_b[1]) / 2) + i)
            b = (cell_b[0], cell_b[1] + i)
            for cell in self._traverse_grid(mid_a, b):
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.grid_height or cell[1] >= self.grid_width:
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True
                except:
                    return True
        return False

    def _traverse_grid(self, start, end):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        dx = x2 - x1
        dy = y2 - y1
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            points.append((y, x) if is_steep else (x, y))
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points

    def _publish_target(self, frame_id, stamp, point, avoiding):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        if avoiding:
            marker.color.r = 1.0
        else:
            marker.color.g = 1.0
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        self.target_pub.publish(marker)


def _send_stop(node):
    """[ADAPT-10] Publish (0, 0) several times so the VESC + servo
    actually receive the stop command before the node tears down."""
    msg = AckermannDriveStamped()
    msg.drive.speed = 0.0
    msg.drive.steering_angle = 0.0
    for _ in range(5):
        node.drive_pub.publish(msg)
        time.sleep(0.02)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyAvoidance()

    def _on_term(signum, _frame):
        try:
            _send_stop(node)
        except Exception:
            pass
        if signum == signal.SIGTSTP:
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTSTP)
        else:
            rclpy.try_shutdown()

    # Override rclpy's default handlers AFTER rclpy.init() so we run on Ctrl+C
    signal.signal(signal.SIGINT,  _on_term)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGTSTP, _on_term)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            _send_stop(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
