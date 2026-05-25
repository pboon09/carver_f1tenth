#!/usr/bin/python3

import numpy as np
from math import radians, cos, pi
import signal

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import qos_profile_sensor_data


class GapFollow(Node):
    def __init__(self):
        super().__init__("gap_follow_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("max_speed", 0.7
                               )
        self.declare_parameter("max_steering_angle", 0.4189)
        self.declare_parameter("steering_smoothing", 0.65)
        self.declare_parameter("car_width", 0.33)
        self.declare_parameter("min_distance_threshold", 1.2)
        self.declare_parameter("obstacle_inflated", 30)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_steering_angle = float(self.get_parameter("max_steering_angle").value)
        self.steering_smoothing = float(self.get_parameter("steering_smoothing").value)
        self.car_width = float(self.get_parameter("car_width").value)
        self.min_distance_threshold = float(self.get_parameter("min_distance_threshold").value)
        self.obstacle_inflated = int(self.get_parameter("obstacle_inflated").value)

        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        self.previous_steering = 0.0
        self.fov_min = -90
        self.fov_max = 90

        self.declare_parameter("log_every_n", 10)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))
        self._scan_count = 0

        # shutdown hook
        signal.signal(signal.SIGINT, self._shutdown_handler)

        # self.get_logger().info("=" * 60)
        # self.get_logger().info("Gap Follow Node Initialized")
        # self.get_logger().info(f"  Speed range     : {self.min_speed} - {self.max_speed} m/s")
        # self.get_logger().info(f"  Max steer       : {np.degrees(self.max_steering_angle):.1f} deg")
        # self.get_logger().info(f"  Car width       : {self.car_width} m")
        # self.get_logger().info(f"  Bubble range    : {self.min_distance_threshold} m")
        # self.get_logger().info(f"  Steering smooth : {self.steering_smoothing}")
        # self.get_logger().info(f"  FOV             : {self.fov_min} to {self.fov_max} deg")
        # self.get_logger().info("=" * 60)

    def _shutdown_handler(self, sig, frame):
        self.get_logger().warn("Ctrl+C detected — sending zero command and shutting down")
        self._publish_stop()
        rclpy.shutdown()

    def _publish_stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)
        self.get_logger().info("Stop command published (speed=0, steer=0)")

    def angle_to_index(self, angle, angle_min, angle_increment):
        angle_rad = radians(angle)
        return int((angle_rad - angle_min) / angle_increment)

    def angle_to_index_rad(self, angle_rad, angle_min, angle_increment):
        return int((angle_rad - angle_min) / angle_increment)

    def index_to_rad(self, idx, angle_min, angle_increment):
        return angle_min + idx * angle_increment

    def create_bubble_radius(self, distance_to_obstacle):
        if distance_to_obstacle < 1.5:
            return self.car_width * 1.3
        elif distance_to_obstacle < 2.0:
            return self.car_width * 1.0
        else:
            return self.car_width * 0.9

    def get_gap_score(self, gap_avg_distance, gap_max_distance, gap_start_idx, gap_end_idx, num_ranges):
        gap_width = gap_end_idx - gap_start_idx + 1
        width_factor = min(gap_width / 30.0, 1.0)
        gap_score = (gap_max_distance * 0.3 + gap_avg_distance * 0.5) * width_factor
        center_idx = num_ranges // 2
        gap_center_idx = (gap_start_idx + gap_end_idx) // 2
        center_score = 1.0 - 0.3 * (abs(center_idx - gap_center_idx) / center_idx)
        return gap_score * center_score

    def get_clearance_distance(self, processed_ranges, heading_direction, angle_min, angle_increment):
        turn_threshold = 0.1
        sweep_angle = 45

        if heading_direction > turn_threshold:
            window_start_angle = heading_direction
            window_end_angle = heading_direction + radians(sweep_angle)
        elif heading_direction < -turn_threshold:
            window_start_angle = heading_direction - radians(sweep_angle)
            window_end_angle = heading_direction
        else:
            window_start_angle = heading_direction - radians(30)
            window_end_angle = heading_direction + radians(30)

        # fixed: use rad version to avoid double-radians bug
        start_idx = self.angle_to_index_rad(window_start_angle, angle_min, angle_increment)
        end_idx = self.angle_to_index_rad(window_end_angle, angle_min, angle_increment)

        start_idx = max(0, min(start_idx, len(processed_ranges) - 1))
        end_idx = max(0, min(end_idx, len(processed_ranges) - 1))

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        if start_idx <= end_idx and end_idx < len(processed_ranges):
            clearance_window = processed_ranges[start_idx : end_idx + 1]
            valid_ranges = clearance_window[(clearance_window > 0) & (clearance_window != float('inf'))]
            if len(valid_ranges) > 0:
                return np.min(valid_ranges)

        return 1.0

    def linear_velocity_controller(self, delta_distance, steering_angle):
        if delta_distance < 0.3:
            speed = 0.0
        elif delta_distance < 0.8:
            speed = self.min_speed
        elif delta_distance > 2.5:
            speed = self.max_speed
        else:
            t = (delta_distance - 0.8) / (2.5 - 0.8)
            speed = self.min_speed + t * (self.max_speed - self.min_speed)

        speed = np.clip(speed, self.min_speed, self.max_speed)
        return speed

    def steering_controller(self, heading_direction, speed):
        desired_steering = np.clip(heading_direction, -self.max_steering_angle, self.max_steering_angle)
        smoothed_steering = (
            self.steering_smoothing * desired_steering +
            (1 - self.steering_smoothing) * self.previous_steering
        )
        self.previous_steering = smoothed_steering
        return smoothed_steering

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=float('inf'), posinf=float('inf'))

        total_points = len(ranges)
        valid_raw = np.sum((ranges > 0) & (ranges != float('inf')))

        fov_min_idx = self.angle_to_index(self.fov_min, msg.angle_min, msg.angle_increment)
        fov_max_idx = self.angle_to_index(self.fov_max, msg.angle_min, msg.angle_increment)
        fov_points = fov_max_idx - fov_min_idx + 1

        processed_ranges = ranges.copy()
        obstacle_idxs = np.where((processed_ranges > 0) & (processed_ranges <= self.min_distance_threshold))[0]
        num_obstacles = len(obstacle_idxs)

        n = len(processed_ranges)
        if num_obstacles > 0:
            obs_dist = ranges[obstacle_idxs]

            # Per-obstacle bubble radius (metres). Mirrors create_bubble_radius:
            #   d < 1.0 → 1.1·w,  d < 2.0 → 1.0·w,  else → 0.9·w
            bubble_m = np.where(obs_dist < 1.0, self.car_width * 1.1,
                       np.where(obs_dist < 2.0, self.car_width * 1.0,
                                                self.car_width * 0.9))

            # Convert metres → index half-width via arc length. For d ≤ 0.1 m
            # the original code forced inflation_size = 60.
            safe_dist = np.maximum(obs_dist, 1e-6)
            arc_len = safe_dist * msg.angle_increment
            inflation = np.where(obs_dist > 0.1,
                                 (bubble_m / arc_len).astype(np.int32),
                                 60)
            inflation = np.minimum(inflation, 60)

            # Boolean mask of every index inside any obstacle's bubble.
            kill = np.zeros(n, dtype=bool)
            for obs_idx, infl in zip(obstacle_idxs, inflation):
                lo = max(fov_min_idx, obs_idx - int(infl))
                hi = min(fov_max_idx, obs_idx + int(infl))
                if lo <= hi:
                    kill[lo:hi + 1] = True

            total_zeroed = int(np.count_nonzero(kill & (processed_ranges != 0)))
            processed_ranges[kill] = 0
            bubble_sizes = inflation.tolist()
        else:
            total_zeroed = 0
            bubble_sizes = []

        fov_slice = processed_ranges[fov_min_idx:fov_max_idx + 1]
        valid_after_bubble = int(np.sum((fov_slice > 0) & (fov_slice != float('inf'))))

        # gap finding
        process_idx = fov_min_idx
        safest_score = float('-inf')
        heading_direction = 0.0
        num_gaps = 0
        best_gap_width = 0
        best_gap_distance = 0.0
        best_gap_physical_width = 0.0
        all_gaps = []

        while process_idx <= fov_max_idx:
            if processed_ranges[process_idx] > 0:
                gap_distances_sum = 0
                gap_min_distance = float('inf')
                gap_max_distance = float('-inf')
                gap_start_idx = process_idx

                while process_idx <= fov_max_idx and processed_ranges[process_idx] > 0:
                    current_distance = processed_ranges[process_idx]
                    gap_distances_sum += current_distance
                    gap_min_distance = min(gap_min_distance, current_distance)
                    gap_max_distance = max(gap_max_distance, current_distance)
                    process_idx += 1

                gap_end_idx = process_idx - 1
                gap_width_indices = gap_end_idx - gap_start_idx + 1
                gap_avg_distance = gap_distances_sum / gap_width_indices

                # physical width of gap at avg distance
                gap_physical_width = gap_width_indices * msg.angle_increment * gap_avg_distance

                if gap_width_indices >= 3 and gap_min_distance > 0.3:
                    num_gaps += 1
                    gap_score = self.get_gap_score(
                        gap_avg_distance, gap_max_distance, gap_start_idx, gap_end_idx, len(ranges)
                    )
                    all_gaps.append({
                        "score": gap_score,
                        "width_idx": gap_width_indices,
                        "width_m": gap_physical_width,
                        "avg_dist": gap_avg_distance,
                        "max_dist": gap_max_distance,
                    })

                    if gap_score > safest_score:
                        safest_score = gap_score
                        gap_center_idx = (gap_start_idx + gap_end_idx) // 2
                        heading_direction = self.index_to_rad(gap_center_idx, msg.angle_min, msg.angle_increment)
                        best_gap_width = gap_width_indices
                        best_gap_distance = gap_avg_distance
                        best_gap_physical_width = gap_physical_width
            else:
                process_idx += 1

        delta_distance = self.get_clearance_distance(
            processed_ranges, heading_direction, msg.angle_min, msg.angle_increment
        )
        if delta_distance == 0 or delta_distance == float('inf'):
            delta_distance = 1.0

        speed = self.linear_velocity_controller(delta_distance, heading_direction)
        steering_angle = self.steering_controller(heading_direction, speed)

        self._scan_count += 1
        if self._scan_count % self.log_every_n == 0:
            self.get_logger().info(
                f"[SCAN ] total={total_points} | valid={valid_raw} | FOV_pts={fov_points}"
            )
            self.get_logger().info(
                f"[BUBBL] obstacles={num_obstacles} | "
                f"avg_bubble={int(np.mean(bubble_sizes)) if bubble_sizes else 0} idx | "
                f"zeroed={total_zeroed} pts | "
                f"remaining={valid_after_bubble} pts"
            )
            if bubble_sizes:
                self.get_logger().info(
                    f"[BUBBL] sizes min={min(bubble_sizes)} max={max(bubble_sizes)} "
                    f"(cap=60)"
                )

            self.get_logger().info(
                f"[GAP  ] found={num_gaps} gaps | "
                f"best: {best_gap_width} idx wide | "
                f"~{best_gap_physical_width:.2f}m wide | "
                f"avg_dist={best_gap_distance:.2f}m | "
                f"score={safest_score:.3f}"
            )
            if len(all_gaps) > 1:
                sorted_gaps = sorted(all_gaps, key=lambda g: g["score"], reverse=True)
                runner = sorted_gaps[1]
                self.get_logger().info(
                    f"[GAP  ] runner-up: {runner['width_idx']} idx | "
                    f"~{runner['width_m']:.2f}m | score={runner['score']:.3f}"
                )

            self.get_logger().info(
                f"[HEAD ] heading={np.degrees(heading_direction):+.1f} deg | "
                f"clearance={delta_distance:.2f}m"
            )
            self.get_logger().info(
                f"[DRIVE] speed={speed:.3f} m/s | "
                f"steer_raw={np.degrees(heading_direction):+.1f} deg | "
                f"steer_smoothed={np.degrees(steering_angle):+.1f} deg | "
                f"prev_steer={np.degrees(self.previous_steering):+.1f} deg"
            )

        if num_gaps > 0 and best_gap_physical_width < self.car_width:
            self.get_logger().warn(
                f"[WARN ] Best gap ({best_gap_physical_width:.2f}m) is NARROWER than car ({self.car_width}m)!"
            )
        if num_gaps == 0:
            self.get_logger().warn("[WARN ] No valid gaps found — car may be stuck or fully blocked")
        if valid_after_bubble < 10:
            self.get_logger().warn(f"[WARN ] Only {valid_after_bubble} pts remain after bubble inflation — bubbles may be too large")

        msg_drive = AckermannDriveStamped()
        msg_drive.header.stamp = self.get_clock().now().to_msg()
        msg_drive.drive.speed = float(speed)
        msg_drive.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg_drive)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()