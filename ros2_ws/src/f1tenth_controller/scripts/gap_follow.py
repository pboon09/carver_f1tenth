#!/usr/bin/python3

import numpy as np
from math import radians, cos, pi

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class GapFollow(Node):
    def __init__(self):
        super().__init__("gap_follow_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("max_speed", 1.5)
        self.declare_parameter("max_steering_angle", 0.35)
        self.declare_parameter("steering_smoothing", 0.4)
        self.declare_parameter("car_width", 0.2)
        self.declare_parameter("min_distance_threshold", 2.5)
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
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        self.previous_steering = 0.0
        self.fov_min = -120
        self.fov_max = 120

        self.get_logger().info(f"Gap Follow initialized - speed: {self.min_speed}-{self.max_speed} m/s")

    def angle_to_index(self, angle, angle_min, angle_increment):
        angle_rad = radians(angle)
        return int((angle_rad - angle_min) / angle_increment)

    def index_to_rad(self, idx, angle_min, angle_increment):
        return angle_min + idx * angle_increment

    def create_bubble_radius(self, distance_to_obstacle):
        if distance_to_obstacle < 1.0:
            return self.car_width * 2.0
        elif distance_to_obstacle < 2.0:
            return self.car_width * 1.5
        else:
            return self.car_width * 1.2

    def get_gap_score(self, gap_avg_distance, gap_max_distance, gap_start_idx, gap_end_idx, num_ranges):
        gap_score = gap_max_distance * 0.4 + gap_avg_distance * 0.6
        center_idx = num_ranges // 2
        gap_center_idx = (gap_start_idx + gap_end_idx) // 2
        center_score = 1.0 - 0.3 * (abs(center_idx - gap_center_idx) / center_idx)
        gap_score *= center_score
        return gap_score

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

        start_idx = self.angle_to_index(window_start_angle, angle_min, angle_increment)
        end_idx = self.angle_to_index(window_end_angle, angle_min, angle_increment)

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

    def linear_velocity_controller(self, delta_distance, steering_angle, K_p=1.0):
        steering_factor = cos(abs(steering_angle))

        if delta_distance < 1.5:
            speed = self.min_speed
        elif delta_distance > 4.0:
            speed = self.max_speed
        else:
            speed = K_p * delta_distance

        speed = speed * (0.4 + 0.6 * steering_factor)
        speed = np.clip(speed, self.min_speed, self.max_speed)
        return speed

    def steering_controller(self, heading_direction, speed):
        desired_steering = np.clip(heading_direction, -self.max_steering_angle, self.max_steering_angle)
        smoothed_steering = (
            self.steering_smoothing * desired_steering +
            (1 - self.steering_smoothing) * self.previous_steering
        )
        speed_factor = 1.0 - (speed - self.min_speed) / (self.max_speed - self.min_speed) * 0.5
        final_steering = smoothed_steering * speed_factor
        self.previous_steering = final_steering
        return final_steering

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=float('inf'), posinf=float('inf'))

        fov_min_idx = self.angle_to_index(self.fov_min, msg.angle_min, msg.angle_increment)
        fov_max_idx = self.angle_to_index(self.fov_max, msg.angle_min, msg.angle_increment)

        processed_ranges = ranges.copy()
        obstacle_idxs = np.where((processed_ranges > 0) & (processed_ranges <= self.min_distance_threshold))[0]

        for obs_idx in obstacle_idxs:
            distance_to_object = ranges[obs_idx]

            if distance_to_object > 0.1:
                bubble_radius = self.create_bubble_radius(distance_to_object)
                arc_length_from_car = distance_to_object * msg.angle_increment
                angular_bubble = int(bubble_radius / arc_length_from_car)
                inflation_size = min(max(self.obstacle_inflated, angular_bubble), 60)
            else:
                inflation_size = self.obstacle_inflated * 2

            for offset_idx in range(-inflation_size, inflation_size + 1):
                idx = obs_idx + offset_idx
                if fov_min_idx <= idx <= fov_max_idx:
                    processed_ranges[idx] = 0

        process_idx = fov_min_idx
        safest_score = float('-inf')
        heading_direction = 0.0

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

                if gap_width_indices >= 3 and gap_min_distance > 0.3:
                    gap_score = self.get_gap_score(
                        gap_avg_distance, gap_max_distance, gap_start_idx, gap_end_idx, len(ranges)
                    )

                    if gap_score > safest_score:
                        safest_score = gap_score
                        gap_center_idx = (gap_start_idx + gap_end_idx) // 2
                        heading_direction = self.index_to_rad(gap_center_idx, msg.angle_min, msg.angle_increment)
            else:
                process_idx += 1

        delta_distance = self.get_clearance_distance(processed_ranges, heading_direction, msg.angle_min, msg.angle_increment)
        if delta_distance == 0 or delta_distance == float('inf'):
            delta_distance = 1.0

        speed = self.linear_velocity_controller(delta_distance, heading_direction, K_p=1.0)
        steering_angle = self.steering_controller(heading_direction, speed)

        self.get_logger().info(
            f"Heading: {np.degrees(heading_direction):.1f}° | "
            f"Safety: {safest_score:.2f} | Speed: {speed:.2f} m/s | Clearance: {delta_distance:.2f}m"
        )

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
