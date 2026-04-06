#!/usr/bin/python3

import math
import os

import numpy as np
import yaml
from scipy import signal
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory


class StanleyAvoidance(Node):
    def __init__(self):
        super().__init__("stanley_avoidance_node")

        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("velocity", 1.5)
        self.declare_parameter("K_E", 2.0)
        self.declare_parameter("K_H", 1.5)
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("K_p_obstacle", 0.8)
        self.declare_parameter("min_lookahead", 1.0)
        self.declare_parameter("max_lookahead", 3.0)
        self.declare_parameter("min_lookahead_speed", 3.0)
        self.declare_parameter("max_lookahead_speed", 6.0)
        self.declare_parameter("velocity_percentage", 0.5)
        self.declare_parameter("velocity_min", 0.5)
        self.declare_parameter("velocity_max", 2.0)
        self.declare_parameter("steering_limit", 25.0)
        self.declare_parameter("grid_width_meters", 6.0)
        self.declare_parameter("cells_per_meter", 10)
        self.declare_parameter("wheelbase", 0.33)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.grid_width_meters = float(self.get_parameter("grid_width_meters").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.base_velocity = float(self.get_parameter("velocity").value)

        self.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)
        self.L = self.max_lookahead

        self.waypoints_world, self.velocities = self._load_waypoints()
        self.get_logger().info(f"Loaded {len(self.waypoints_world)} waypoints")

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.target_pub = self.create_publisher(Marker, "/viz/drive_target", 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)

        self.grid_height = int(self.L * self.CELLS_PER_METER)
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
        self.velocity_index = 0
        self.index = 0
        self.odom_frame = "map"

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

        return points, velocities

    def _transform_waypoints(self, waypoints, position, pose):
        translated = waypoints - np.array(position)
        quaternion = np.array([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ])
        return R.inv(R.from_quat(quaternion)).apply(translated)

    def _get_closest_waypoint_with_velocity(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        self.velocity_index = np.argmin(distances)
        return self.waypoints_world[self.velocity_index], self.velocities[self.velocity_index]

    def _get_waypoint(self, pose, target_velocity):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)

        self.L = min(
            max(
                self.min_lookahead,
                self.min_lookahead
                + (self.max_lookahead - self.min_lookahead)
                * (target_velocity - self.min_lookahead_speed)
                / max(self.max_lookahead_speed - self.min_lookahead_speed, 0.01),
            ),
            self.max_lookahead,
        )

        indices_L = np.argsort(np.where(distances < self.L, distances, -1))[::-1]

        for i in indices_L:
            if waypoints_car[i][0] > 0:
                self.index = i
                return waypoints_car[self.index], self.waypoints_world[self.index]
        return None, None

    def _get_waypoint_stanley(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        index = np.argmin(distances)
        return waypoints_car[index], self.waypoints_world[index]

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.odom_frame = msg.header.frame_id

        current_pose_quaternion = np.array([
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w,
        ])

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
        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))

        if self.obstacle_detected and self.velocity_percentage > 0.0:
            if abs(np.degrees(angle)) < 10.0:
                velocity = self.velocity_max
            elif abs(np.degrees(angle)) < 20.0:
                velocity = (self.velocity_max + self.velocity_min) / 2
            else:
                velocity = self.velocity_min
        else:
            velocity = self.target_velocity * self.velocity_percentage

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)
        return angle, velocity

    def drive_to_target_stanley(self):
        closest_wheelbase_front_point_car, closest_wheelbase_front_point_world = self._get_waypoint_stanley(
            self.current_pose_wheelbase_front
        )

        path_heading = math.atan2(
            closest_wheelbase_front_point_world[1] - self.closest_wheelbase_rear_point[1],
            closest_wheelbase_front_point_world[0] - self.closest_wheelbase_rear_point[0],
        )
        current_heading = math.atan2(
            self.current_pose_wheelbase_front.position.y - self.current_pose.position.y,
            self.current_pose_wheelbase_front.position.x - self.current_pose.position.x,
        )

        if current_heading < 0:
            current_heading += 2 * math.pi
        if path_heading < 0:
            path_heading += 2 * math.pi

        crosstrack_error = math.atan2(self.K_E * closest_wheelbase_front_point_car[1], self.target_velocity)
        heading_error = path_heading - current_heading
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi
        heading_error *= self.K_H

        angle = heading_error + crosstrack_error
        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))

        velocity = self.target_velocity * self.velocity_percentage

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)
        return angle, velocity

    def scan_callback(self, msg):
        if self.current_pose is None or self.goal_pos is None:
            return

        self._populate_grid(msg.ranges, msg.angle_increment, msg.angle_min)
        self._convolve_grid()
        self._publish_grid(msg.header.frame_id, msg.header.stamp)

        current_pos = np.array(self._to_grid(0, 0))
        goal_pos = np.array(self._to_grid(self.goal_pos[0], self.goal_pos[1]))
        target = None
        MARGIN = int(self.CELLS_PER_METER * 0.15)

        if self._check_collision(current_pos, goal_pos, margin=MARGIN):
            self.obstacle_detected = True
            shifts = [i * (-1 if i % 2 else 1) for i in range(1, 21)]

            found = False
            for shift in shifts:
                new_goal = goal_pos + np.array([0, shift])
                if not self._check_collision(current_pos, new_goal, margin=int(1.5 * MARGIN)):
                    target = self._from_grid(new_goal)
                    found = True
                    break

            if not found:
                middle_grid_point = np.array(current_pos + (goal_pos - current_pos) / 2).astype(int)
                for shift in shifts:
                    new_goal = middle_grid_point + np.array([0, shift])
                    if not self._check_collision(current_pos, new_goal, margin=int(1.5 * MARGIN)):
                        target = self._from_grid(new_goal)
                        found = True
                        break

            if not found:
                middle_grid_point = np.array(current_pos + (goal_pos - current_pos) / 2).astype(int)
                for shift in shifts:
                    new_goal = middle_grid_point + np.array([0, shift])
                    if not self._check_collision_loose(current_pos, new_goal, margin=MARGIN):
                        target = self._from_grid(new_goal)
                        found = True
                        break
        else:
            self.obstacle_detected = False

        if target:
            self._publish_target(msg.header.frame_id, msg.header.stamp, target, avoiding=True)
            self.drive_to_target(target, self.K_p_obstacle)
        elif self.obstacle_detected:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
        else:
            self._publish_target(msg.header.frame_id, msg.header.stamp, self.goal_pos, avoiding=False)
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
        forward = (xs > 0.2) & (xs < self.L)
        half_w = self.grid_width_meters / 2
        lateral = (ys > -half_w) & (ys < half_w)
        in_range = forward & lateral
        i = np.round(xs * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
        j = np.round(ys * self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)
        valid = in_range & (i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width)
        self.occupancy_grid[i[valid], j[valid]] = self.IS_OCCUPIED

    def _convolve_grid(self):
        kernel = np.ones((2, 2))
        self.occupancy_grid = signal.convolve2d(
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


def main(args=None):
    rclpy.init(args=args)
    node = StanleyAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
