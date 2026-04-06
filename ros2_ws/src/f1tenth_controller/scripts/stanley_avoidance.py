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
        self.waypoint_pub = self.create_publisher(Marker, "/current_waypoint", 10)
        self.lookahead_pub = self.create_publisher(Marker, "/lookahead_waypoint", 10)
        self.path_pub = self.create_publisher(MarkerArray, "/stanley_path", 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)

        self.grid_height = int(self.L * self.CELLS_PER_METER)
        self.grid_width = int(self.grid_width_meters * self.CELLS_PER_METER)
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0, dtype=int)

        self.current_pose = None
        self.current_pose_front = None
        self.closest_rear_point = None
        self.goal_pos = None
        self.target_velocity = 0.0
        self.obstacle_detected = False
        self.velocity_index = 0

        self.IS_OCCUPIED = 100
        self.IS_FREE = 0

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
            pose.orientation.z, pose.orientation.w
        ])
        return R.inv(R.from_quat(quaternion)).apply(translated)

    def _get_closest_waypoint(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        self.velocity_index = np.argmin(distances)
        return self.waypoints_world[self.velocity_index], self.velocities[self.velocity_index]

    def _get_stanley_waypoint(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)
        index = np.argmin(distances)
        return waypoints_car[index], self.waypoints_world[index]

    def _get_lookahead_waypoint(self, pose):
        position = (pose.position.x, pose.position.y, 0)
        waypoints_car = self._transform_waypoints(self.waypoints_world, position, pose)
        distances = np.linalg.norm(waypoints_car, axis=1)

        self.L = min(max(
            self.min_lookahead,
            self.min_lookahead + (self.max_lookahead - self.min_lookahead)
            * (self.target_velocity - self.min_lookahead_speed)
            / max(self.max_lookahead_speed - self.min_lookahead_speed, 0.01)
        ), self.max_lookahead)

        indices = np.argsort(np.where(distances < self.L, distances, -1))[::-1]
        for i in indices:
            if waypoints_car[i][0] > 0:
                return waypoints_car[i], self.waypoints_world[i]
        return None, None

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

        quat = np.array([
            self.current_pose.orientation.x, self.current_pose.orientation.y,
            self.current_pose.orientation.z, self.current_pose.orientation.w
        ])
        front_offset = R.from_quat(quat).apply((self.wheelbase, 0, 0))
        self.current_pose_front = Pose()
        self.current_pose_front.position.x = self.current_pose.position.x + front_offset[0]
        self.current_pose_front.position.y = self.current_pose.position.y + front_offset[1]
        self.current_pose_front.position.z = 0.0
        self.current_pose_front.orientation = self.current_pose.orientation

        self.closest_rear_point, self.target_velocity = self._get_closest_waypoint(self.current_pose)
        self.goal_pos, _ = self._get_lookahead_waypoint(self.current_pose)

        self._draw_marker(
            msg.header.frame_id, msg.header.stamp,
            self.closest_rear_point, self.waypoint_pub, "blue"
        )

    def drive_stanley(self):
        closest_car, closest_world = self._get_stanley_waypoint(self.current_pose_front)

        path_heading = math.atan2(
            closest_world[1] - self.closest_rear_point[1],
            closest_world[0] - self.closest_rear_point[0]
        )
        current_heading = math.atan2(
            self.current_pose_front.position.y - self.current_pose.position.y,
            self.current_pose_front.position.x - self.current_pose.position.x
        )

        if current_heading < 0:
            current_heading += 2 * math.pi
        if path_heading < 0:
            path_heading += 2 * math.pi

        crosstrack_error = math.atan2(self.K_E * closest_car[1], max(self.target_velocity, 0.1))
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

    def drive_pure_pursuit(self, point, k_p):
        L = np.linalg.norm(point)
        y = point[1]
        angle = k_p * (2 * y) / (L ** 2)
        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))

        velocity = self.target_velocity * self.velocity_percentage

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, msg):
        if self.current_pose is None or self.goal_pos is None:
            return

        self._populate_grid(msg.ranges, msg.angle_increment, msg.angle_min)
        self._convolve_grid()
        self._publish_grid(msg.header.frame_id, msg.header.stamp)

        current_cell = np.array(self._to_grid(0, 0))
        goal_cell = np.array(self._to_grid(self.goal_pos[0], self.goal_pos[1]))
        target = None
        MARGIN = int(self.CELLS_PER_METER * 0.15)

        if self._check_collision(current_cell, goal_cell, MARGIN):
            self.obstacle_detected = True
            shifts = [i * (-1 if i % 2 else 1) for i in range(1, 21)]

            for shift in shifts:
                new_goal = goal_cell + np.array([0, shift])
                if not self._check_collision(current_cell, new_goal, int(1.5 * MARGIN)):
                    target = self._from_grid(new_goal)
                    break

            if target is None:
                mid = np.array(current_cell + (goal_cell - current_cell) / 2).astype(int)
                for shift in shifts:
                    new_goal = mid + np.array([0, shift])
                    if not self._check_collision(current_cell, new_goal, int(1.5 * MARGIN)):
                        target = self._from_grid(new_goal)
                        break

            if target is None:
                mid = np.array(current_cell + (goal_cell - current_cell) / 2).astype(int)
                for shift in shifts:
                    new_goal = mid + np.array([0, shift])
                    if not self._check_collision_loose(current_cell, new_goal, MARGIN):
                        target = self._from_grid(new_goal)
                        break
        else:
            self.obstacle_detected = False
            target = self._from_grid(goal_cell)

        if target is not None:
            if self.obstacle_detected:
                self.drive_pure_pursuit(target, self.K_p_obstacle)
            else:
                self.drive_stanley()
        else:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)

    def _to_grid(self, x, y):
        i = (self.grid_height - 1) - int(x * self.CELLS_PER_METER)
        j = self.CELL_Y_OFFSET - int(y * self.CELLS_PER_METER)
        return (i, j)

    def _from_grid(self, point):
        x = ((self.grid_height - 1) - point[0]) / self.CELLS_PER_METER
        y = (self.CELL_Y_OFFSET - point[1]) / self.CELLS_PER_METER
        return (x, y)

    def _populate_grid(self, ranges, angle_increment, angle_min):
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), self.IS_FREE, dtype=int)
        ranges = np.array(ranges)
        indices = np.arange(len(ranges))
        thetas = angle_min + indices * angle_increment
        xs = ranges * np.cos(thetas)
        ys = ranges * np.sin(thetas)
        i = np.round((self.grid_height - 1) - xs * self.CELLS_PER_METER).astype(int)
        j = np.round(self.CELL_Y_OFFSET - ys * self.CELLS_PER_METER).astype(int)
        valid = (i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width)
        self.occupancy_grid[i[valid], j[valid]] = self.IS_OCCUPIED

    def _convolve_grid(self):
        kernel = np.ones((2, 2))
        self.occupancy_grid = signal.convolve2d(
            self.occupancy_grid.astype("int"), kernel.astype("int"),
            boundary="symm", mode="same"
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
        oc.data = np.fliplr(np.rot90(self.occupancy_grid, k=1)).flatten().tolist()
        self.grid_pub.publish(oc)

    def _check_collision(self, cell_a, cell_b, margin=0):
        for i in range(-margin, margin + 1):
            a = (cell_a[0], cell_a[1] + i)
            b = (cell_b[0], cell_b[1] + i)
            for cell in self._traverse_grid(a, b):
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.grid_height or cell[1] >= self.grid_width:
                    continue
                if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                    return True
        return False

    def _check_collision_loose(self, cell_a, cell_b, margin=0):
        for i in range(-margin, margin + 1):
            mid_a = (int((cell_a[0] + cell_b[0]) / 2), int((cell_a[1] + cell_b[1]) / 2) + i)
            b = (cell_b[0], cell_b[1] + i)
            for cell in self._traverse_grid(mid_a, b):
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.grid_height or cell[1] >= self.grid_width:
                    continue
                if self.occupancy_grid[cell] == self.IS_OCCUPIED:
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

    def _draw_marker(self, frame_id, stamp, position, publisher, color="red"):
        if position is None:
            return
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
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
