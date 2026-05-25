#!/usr/bin/python3

import os

import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory


class ControllerViz(Node):
    def __init__(self):
        super().__init__("viz_node")

        self.declare_parameter("waypoints_path", "")

        self.waypoints = self._load_waypoints()
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints for viz")

        self.path_pub = self.create_publisher(Path, "/viz/path", 10)

        self.create_timer(1.0, self.publish_path)

    def _load_waypoints(self):
        waypoints_path = str(self.get_parameter("waypoints_path").value)

        if not waypoints_path:
            pkg_share = get_package_share_directory("f1tenth_controller")
            waypoints_path = os.path.join(pkg_share, "path", "path.yaml")

        with open(waypoints_path, "r") as f:
            data = yaml.safe_load(f)

        return data["waypoints"]

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(wp["x"])
            pose.pose.position.y = float(wp["y"])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        # Close the loop
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = path_msg.header.stamp
        pose.pose.position.x = float(self.waypoints[0]["x"])
        pose.pose.position.y = float(self.waypoints[0]["y"])
        pose.pose.position.z = 0.0
        path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
