#!/usr/bin/python3

import os

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory


class StanleyViz(Node):
    def __init__(self):
        super().__init__("stanley_viz_node")

        self.declare_parameter("waypoints_path", "")

        self.waypoints = self._load_waypoints()
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints for viz")

        self.path_pub = self.create_publisher(MarkerArray, "/viz/path", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10
        )

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
        marker_array = MarkerArray()

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "raceline"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color.r = 0.0
        line.color.g = 0.5
        line.color.b = 1.0
        line.color.a = 0.6
        line.lifetime = Duration(seconds=2).to_msg()

        from geometry_msgs.msg import Point
        for wp in self.waypoints:
            p = Point()
            p.x = float(wp["x"])
            p.y = float(wp["y"])
            p.z = 0.0
            line.points.append(p)

        close = Point()
        close.x = float(self.waypoints[0]["x"])
        close.y = float(self.waypoints[0]["y"])
        close.z = 0.0
        line.points.append(close)

        marker_array.markers.append(line)

        for i, wp in enumerate(self.waypoints):
            if i % 20 != 0:
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "waypoint_dots"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 0.08
            m.color.r = 0.0
            m.color.g = 0.5
            m.color.b = 1.0
            m.color.a = 0.4
            m.lifetime = Duration(seconds=2).to_msg()
            m.pose.position.x = float(wp["x"])
            m.pose.position.y = float(wp["y"])
            m.pose.position.z = 0.0
            marker_array.markers.append(m)

        self.path_pub.publish(marker_array)

    def odom_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = StanleyViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
