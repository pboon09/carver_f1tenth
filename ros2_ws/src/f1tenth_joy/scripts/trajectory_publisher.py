#!/usr/bin/env python3
"""
Publish a nav_msgs/Path of the vehicle's map-frame trajectory plus a
nav_msgs/Odometry of just the front (latest) pose.

Polls TF for `map → base_link` at the configured rate, appends each pose
to a Path, and republishes the growing Path on /trajectory. The latest
sampled pose is also emitted as a single Odometry message on
/dedicate_odom — cheap, fixed-size pose stream for controllers / loggers
that don't want the 5000-pose Path bandwidth.

Topics out:
    /trajectory       nav_msgs/Path        (in `map` frame, grows)
    /dedicate_odom    nav_msgs/Odometry    (in `map` frame, single latest pose)

Params:
    parent_frame   (str, "map")             frame the path is expressed in
    child_frame    (str, "base_link")       frame whose pose is tracked
    rate           (double, 10.0)           Hz the path is sampled
    max_poses      (int, 5000)              ring-buffer cap (oldest dropped)
    min_distance   (double, 0.01)           only append to Path if moved > this much
                                            (does NOT throttle /dedicate_odom)
    publish_rate   (double, 10.0)           Hz the full path + odom are republished
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_publisher")

        self.declare_parameter("parent_frame", "map")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("max_poses", 5000)
        self.declare_parameter("min_distance", 0.01)

        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        sample_hz = float(self.get_parameter("rate").value)
        pub_hz = float(self.get_parameter("publish_rate").value)
        self.max_poses = int(self.get_parameter("max_poses").value)
        self.min_d = float(self.get_parameter("min_distance").value)

        self.path = Path()
        self.path.header.frame_id = self.parent
        self.latest_pose: PoseStamped | None = None   # most recent TF sample

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(Path, "/trajectory", 10)
        self.odom_pub = self.create_publisher(Odometry, "/dedicate_odom", 10)
        self.create_timer(1.0 / sample_hz, self._sample)
        self.create_timer(1.0 / pub_hz, self._publish)

        self.get_logger().info(
            f"trajectory_publisher up: "
            f"{self.parent} → {self.child}, "
            f"sample={sample_hz} Hz, publish={pub_hz} Hz, "
            f"max_poses={self.max_poses}, min_d={self.min_d} m; "
            f"out: /trajectory (Path), /dedicate_odom (Odometry)")

    def _sample(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.parent, self.child, Time(),
                timeout=Duration(seconds=0.05))
        except (LookupException, ExtrapolationException):
            return

        pose = PoseStamped()
        pose.header.stamp = t.header.stamp
        pose.header.frame_id = self.parent
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation

        # Always update the "front" pose — /dedicate_odom streams every sample
        # regardless of min_distance (it's a single fixed-size msg, no buffer cost).
        self.latest_pose = pose

        # De-duplicate small motions for the Path so the buffer doesn't fill up sitting still.
        if self.path.poses:
            last = self.path.poses[-1].pose.position
            dx = pose.pose.position.x - last.x
            dy = pose.pose.position.y - last.y
            if math.hypot(dx, dy) < self.min_d:
                return

        self.path.poses.append(pose)
        if len(self.path.poses) > self.max_poses:
            self.path.poses = self.path.poses[-self.max_poses:]

    def _publish(self):
        now = self.get_clock().now().to_msg()

        # Path republish — only after at least one sample landed.
        if self.path.poses:
            self.path.header.stamp = now
            self.pub.publish(self.path)

        # Front-pose odometry — single fixed-size message of the latest TF sample.
        if self.latest_pose is not None:
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = self.parent
            odom.child_frame_id = self.child
            odom.pose.pose = self.latest_pose.pose
            # twist left zero — we don't compute velocity here; consumers
            # that need it should subscribe to /odometry/filtered instead.
            self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
