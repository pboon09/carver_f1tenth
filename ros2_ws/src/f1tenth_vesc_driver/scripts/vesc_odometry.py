#!/usr/bin/env python3
"""
Wheel + Ackermann odometry from VESC telemetry & joystick steering.

Reads motor RPM from /vesc/state (published by vesc_node.py) and steering
angle from /steering_angle (published by joystick.py / autopilot). Treats the
car as a bicycle: linear velocity from rear-wheel RPM, angular velocity from
v/L · tan(steer). Integrates and broadcasts:

  - /odom                 nav_msgs/Odometry
  - TF odom -> base_link  (replaces the static placeholder in mapping.launch.py)

When VESC isn't running (no /vesc/state messages), RPM stays at 0 and the
robot just sits at the origin — same behaviour the static placeholder had,
so SLAM still has the TF chain it needs and won't fall over.

The z component of the published TF is constant (`base_z`), so the robot
stays at the correct height above the ground (REP-105 base_footprint at
z=0). Only x, y, yaw change with motion.

Params:
    wheelbase     (m,  0.30)    front-axle ↔ rear-axle distance
    wheel_radius  (m,  0.0594)  matches the URDF inertials
    gear_ratio    (—,  4.0)     motor:wheel reduction; tune if speed feels off
    base_z        (m,  0.1189)  lift base_link above the ground plane
    publish_rate  (Hz, 50.0)
    odom_frame    (str, "odom")
    base_frame    (str, "base_link")
"""

import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster


class VescOdometry(Node):
    def __init__(self):
        super().__init__("vesc_odometry")

        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("wheel_radius", 0.0594)
        self.declare_parameter("gear_ratio", 4.0)
        self.declare_parameter("base_z", 0.1189)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self.L = float(self.get_parameter("wheelbase").value)
        self.r = float(self.get_parameter("wheel_radius").value)
        self.gear_ratio = float(self.get_parameter("gear_ratio").value)
        self.base_z = float(self.get_parameter("base_z").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Pose state (in odom frame)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = time.time()

        # Latest sensor / cmd values
        self.motor_rpm = 0.0
        self.steer_rad = 0.0

        self.create_subscription(JointState, "/vesc/state", self._on_vesc_state, 10)
        self.create_subscription(Float32, "/steering_angle", self._on_steer, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 50)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(1.0 / rate, self._update)

        self.get_logger().info(
            f"vesc_odometry up: L={self.L} m, r={self.r} m, "
            f"gear={self.gear_ratio}, base_z={self.base_z} m, rate={rate} Hz")

    def _on_vesc_state(self, msg: JointState):
        # vesc_node publishes a JointState with name[0]='rpm', position[0]=motor RPM
        if msg.name and msg.position and msg.name[0] == "rpm":
            self.motor_rpm = float(msg.position[0])

    def _on_steer(self, msg: Float32):
        self.steer_rad = float(msg.data)

    def _update(self):
        now_t = time.time()
        dt = now_t - self.last_time
        self.last_time = now_t
        if dt <= 0.0 or dt > 0.5:  # skip first tick & big stalls
            return

        # motor RPM → wheel rad/s → vehicle linear velocity (m/s)
        wheel_omega = (self.motor_rpm / self.gear_ratio) * 2.0 * math.pi / 60.0
        v = wheel_omega * self.r

        # bicycle model angular velocity
        omega = (v / self.L) * math.tan(self.steer_rad) if self.L > 1e-6 else 0.0

        # integrate
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += omega * dt
        # normalize
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)
        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.base_z
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = self.base_z
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = VescOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
