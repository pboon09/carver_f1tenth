#!/usr/bin/env python3
"""
Bridge joystick / autopilot commands into JointState messages so the URDF
model in RViz animates as we drive:

  - throttle (RPM)          -> FL_wheel / FR_wheel / RearWheel spin
  - steering angle (rad)    -> AckermannLeftTurn / AckermannRightTurn rotate

The actual robot pose on the map is still driven by SLAM via
map -> odom -> base_link. This node only publishes /joint_states; it does
NOT move the robot through space.

Subscribes:
    /vesc/cmd        std_msgs/Int32     motor RPM command from joystick.py
    /steering_angle  std_msgs/Float32   steering angle in radians

Publishes:
    /joint_states    sensor_msgs/JointState   @ publish_rate

Params:
    gear_ratio   (double, 4.0)   motor revs per wheel rev — purely visual,
                                 tune so the wheels look right at full RPM
    publish_rate (double, 30.0)  Hz
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32


JOINT_STEER_LEFT  = "AckermannLeftTurn"
JOINT_STEER_RIGHT = "AckermannRightTurn"
JOINT_FL_WHEEL    = "FL_wheel"
JOINT_FR_WHEEL    = "FR_wheel"
JOINT_REAR_WHEEL  = "RearWheel"


class CmdToJointState(Node):
    def __init__(self):
        super().__init__("cmd_to_joint_state")

        self.declare_parameter("gear_ratio", 4.0)
        self.declare_parameter("publish_rate", 30.0)
        self.gear_ratio = float(self.get_parameter("gear_ratio").value)
        rate = float(self.get_parameter("publish_rate").value)

        # Latest command values (default to 0 — model is static until cmds arrive)
        self.current_rpm = 0
        self.current_steer = 0.0
        # Accumulated wheel rotation (radians) — integrated from RPM
        self.wheel_angle = 0.0
        self.last_time = time.time()

        self.create_subscription(Int32,   "/vesc/cmd",       self._on_rpm,   10)
        self.create_subscription(Float32, "/steering_angle", self._on_steer, 10)
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"cmd_to_joint_state up: gear_ratio={self.gear_ratio}, "
            f"publish_rate={rate} Hz")

    def _on_rpm(self, msg: Int32):
        self.current_rpm = msg.data

    def _on_steer(self, msg: Float32):
        self.current_steer = float(msg.data)

    def _publish(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # motor RPM -> wheel rad/s, integrate to running angle.
        # Negated because the URDF's wheel joint Z axis points opposite the
        # direction of travel — without this, pushing throttle forward made
        # the visual wheels spin backward.
        wheel_omega = -(self.current_rpm / self.gear_ratio) * 2.0 * math.pi / 60.0
        self.wheel_angle += wheel_omega * dt
        # keep the accumulator bounded so it doesn't grow forever
        self.wheel_angle = math.fmod(self.wheel_angle, 2.0 * math.pi)

        # Negate steering for the URDF Ackermann joints — the URDF's joint
        # axis convention turns out to be opposite to the real servo, so
        # without this negation the model in RViz steers the wrong way
        # while the physical car steers correctly (or vice-versa).
        urdf_steer = -self.current_steer

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            JOINT_STEER_LEFT, JOINT_STEER_RIGHT,
            JOINT_FL_WHEEL, JOINT_FR_WHEEL, JOINT_REAR_WHEEL,
        ]
        msg.position = [
            urdf_steer, urdf_steer,
            self.wheel_angle, self.wheel_angle, self.wheel_angle,
        ]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdToJointState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
