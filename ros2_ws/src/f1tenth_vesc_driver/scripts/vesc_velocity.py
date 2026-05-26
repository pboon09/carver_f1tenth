#!/usr/bin/env python3
"""
Ackermann (bicycle) wheel-odom bridge for the EKF, with slip diagnostic.

Combines two measurements into the canonical wheel-odom twist:
  - vx   from motor RPM (via /vesc/state JointState position[0])
  - vyaw from bicycle-model kinematics  ω = vx · tan(δ) / L
         where δ is the commanded steering angle (/steering_angle Float32)
         and L is the wheelbase from URDF (~0.30 m).

Both channels get LOOSE covariance — this is the f1tenth high-slip regime, so
the EKF should only weakly trust either:
  - vx_covariance ≈ 2.0 m²/s²   (1σ ≈ 1.4 m/s)
  - vyaw_covariance ≈ 1.0 rad²/s² (1σ ≈ 57°/s)
That lets IMU gyro Z dominate yaw rate when scrubbing, and lets SLAM scan-
match veto position when the rear wheels spin. Between scans the wheel+model
keeps the dead-reckoning honest enough to be a good motion prior.

A side-output /vesc/slip (Float32) publishes
   slip ≈ wheel_velocity − /odometry/filtered.twist.linear.x
so you can watch slip live (rqt_plot, foxglove) and tune covariance.

Note: /steering_angle is the COMMAND to the STM32 servo — no feedback topic
exists for actual angle. At low servo lag this is a fine proxy.
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry


class VescVelocity(Node):
    def __init__(self):
        super().__init__("vesc_velocity")

        self.declare_parameter("wheel_radius", 0.0594)
        self.declare_parameter("gear_ratio", 29.5)
        self.declare_parameter("wheelbase", 0.30)   # URDF: 0.192757 - (-0.10719)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("base_frame", "basefootprint")
        # Loose covariance — high-slip regime, EKF should only weakly trust
        # either channel. Lower if you regrip / drive smooth.
        self.declare_parameter("vx_covariance", 2.0)
        self.declare_parameter("vyaw_covariance", 1.0)

        self.r = float(self.get_parameter("wheel_radius").value)
        self.gear_ratio = float(self.get_parameter("gear_ratio").value)
        self.L = float(self.get_parameter("wheelbase").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.vx_cov = float(self.get_parameter("vx_covariance").value)
        self.vyaw_cov = float(self.get_parameter("vyaw_covariance").value)

        self._lock = threading.Lock()
        self.motor_rpm = 0.0
        self.steering_rad = 0.0
        self.ekf_vx = 0.0
        self.last_wheel_v = 0.0

        self.create_subscription(JointState, "/vesc/state", self._on_state, 10)
        # /steering_angle is a control command — sample-and-hold, always want
        # the freshest value. BEST_EFFORT + small depth avoids retransmit lag
        # and buffer staleness. Accepts both reliable (joystick/mixer) and
        # best-effort publishers.
        self.create_subscription(
            Float32, "/steering_angle", self._on_steering,
            qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_ekf, 10)
        self.twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, "/vesc/twist", 10)
        self.slip_pub = self.create_publisher(Float32, "/vesc/slip", 10)

        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"vesc_velocity up: r={self.r} m, gear={self.gear_ratio}, "
            f"L={self.L} m, rate={rate} Hz, "
            f"vx_cov={self.vx_cov}, vyaw_cov={self.vyaw_cov}; "
            f"twist out on /vesc/twist (vx+vyaw), slip diag on /vesc/slip")

    def _on_state(self, msg: JointState):
        if msg.name and msg.position and msg.name[0] == "rpm":
            with self._lock:
                self.motor_rpm = float(msg.position[0])

    def _on_steering(self, msg: Float32):
        with self._lock:
            self.steering_rad = float(msg.data)

    def _on_ekf(self, msg: Odometry):
        with self._lock:
            self.ekf_vx = float(msg.twist.twist.linear.x)

    def _publish(self):
        with self._lock:
            rpm = self.motor_rpm
            delta = self.steering_rad
            ekf_vx = self.ekf_vx

        wheel_omega = (rpm / self.gear_ratio) * 2.0 * math.pi / 60.0
        v = wheel_omega * self.r
        self.last_wheel_v = v

        # Bicycle model: ω = v · tan(δ) / L. Holds at low slip — high-slip
        # divergence is exactly what the loose vyaw covariance is there for.
        vyaw = v * math.tan(delta) / self.L

        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = vyaw

        # 6×6 row-major covariance: [vx, vy, vz, vroll, vpitch, vyaw]
        cov = [0.0] * 36
        cov[0]  = self.vx_cov     # vx — measured (motor RPM)
        cov[7]  = 1.0e9           # vy
        cov[14] = 1.0e9           # vz
        cov[21] = 1.0e9           # vroll
        cov[28] = 1.0e9           # vpitch
        cov[35] = self.vyaw_cov   # vyaw — bicycle-model estimate
        msg.twist.covariance = cov
        self.twist_pub.publish(msg)

        # Slip diagnostic: positive when wheels go faster than EKF thinks the
        # robot is actually moving (i.e. wheels are slipping forward).
        slip = Float32()
        slip.data = v - ekf_vx
        self.slip_pub.publish(slip)


def main(args=None):
    rclpy.init(args=args)
    node = VescVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
