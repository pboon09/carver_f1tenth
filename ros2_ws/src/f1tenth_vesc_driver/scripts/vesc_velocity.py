#!/usr/bin/env python3
"""
VESC wheel velocity bridge for the EKF, with online slip diagnostic.

Reads motor RPM from /vesc/state (JointState position[0]) and publishes the
implied longitudinal velocity as geometry_msgs/TwistWithCovarianceStamped on
/vesc/twist. The EKF subscribes to that as `twist0`.

The covariance on vx is set INTENTIONALLY HIGH so the EKF doesn't trust the
wheel velocity too much — when the wheels slip catastrophically the wheel
reading is wrong and the filter should defer to the IMU yaw + SLAM scan-match
correction. With vx_covariance ≈ 1.0 m²/s² (1σ = 1 m/s) the filter blends in
the wheel velocity for smooth 100 Hz forward propagation while letting SLAM
veto it during slip events.

A side-output `/vesc/slip` (std_msgs/Float32) publishes the residual
   slip ≈ wheel_velocity − /odometry/filtered.twist.linear.x
so you can watch slip live (rqt_plot, foxglove) — useful for tuning the
covariance and for sanity-checking whether the EKF is correctly down-weighting
wheel-vel during slip.
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry


class VescVelocity(Node):
    def __init__(self):
        super().__init__("vesc_velocity")

        self.declare_parameter("wheel_radius", 0.0594)
        self.declare_parameter("gear_ratio", 4.0)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("base_frame", "basefootprint")
        # High vx covariance so the EKF only weakly trusts wheel velocity.
        # 1.0 m²/s² = 1σ of 1 m/s — comfortable margin against catastrophic
        # slip events. Lower (e.g. 0.1) if your wheels grip well; higher
        # if slip is severe.
        self.declare_parameter("vx_covariance", 1.0)

        self.r = float(self.get_parameter("wheel_radius").value)
        self.gear_ratio = float(self.get_parameter("gear_ratio").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.vx_cov = float(self.get_parameter("vx_covariance").value)

        self._lock = threading.Lock()
        self.motor_rpm = 0.0
        self.ekf_vx = 0.0
        self.last_wheel_v = 0.0

        self.create_subscription(JointState, "/vesc/state", self._on_state, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_ekf, 10)
        self.twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, "/vesc/twist", 10)
        self.slip_pub = self.create_publisher(Float32, "/vesc/slip", 10)

        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"vesc_velocity up: r={self.r} m, gear={self.gear_ratio}, "
            f"rate={rate} Hz, vx_covariance={self.vx_cov}; "
            f"twist out on /vesc/twist, slip diag on /vesc/slip")

    def _on_state(self, msg: JointState):
        if msg.name and msg.position and msg.name[0] == "rpm":
            with self._lock:
                self.motor_rpm = float(msg.position[0])

    def _on_ekf(self, msg: Odometry):
        with self._lock:
            self.ekf_vx = float(msg.twist.twist.linear.x)

    def _publish(self):
        with self._lock:
            rpm = self.motor_rpm
            ekf_vx = self.ekf_vx

        wheel_omega = (rpm / self.gear_ratio) * 2.0 * math.pi / 60.0
        v = wheel_omega * self.r
        self.last_wheel_v = v

        # Twist for the EKF — only vx is real, everything else has huge
        # covariance so the EKF treats those channels as missing.
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.twist.linear.x = v

        # 6×6 row-major covariance: [vx, vy, vz, vroll, vpitch, vyaw]
        cov = [0.0] * 36
        cov[0]  = self.vx_cov   # vx — the one we measure
        cov[7]  = 1.0e9         # vy
        cov[14] = 1.0e9         # vz
        cov[21] = 1.0e9         # vroll
        cov[28] = 1.0e9         # vpitch
        cov[35] = 1.0e9         # vyaw — we don't measure
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
