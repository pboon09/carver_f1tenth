#!/usr/bin/env python3
"""
IMU noise auto-calibrator + mount-tilt measurer (BNO055).

What it does at every startup:
  1. Waits for the first /imu/data message.
  2. For `calibration_seconds` (default 5 s) the robot MUST be IDLE on a
     level surface. Samples accumulate.
  3. After the window:
     - bias  (mean) per channel is subtracted from every republished message
     - variance per channel goes into the message covariance fields the EKF
       uses to weight the IMU
     - **mount tilt** (roll, pitch) measured from the BNO055's fused quaternion
       is broadcast as a *static* TF base_link → imu_link. The EKF and SLAM
       see the IMU axes correctly aligned to base_link, even if the sensor
       was physically glued on a few degrees off.
  4. A sanity check compares the quaternion-derived tilt against the accel
     leakage on x/y — if they disagree by > 0.5 m/s² the calibrator warns
     that the BNO055's sensor fusion probably hasn't converged yet (move the
     IMU around for 30 s before re-launching to help it settle).

So the calibrator now owns base_link → imu_link entirely — no separate
static_transform_publisher needed in the launch file.

Subscribes:
    /imu/data              sensor_msgs/Imu       (raw from bno055)

Publishes:
    /imu/data_calibrated   sensor_msgs/Imu       (bias-removed + covariance)
    /tf_static             base_link → imu_link  (with measured mount tilt)

Params:
    calibration_seconds  (double, 5.0)   idle window length
    imu_x, imu_y, imu_z  (doubles)       IMU mount POSITION on the chassis
                                         (defaults match the URDF)
"""

import math
import statistics
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


# URDF link names that count as "the IMU". The SolidWorks exporter named ours
# `imu`; the BNO055 driver publishes /imu/data with frame_id = "imu_link".
# We accept either, then publish the static TF as base_link → imu_link.
URDF_IMU_LINK_NAMES = ("imu", "imu_link")


# Conservative defaults used WHILE calibrating, before we have measured
# values. Roughly equal to "trust the IMU a little but not a lot".
DEFAULT_GYRO_VAR  = 1.0e-3   # (rad/s)²
DEFAULT_ACCEL_VAR = 1.0e-1   # (m/s²)²
DEFAULT_ORIENT_VAR = 1.0e-1  # rad²


class ImuCalibrator(Node):
    def __init__(self):
        super().__init__("imu_calibrator")

        self.declare_parameter("calibration_seconds", 5.0)
        # Fallback mount position used until /robot_description arrives and we
        # parse the URDF imu joint. Override via launch params if you want a
        # hard-coded value (e.g., URDF not running).
        self.declare_parameter("imu_x", 0.0)
        self.declare_parameter("imu_y", 0.0)
        self.declare_parameter("imu_z", 0.0)

        self.calib_seconds = float(self.get_parameter("calibration_seconds").value)
        self.imu_x = float(self.get_parameter("imu_x").value)
        self.imu_y = float(self.get_parameter("imu_y").value)
        self.imu_z = float(self.get_parameter("imu_z").value)
        self._urdf_xyz_loaded = (self.imu_x != 0.0 or self.imu_y != 0.0
                                  or self.imu_z != 0.0)
        # Track the latest measured tilt so we can re-broadcast when the URDF
        # arrives later (the imu mount position becomes known after we've
        # already published the first identity TF).
        self._last_roll = 0.0
        self._last_pitch = 0.0

        self.start_time: Optional[float] = None
        self.calibrated = False
        # Online bias refresh: ringbuffer of the last `idle_window` seconds
        # of gyro samples. We detect "idle" by the MEAN magnitude in the
        # buffer being under threshold — much more robust to noise spikes
        # than "every sample below threshold" (which never fires when
        # gyro noise σ approaches the threshold).
        # When idle, refresh gyro bias from the buffer's mean to catch
        # BNO055 thermal drift that dominates long-session yaw drift.
        self.idle_window_sec = 2.0
        self.idle_gyro_threshold = 0.1    # rad/s (~5.7°/s) — mean threshold
        self.idle_buffer: List[Tuple[float, float, float, float]] = []
        self.last_refresh_t = 0.0
        self.refresh_min_interval_sec = 3.0   # don't refresh more than every 3 s

        # sample buffers (cleared after calibration)
        self.gx: List[float] = []
        self.gy: List[float] = []
        self.gz: List[float] = []
        self.ax: List[float] = []
        self.ay: List[float] = []
        self.az: List[float] = []
        self.yaw: List[float] = []
        # full quaternion samples for tilt measurement
        self.quats: List[Tuple[float, float, float, float]] = []

        # active variances + biases (start with defaults; replaced on completion)
        self.gyro_var  = (DEFAULT_GYRO_VAR,)  * 3
        self.accel_var = (DEFAULT_ACCEL_VAR,) * 3
        self.orient_var = (DEFAULT_ORIENT_VAR,) * 3
        self.gyro_bias  = (0.0, 0.0, 0.0)
        self.accel_bias = (0.0, 0.0, 0.0)

        # I/O — sensor_data QoS to match the BNO055 driver (BEST_EFFORT)
        self.create_subscription(Imu, "/imu/data", self._on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(Imu, "/imu/data_calibrated", qos_profile_sensor_data)
        self.tf_static = StaticTransformBroadcaster(self)

        # /robot_description is published latched (transient_local) by
        # robot_state_publisher. Use matching durability so we get the latest
        # value as soon as we subscribe — that lets us pull the IMU joint's
        # xyz straight out of the URDF and avoid duplicating it in the launch.
        latched = QoSProfile(depth=1,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description",
                                  self._on_urdf, latched)

        # Publish an initial identity (roll=0, pitch=0) TF so the chain is
        # closed from t=0 — the EKF can start consuming IMU msgs before
        # calibration completes. After calibration we overwrite this TF
        # with the measured tilt.
        self._broadcast_tilt(0.0, 0.0)

        # heartbeat — visible in the GUI log so you can tell whether the
        # calibrator is actually alive and what state it's in.
        self.msg_count = 0
        self.create_timer(2.0, self._heartbeat)

        self.get_logger().info(
            f"imu_calibrator up — will calibrate from the first IMU sample. "
            f"Keep the robot STILL ON A LEVEL SURFACE for "
            f"{self.calib_seconds:.1f} s after sensors come up.")

    def _heartbeat(self):
        if self.start_time is None:
            self.get_logger().info("heartbeat: waiting for first /imu/data message…")
            return
        if self.calibrated:
            self.get_logger().info(
                f"heartbeat: calibrated. accel bias = "
                f"({self.accel_bias[0]:+.3e}, {self.accel_bias[1]:+.3e}, "
                f"{self.accel_bias[2]:+.3e}) m/s²; "
                f"publishing /imu/data_calibrated, "
                f"msgs since last heartbeat: {self.msg_count}")
        else:
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"heartbeat: CALIBRATING — {elapsed:.1f}/{self.calib_seconds:.1f} s, "
                f"{len(self.gx)} samples collected")
        self.msg_count = 0

    def _on_imu(self, msg: Imu):
        self.msg_count += 1
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info(
                f"First /imu/data received. Robot must be IDLE on a LEVEL "
                f"surface for {self.calib_seconds:.1f} s — sampling now…")

        elapsed = time.time() - self.start_time

        if not self.calibrated:
            self.gx.append(msg.angular_velocity.x)
            self.gy.append(msg.angular_velocity.y)
            self.gz.append(msg.angular_velocity.z)
            self.ax.append(msg.linear_acceleration.x)
            self.ay.append(msg.linear_acceleration.y)
            self.az.append(msg.linear_acceleration.z)
            q = msg.orientation
            self.quats.append((q.x, q.y, q.z, q.w))
            self.yaw.append(self._quat_to_yaw(msg.orientation))

            if elapsed >= self.calib_seconds:
                self._finish()
                self.calibrated = True
        # NOTE: online bias refresh tried with 2 s mean window — actually
        # made yaw drift WORSE (150°/min vs original 17°/min) because the
        # IMU noise floor (gyro 1σ ≈ 0.031 rad/s) is too high to estimate
        # bias accurately in a few seconds. Disabled. Strategy now: trust
        # the initial 5 s calibration, accept some gyro bias drift, let
        # slam_toolbox correct it at scan rate.

        # Always republish — during the window with conservative defaults,
        # after with measured variances. Keeps the EKF fed.
        self._publish(msg)

    def _maybe_refresh_bias(self, msg: Imu, now: float) -> None:
        """Online gyro-bias refresh. When the robot has been idle (all gyro
        readings under the threshold) for `idle_window_sec`, recompute the
        bias from those samples. This kills BNO055 thermal drift that would
        otherwise turn into runaway yaw.

        Bias on accel is NOT refreshed here — accel readings while
        stationary include gravity leakage that varies with attitude, so
        re-mean would mis-calibrate."""
        gxr = msg.angular_velocity.x
        gyr = msg.angular_velocity.y
        gzr = msg.angular_velocity.z
        # Magnitude of the bias-corrected gyro vector
        debiased_mag = math.sqrt(
            (gxr - self.gyro_bias[0]) ** 2
            + (gyr - self.gyro_bias[1]) ** 2
            + (gzr - self.gyro_bias[2]) ** 2
        )
        # Keep RAW (not debiased) samples in the buffer so the new mean is
        # the absolute bias, not "bias of the residual".
        self.idle_buffer.append((gxr, gyr, gzr, debiased_mag))
        # Trim to the idle window — assume ~100 Hz IMU rate
        max_len = int(self.idle_window_sec * 100)
        if len(self.idle_buffer) > max_len:
            self.idle_buffer.pop(0)
        if len(self.idle_buffer) < max_len:
            return
        # Mean magnitude over the window must be below threshold — noise
        # spikes in individual samples are OK; we want the time-average to
        # indicate stationarity.
        mean_mag = sum(s[3] for s in self.idle_buffer) / len(self.idle_buffer)
        if mean_mag > self.idle_gyro_threshold:
            return
        # Don't refresh too often
        if now - self.last_refresh_t < self.refresh_min_interval_sec:
            return
        # Robot has been still for `idle_window_sec` — refresh bias.
        new_bias = (
            sum(s[0] for s in self.idle_buffer) / len(self.idle_buffer),
            sum(s[1] for s in self.idle_buffer) / len(self.idle_buffer),
            sum(s[2] for s in self.idle_buffer) / len(self.idle_buffer),
        )
        delta = (new_bias[0] - self.gyro_bias[0],
                 new_bias[1] - self.gyro_bias[1],
                 new_bias[2] - self.gyro_bias[2])
        self.gyro_bias = new_bias
        self.last_refresh_t = now
        self.get_logger().info(
            f"online bias refresh: new gyro bias = "
            f"({new_bias[0]:+.4e}, {new_bias[1]:+.4e}, {new_bias[2]:+.4e}) rad/s "
            f"(Δ from previous: {delta[0]:+.4e}, {delta[1]:+.4e}, {delta[2]:+.4e})")

    def _finish(self):
        n = len(self.gx)
        if n < 10:
            self.get_logger().warn(
                f"Only {n} samples collected — keeping default variances. "
                f"Calibration window too short for this IMU rate.")
            return

        def var(xs, floor=1.0e-9):
            v = statistics.variance(xs)
            return max(v, floor)

        self.gyro_var   = (var(self.gx), var(self.gy), var(self.gz))
        self.accel_var  = (var(self.ax), var(self.ay), var(self.az))
        # Bigger floor on yaw — when the robot is perfectly still, raw yaw
        # variance is ≈ 0, which the EKF would treat as "yaw is ground truth"
        # and refuse to correct it from scan-match. 1e-4 rad² ≈ ±0.6° lets
        # the scan-match update gently re-anchor heading over time.
        yaw_v = var(self.yaw, floor=1.0e-4)
        self.orient_var = (yaw_v, yaw_v, yaw_v)

        # bias = mean reading while stationary
        self.gyro_bias  = (statistics.fmean(self.gx),
                           statistics.fmean(self.gy),
                           statistics.fmean(self.gz))
        self.accel_bias = (statistics.fmean(self.ax),
                           statistics.fmean(self.ay),
                           statistics.fmean(self.az))

        # --- mount tilt measurement ---
        # Average the LAST half of the quaternion samples — the BNO055's
        # sensor fusion converges over the first second or two, so the
        # later half is more reliable.
        stable = self.quats[max(0, n // 2):]
        mq = self._avg_quat(stable)
        roll, pitch, _yaw_meas = self._quat_to_euler(mq)

        # --- sanity check: does the quaternion tilt match the accel leakage? ---
        # If the IMU is mounted with tilt θ on a level surface, the BNO055's
        # LIA register *should* read ~0 (gravity removed), but if the BNO055's
        # orientation estimate is off by Δ, you'd see ~9.81·sin(Δ) m/s² leakage.
        # We compare the EXPECTED leakage (zero, since quaternion says tilt is
        # already accounted for) against MEASURED bias and warn if it's big.
        leak_mag = math.hypot(self.accel_bias[0], self.accel_bias[1])
        if leak_mag > 0.5:
            implied_extra_tilt = math.degrees(math.asin(min(leak_mag / 9.81, 1.0)))
            self.get_logger().warn(
                f"BNO055 orientation may not have converged: linear_acceleration "
                f"bias magnitude {leak_mag:.2f} m/s² implies an EXTRA ~{implied_extra_tilt:.1f}° "
                f"of tilt that the BNO055 isn't reporting in its quaternion. "
                f"Wave the IMU around for ~30 s before re-launching so its "
                f"fusion algorithm has data to settle, then re-calibrate."
            )

        # Tilt measurement is now LOGGED ONLY — NOT baked into the static TF.
        # Reason: this is an F1 RC chassis with suspension; roll/pitch change
        # dynamically during cornering, bumps, and after pick-up. A one-shot
        # static rotation in the TF would be wrong as soon as the suspension
        # moves. Instead, we project gyro readings into the gravity-aligned
        # horizontal frame at runtime using BNO055's live orientation
        # quaternion (see _publish). That way angular_velocity.z is the TRUE
        # horizontal yaw rate regardless of momentary tilt.

        self.get_logger().info(
            f"IMU calibration complete (n={n}):\n"
            f"  ── biases (subtracted from every msg) ──\n"
            f"  gyro   (x,y,z) = ({self.gyro_bias[0]:+.4e}, "
            f"{self.gyro_bias[1]:+.4e}, {self.gyro_bias[2]:+.4e}) rad/s\n"
            f"  accel  (x,y,z) = ({self.accel_bias[0]:+.4e}, "
            f"{self.accel_bias[1]:+.4e}, {self.accel_bias[2]:+.4e}) m/s²\n"
            f"  ── variances (baked into msg covariance) ──\n"
            f"  gyro   (x,y,z) = ({self.gyro_var[0]:.3e}, "
            f"{self.gyro_var[1]:.3e}, {self.gyro_var[2]:.3e})\n"
            f"  accel  (x,y,z) = ({self.accel_var[0]:.3e}, "
            f"{self.accel_var[1]:.3e}, {self.accel_var[2]:.3e})\n"
            f"  yaw            = {yaw_v:.3e}\n"
            f"  ── mount tilt (measured at rest — INFORMATIONAL, NOT baked) ──\n"
            f"  roll  = {roll:+.5f} rad ({math.degrees(roll):+.3f}°)\n"
            f"  pitch = {pitch:+.5f} rad ({math.degrees(pitch):+.3f}°)\n"
            f"  gyro is projected to horizontal via BNO055 orientation live;\n"
            f"  base_link → imu_link static TF stays at URDF nominal (rpy=0).")

        # drop sample buffers
        self.gx.clear(); self.gy.clear(); self.gz.clear()
        self.ax.clear(); self.ay.clear(); self.az.clear()
        self.yaw.clear(); self.quats.clear()

    def _publish(self, msg: Imu):
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation

        # Bias-removed gyro vector in IMU local frame
        gx = msg.angular_velocity.x - self.gyro_bias[0]
        gy = msg.angular_velocity.y - self.gyro_bias[1]
        gz = msg.angular_velocity.z - self.gyro_bias[2]

        # ROTATE gyro into the gravity-aligned WORLD frame using BNO055's
        # live orientation quaternion. The Z component of the rotated vector
        # is the TRUE horizontal yaw rate — independent of the IMU's current
        # tilt (which changes constantly with suspension on an RC car).
        # The EKF uses only the Z component (vyaw), so this kills cross-axis
        # leakage from a rolled/pitched chassis.
        wx, wy, wz = self._rotate_vec_by_quat(
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w,
            gx, gy, gz)
        out.angular_velocity.x = wx
        out.angular_velocity.y = wy
        out.angular_velocity.z = wz

        out.linear_acceleration.x = msg.linear_acceleration.x - self.accel_bias[0]
        out.linear_acceleration.y = msg.linear_acceleration.y - self.accel_bias[1]
        out.linear_acceleration.z = msg.linear_acceleration.z - self.accel_bias[2]

        gv, av, ov = self.gyro_var, self.accel_var, self.orient_var
        out.orientation_covariance = [
            ov[0], 0.0, 0.0,
            0.0, ov[1], 0.0,
            0.0, 0.0, ov[2],
        ]
        out.angular_velocity_covariance = [
            gv[0], 0.0, 0.0,
            0.0, gv[1], 0.0,
            0.0, 0.0, gv[2],
        ]
        out.linear_acceleration_covariance = [
            av[0], 0.0, 0.0,
            0.0, av[1], 0.0,
            0.0, 0.0, av[2],
        ]
        self.pub.publish(out)

    def _broadcast_tilt(self, roll: float, pitch: float) -> None:
        """Publish base_link → imu_link with the measured mount tilt.
        Yaw is forced to 0 — BNO055 yaw drifts and absolute heading is
        controlled by slam_toolbox via map → odom."""
        self._last_roll, self._last_pitch = roll, pitch
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "imu_link"
        t.transform.translation.x = self.imu_x
        t.transform.translation.y = self.imu_y
        t.transform.translation.z = self.imu_z
        qx, qy, qz, qw = self._euler_to_quat(roll, pitch, 0.0)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_static.sendTransform(t)

    def _on_urdf(self, msg: String) -> None:
        """Parse the URDF latched on /robot_description, pull out the IMU
        joint's xyz, and re-broadcast the TF with the URDF-derived position.
        Runs at most once successfully — after that the URDF is the source
        of truth for mount POSITION, and the calibrator only owns rotation."""
        if self._urdf_xyz_loaded:
            return
        try:
            root = ET.fromstring(msg.data)
        except ET.ParseError as e:
            self.get_logger().warn(f"Couldn't parse /robot_description: {e}")
            return

        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is None:
                continue
            link_name = child.attrib.get("link", "")
            if link_name not in URDF_IMU_LINK_NAMES:
                continue
            origin = joint.find("origin")
            xyz = (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            try:
                x, y, z = (float(v) for v in xyz.split())
            except ValueError:
                self.get_logger().warn(
                    f"IMU joint origin has unparseable xyz {xyz!r}")
                return
            self.imu_x, self.imu_y, self.imu_z = x, y, z
            self._urdf_xyz_loaded = True
            self.get_logger().info(
                f"IMU mount position from URDF joint child='{link_name}': "
                f"({x:+.5f}, {y:+.5f}, {z:+.5f}) m — re-broadcasting "
                f"base_link → imu_link with the latest measured tilt.")
            self._broadcast_tilt(self._last_roll, self._last_pitch)
            return

        self.get_logger().warn(
            "URDF arrived but no joint with child link "
            f"in {URDF_IMU_LINK_NAMES} found; keeping fallback xyz.")

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quat_to_euler(q) -> Tuple[float, float, float]:
        # q = (x, y, z, w) tuple — ZYX → (roll, pitch, yaw)
        x, y, z, w = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = (math.copysign(math.pi / 2, sinp)
                 if abs(sinp) >= 1 else math.asin(sinp))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return (sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy)

    @staticmethod
    def _rotate_vec_by_quat(qx: float, qy: float, qz: float, qw: float,
                              vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """Rotate a 3-vector by a unit quaternion: v_world = q · v_imu · q*.
        Optimised single-vector form (no quaternion-multiplication overhead).
        See Wikipedia: Quaternion → vector rotation."""
        # t = 2 · (q.xyz × v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        # v_rot = v + qw · t + q.xyz × t
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    @staticmethod
    def _avg_quat(qs: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Component-mean + normalise. Good for small angular spreads
        (< 90°) which is always the case for a stationary IMU."""
        n = len(qs)
        mx = sum(q[0] for q in qs) / n
        my = sum(q[1] for q in qs) / n
        mz = sum(q[2] for q in qs) / n
        mw = sum(q[3] for q in qs) / n
        norm = math.sqrt(mx * mx + my * my + mz * mz + mw * mw)
        if norm == 0:
            return (0.0, 0.0, 0.0, 1.0)
        return (mx / norm, my / norm, mz / norm, mw / norm)


def main(args=None):
    rclpy.init(args=args)
    node = ImuCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
