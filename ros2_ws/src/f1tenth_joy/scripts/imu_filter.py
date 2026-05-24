#!/usr/bin/env python3
"""
IMU filter — the single canonical IMU-processing node for this car.

    /imu/data  (raw from BNO055)  ──►  imu_filter  ──►  /imu_filter
                                                    ──►  TF base_link → imu_link

Anything that wants IMU data (EKF, debug tools, future controllers) should
subscribe to **/imu_filter**, never to `/imu/data` directly — the filter
applies the corrections that make the readings usable.

Pipeline:

1. **Initial 5 s calibration** (robot must be still on a level surface):
   - Measures gyro & accel bias and variance per axis.
   - Measures mount tilt (roll, pitch) from BNO055's fused quaternion —
     LOGGED for reference, NOT baked into a static TF (this car has
     suspension, mount angle is dynamic).
2. **Bias subtraction** on every subsequent message.
3. **Horizontal projection**: rotates the bias-corrected gyro vector by the
   BNO055's live orientation quaternion so angular_velocity.z on /imu_filter
   is the true yaw rate in the gravity-aligned world frame, regardless of
   the chassis's current pitch/roll from suspension or terrain.
4. **First-order IIR low-pass on gyro Z**: gently smooths the projected
   horizontal yaw rate. α = 0.3 → cutoff ≈ 7 Hz at 100 Hz sample rate.
5. **Variance into covariance fields**: the measured variances are baked
   into the message covariance, so robot_localization weights readings
   correctly instead of treating the BNO055 driver's default zeros as
   "perfectly accurate".
6. **Mount-position TF**: publishes a *static* TF base_link → imu_link
   with the IMU mount xyz parsed from /robot_description (URDF imu joint)
   and rpy = (0, 0, 0) — rotation is handled dynamically by the horizontal
   projection above, not baked into TF.

Subscribes:
    /imu/data           sensor_msgs/Imu (raw from BNO055, sensor_data QoS)
    /robot_description  std_msgs/String (latched, from robot_state_publisher)

Publishes:
    /imu_filter         sensor_msgs/Imu (sensor_data QoS, ready for EKF)
    /tf_static          base_link → imu_link

Params:
    calibration_seconds  (double, 5.0)   idle window length at startup
    gyro_z_lpf_alpha     (double, 0.3)   IIR LPF coefficient on horizontal
                                          yaw rate. 1.0 = no smoothing;
                                          lower = smoother but more lag.
                                          0.3 ≈ 7 Hz cutoff at 100 Hz.
    imu_x, imu_y, imu_z  (doubles, 0)    Fallback mount xyz if URDF unavailable.
"""

import math
import statistics
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Deque, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


URDF_IMU_LINK_NAMES = ("imu", "imu_link")

# Defaults used WHILE the calibration window is running.
DEFAULT_GYRO_VAR  = 1.0e-3
DEFAULT_ACCEL_VAR = 1.0e-1
DEFAULT_ORIENT_VAR = 1.0e-1


class ImuFilter(Node):
    def __init__(self):
        super().__init__("imu_filter")

        self.declare_parameter("calibration_seconds", 5.0)
        # ---- two-stage filter on each horizontal-projected gyro axis ----
        # Stage 1 — MEDIAN: rejects single-sample spikes (impulse noise).
        # Linear-phase, doesn't smear an outlier across neighbours.
        # 1 = off, 3 = smallest useful, 5 = stronger spike rejection.
        self.declare_parameter("gyro_median_window", 3)
        # Stage 2 — MOVING AVERAGE: smooths the remaining low-amplitude noise.
        # Linear phase: constant (N-1)/2 sample group delay.
        # At 100 Hz IMU rate:
        #   N=1   → no smoothing (0 ms lag)
        #   N=3   → ~10 ms group delay
        #   N=5   → ~20 ms
        #   N=10  → ~45 ms
        self.declare_parameter("gyro_ma_window", 5)
        # MA mode:
        #   False (sliding) — for each new sample, output mean of last N
        #                      samples (window slides by 1). Continuous output.
        #   True  (block)   — accumulate N samples, output their mean, RESET
        #                      buffer, then start over. Output is a staircase
        #                      that holds last block's mean between updates.
        #                      Each block is independent — no carry-over.
        self.declare_parameter("gyro_ma_block_mode", True)
        # In NDOF mode the BNO055 already does gravity-removal with the
        # magnetometer-aided orientation, so the residual LIA bias when
        # stationary should be essentially 0. If anything tiny moves during
        # the 5 s cal window, the captured "bias" is just noise and
        # subtracting it actively INJECTS bias into the filtered output.
        # Default: subtract gyro bias (real and constant), don't touch accel.
        self.declare_parameter("subtract_gyro_bias", True)
        self.declare_parameter("subtract_accel_bias", False)
        self.declare_parameter("imu_x", 0.0)
        self.declare_parameter("imu_y", 0.0)
        self.declare_parameter("imu_z", 0.0)
        # ---- Zero-Velocity Update (ZUPT) ----
        # When the robot is verified stationary (both wheels still per
        # /vesc/twist AND raw gyro quiet), slowly refresh gyro_bias via EMA
        # toward the current raw gyro reading. Lets the bias track BNO055
        # warm-up / temperature drift instead of being frozen at the value
        # captured during the initial 5 s calibration.
        # alpha=0.01 at 100 Hz → ~1 s time constant: a half-minute of stillness
        # essentially replaces the bias estimate; 5 s of stillness gets ~40%.
        self.declare_parameter("zupt_enabled", True)
        self.declare_parameter("zupt_gyro_thresh", 0.05)    # rad/s ≈ 2.9°/s
        self.declare_parameter("zupt_vx_thresh", 0.05)      # m/s
        self.declare_parameter("zupt_settle_samples", 50)   # 0.5 s at 100 Hz
        self.declare_parameter("zupt_alpha", 0.01)

        self.calib_seconds = float(self.get_parameter("calibration_seconds").value)
        self.med_window = max(1, int(self.get_parameter("gyro_median_window").value))
        self.ma_window = max(1, int(self.get_parameter("gyro_ma_window").value))
        self.ma_block_mode = bool(self.get_parameter("gyro_ma_block_mode").value)
        self.sub_gyro_bias = bool(self.get_parameter("subtract_gyro_bias").value)
        self.sub_accel_bias = bool(self.get_parameter("subtract_accel_bias").value)
        self.zupt_enabled = bool(self.get_parameter("zupt_enabled").value)
        self.zupt_gyro_thresh = float(self.get_parameter("zupt_gyro_thresh").value)
        self.zupt_vx_thresh = float(self.get_parameter("zupt_vx_thresh").value)
        self.zupt_settle = int(self.get_parameter("zupt_settle_samples").value)
        self.zupt_alpha = float(self.get_parameter("zupt_alpha").value)

        # ZUPT state
        self._wheel_vx = 0.0
        self._wheel_seen = False
        self._still_count = 0
        self._zupt_updates = 0

        # Block-mode state: accumulators + last-block outputs (held between blocks)
        self._block_n = 0
        self._block_sum = {k: 0.0 for k in ("gx", "gy", "gz", "ax", "ay", "az")}
        self._block_out = {k: 0.0 for k in ("gx", "gy", "gz", "ax", "ay", "az")}
        self.imu_x = float(self.get_parameter("imu_x").value)
        self.imu_y = float(self.get_parameter("imu_y").value)
        self.imu_z = float(self.get_parameter("imu_z").value)
        self._urdf_xyz_loaded = (self.imu_x != 0.0 or self.imu_y != 0.0
                                  or self.imu_z != 0.0)

        # ---- state ----
        self.start_time: Optional[float] = None
        self.calibrated = False

        self.gx: List[float] = []; self.gy: List[float] = []; self.gz: List[float] = []
        self.ax: List[float] = []; self.ay: List[float] = []; self.az: List[float] = []
        self.yaw: List[float] = []
        self.quats: List[Tuple[float, float, float, float]] = []

        self.gyro_var  = (DEFAULT_GYRO_VAR,)  * 3
        self.accel_var = (DEFAULT_ACCEL_VAR,) * 3
        self.orient_var = (DEFAULT_ORIENT_VAR,) * 3
        self.gyro_bias  = (0.0, 0.0, 0.0)
        self.accel_bias = (0.0, 0.0, 0.0)

        # Median ringbuffers (stage 1 — spike rejection), gyro + accel
        self._med_gx: Deque[float] = deque(maxlen=self.med_window)
        self._med_gy: Deque[float] = deque(maxlen=self.med_window)
        self._med_gz: Deque[float] = deque(maxlen=self.med_window)
        self._med_ax: Deque[float] = deque(maxlen=self.med_window)
        self._med_ay: Deque[float] = deque(maxlen=self.med_window)
        self._med_az: Deque[float] = deque(maxlen=self.med_window)
        # Moving-average ringbuffers (stage 2 — smoothing), gyro + accel
        self._ma_gx: Deque[float] = deque(maxlen=self.ma_window)
        self._ma_gy: Deque[float] = deque(maxlen=self.ma_window)
        self._ma_gz: Deque[float] = deque(maxlen=self.ma_window)
        self._ma_ax: Deque[float] = deque(maxlen=self.ma_window)
        self._ma_ay: Deque[float] = deque(maxlen=self.ma_window)
        self._ma_az: Deque[float] = deque(maxlen=self.ma_window)

        # ---- I/O ----
        self.create_subscription(Imu, "/imu/data", self._on_imu,
                                  qos_profile_sensor_data)
        self.pub = self.create_publisher(Imu, "/imu_filter",
                                          qos_profile_sensor_data)
        self.create_subscription(TwistWithCovarianceStamped, "/vesc/twist",
                                  self._on_twist, 10)
        self.tf_static = StaticTransformBroadcaster(self)

        latched = QoSProfile(depth=1,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description",
                                  self._on_urdf, latched)

        # publish initial placeholder TF so the chain is closed from t=0
        self._broadcast_mount_tf()

        # heartbeat for GUI log visibility
        self.msg_count = 0
        self.create_timer(2.0, self._heartbeat)

        mode = "BLOCK" if self.ma_block_mode else "SLIDING"
        self.get_logger().info(
            f"imu_filter up — calibrating from first IMU sample. "
            f"Keep the robot STILL on a level surface for "
            f"{self.calib_seconds:.1f} s. Output: /imu_filter "
            f"(median N={self.med_window} → MA[{mode}] N={self.ma_window}; "
            f"subtract_gyro_bias={self.sub_gyro_bias}, "
            f"subtract_accel_bias={self.sub_accel_bias}; "
            f"ZUPT enabled={self.zupt_enabled}, "
            f"gyro<{math.degrees(self.zupt_gyro_thresh):.1f}°/s, "
            f"vx<{self.zupt_vx_thresh}m/s, "
            f"settle={self.zupt_settle} samp, alpha={self.zupt_alpha})")

    # ---- timers / callbacks ----

    def _heartbeat(self):
        if self.start_time is None:
            self.get_logger().info("heartbeat: waiting for first /imu/data…")
            return
        if self.calibrated:
            bias_dps = math.degrees(self.gyro_bias[2])
            zupt = ("ZUPT-ACTIVE" if self._still_count >= self.zupt_settle
                    else f"moving (still={self._still_count})")
            self.get_logger().info(
                f"heartbeat: calibrated, publishing /imu_filter "
                f"({self.msg_count} msgs); gyro_z bias={bias_dps:+.3f}°/s "
                f"[zupt updates={self._zupt_updates}, {zupt}]")
        else:
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"heartbeat: CALIBRATING — {elapsed:.1f}/{self.calib_seconds:.1f} s, "
                f"{len(self.gx)} samples")
        self.msg_count = 0

    def _on_imu(self, msg: Imu):
        self.msg_count += 1
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info(
                f"First /imu/data received. STAY STILL for "
                f"{self.calib_seconds:.1f} s…")

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
                self._finish_calibration()
                self.calibrated = True

        self._publish(msg)

    def _on_twist(self, msg: TwistWithCovarianceStamped):
        self._wheel_vx = msg.twist.twist.linear.x
        self._wheel_seen = True

    def _zupt_step(self, raw_gx: float, raw_gy: float, raw_gz: float):
        """Update gyro_bias when the robot is verified stationary.

        Requires: ZUPT enabled, calibration done, /vesc/twist seen at least
        once (otherwise we'd run ZUPT during all the boot quiet time when
        we just haven't received wheel velocity yet), wheel vx near zero,
        and raw gyro magnitude below threshold. After zupt_settle samples
        of sustained stillness, slowly EMA the bias toward the current
        raw gyro reading.
        """
        if not (self.zupt_enabled and self.calibrated and self._wheel_seen):
            return
        gyro_norm = math.sqrt(raw_gx*raw_gx + raw_gy*raw_gy + raw_gz*raw_gz)
        if abs(self._wheel_vx) > self.zupt_vx_thresh or gyro_norm > self.zupt_gyro_thresh:
            self._still_count = 0
            return
        self._still_count += 1
        if self._still_count < self.zupt_settle:
            return
        a = self.zupt_alpha
        self.gyro_bias = (
            (1.0 - a) * self.gyro_bias[0] + a * raw_gx,
            (1.0 - a) * self.gyro_bias[1] + a * raw_gy,
            (1.0 - a) * self.gyro_bias[2] + a * raw_gz,
        )
        self._zupt_updates += 1

    def _on_urdf(self, msg: String):
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
            if child.attrib.get("link", "") not in URDF_IMU_LINK_NAMES:
                continue
            origin = joint.find("origin")
            xyz = (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            try:
                x, y, z = (float(v) for v in xyz.split())
            except ValueError:
                self.get_logger().warn(f"imu joint xyz unparseable: {xyz!r}")
                return
            self.imu_x, self.imu_y, self.imu_z = x, y, z
            self._urdf_xyz_loaded = True
            self.get_logger().info(
                f"IMU mount xyz from URDF: ({x:+.5f}, {y:+.5f}, {z:+.5f}) m")
            self._broadcast_mount_tf()
            return
        self.get_logger().warn(
            f"/robot_description had no joint with child in {URDF_IMU_LINK_NAMES}")

    # ---- calibration ----

    def _finish_calibration(self):
        n = len(self.gx)
        if n < 10:
            self.get_logger().warn(
                f"Only {n} samples — keeping default bias/variance.")
            return

        def var(xs, floor=1.0e-9):
            return max(statistics.variance(xs), floor)

        self.gyro_var   = (var(self.gx), var(self.gy), var(self.gz))
        self.accel_var  = (var(self.ax), var(self.ay), var(self.az))
        yaw_v = var(self.yaw, floor=1.0e-4)
        self.orient_var = (yaw_v, yaw_v, yaw_v)

        self.gyro_bias  = tuple(statistics.fmean(s) for s in
                                  (self.gx, self.gy, self.gz))
        self.accel_bias = tuple(statistics.fmean(s) for s in
                                  (self.ax, self.ay, self.az))

        stable = self.quats[max(0, n // 2):]
        mq = self._avg_quat(stable)
        roll, pitch, _ = self._quat_to_euler(mq)

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
            f"  ── mount tilt at rest (informational; NOT in TF) ──\n"
            f"  roll  = {roll:+.5f} rad ({math.degrees(roll):+.3f}°)\n"
            f"  pitch = {pitch:+.5f} rad ({math.degrees(pitch):+.3f}°)\n"
            f"  gyro is projected to horizontal via live BNO055 orientation;\n"
            f"  base_link → imu_link static TF stays at URDF nominal (rpy=0).")

        for buf in (self.gx, self.gy, self.gz,
                    self.ax, self.ay, self.az,
                    self.yaw, self.quats):
            buf.clear()

    # ---- publishing ----

    def _publish(self, msg: Imu):
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation

        # ZUPT runs BEFORE bias-subtraction so it sees the raw gyro and
        # slowly EMAs the bias estimate toward whatever the IMU is
        # reading while the robot is verified stationary.
        self._zupt_step(msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z)

        # 1. bias-corrected gyro in IMU local frame
        if self.sub_gyro_bias:
            gx = msg.angular_velocity.x - self.gyro_bias[0]
            gy = msg.angular_velocity.y - self.gyro_bias[1]
            gz = msg.angular_velocity.z - self.gyro_bias[2]
        else:
            gx, gy, gz = (msg.angular_velocity.x, msg.angular_velocity.y,
                          msg.angular_velocity.z)

        # 2. project to gravity-aligned world frame using BNO055 quaternion
        wx, wy, wz = self._rotate_vec_by_quat(
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w,
            gx, gy, gz)

        # 3a. MEDIAN filter — rejects single-sample spikes (outliers in
        # an odd-sized window get sorted to the ends and ignored).
        if self.med_window > 1:
            self._med_gx.append(wx); self._med_gy.append(wy); self._med_gz.append(wz)
            wx = sorted(self._med_gx)[len(self._med_gx) // 2]
            wy = sorted(self._med_gy)[len(self._med_gy) // 2]
            wz = sorted(self._med_gz)[len(self._med_gz) // 2]

        # 3b. MOVING AVERAGE — sliding or block depending on `ma_block_mode`.
        if self.ma_window <= 1:
            wx_out, wy_out, wz_out = wx, wy, wz
        elif not self.ma_block_mode:
            # SLIDING: deque-based rolling window
            self._ma_gx.append(wx); self._ma_gy.append(wy); self._ma_gz.append(wz)
            wx_out = sum(self._ma_gx) / len(self._ma_gx)
            wy_out = sum(self._ma_gy) / len(self._ma_gy)
            wz_out = sum(self._ma_gz) / len(self._ma_gz)
        else:
            # BLOCK: accumulate N samples → output mean → reset → repeat.
            # Between blocks, output is held at the last computed mean.
            wx_out, wy_out, wz_out = self._block_step("gx", wx, "gy", wy, "gz", wz)

        out.angular_velocity.x = wx_out
        out.angular_velocity.y = wy_out
        out.angular_velocity.z = wz_out

        # ---- accel filtering: same median → MA chain as gyro ----
        # Bias subtraction default OFF — BNO055 in NDOF mode already removes
        # gravity using its fused (mag-aided) orientation, so the LIA bias
        # when stationary is ≈ 0 and the calibration-measured bias is
        # mostly noise. Subtracting it would INJECT a bias into the output.
        if self.sub_accel_bias:
            ax = msg.linear_acceleration.x - self.accel_bias[0]
            ay = msg.linear_acceleration.y - self.accel_bias[1]
            az = msg.linear_acceleration.z - self.accel_bias[2]
        else:
            ax, ay, az = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                          msg.linear_acceleration.z)
        if self.med_window > 1:
            self._med_ax.append(ax); self._med_ay.append(ay); self._med_az.append(az)
            ax = sorted(self._med_ax)[len(self._med_ax) // 2]
            ay = sorted(self._med_ay)[len(self._med_ay) // 2]
            az = sorted(self._med_az)[len(self._med_az) // 2]
        if self.ma_window > 1:
            if not self.ma_block_mode:
                self._ma_ax.append(ax); self._ma_ay.append(ay); self._ma_az.append(az)
                ax = sum(self._ma_ax) / len(self._ma_ax)
                ay = sum(self._ma_ay) / len(self._ma_ay)
                az = sum(self._ma_az) / len(self._ma_az)
            else:
                # Block-mode accel uses the SAME 8-sample block counter as gyro
                # (incremented inside _block_step above for the gyro axes).
                # Here we just accumulate the accel components and read their
                # block output at the same boundary.
                ax, ay, az = self._block_step_accel(ax, ay, az)
        out.linear_acceleration.x = ax
        out.linear_acceleration.y = ay
        out.linear_acceleration.z = az

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

    def _block_step(self, kx, vx, ky, vy, kz, vz):
        """Accumulate one sample into the block. On block boundary, finalise
        the mean for all six channels (gyro and accel) and reset. Between
        boundaries, return the previously-computed block means (staircase
        output, last-block value held).

        Increments `self._block_n` exactly once per IMU message — called from
        the gyro path. Accel adds to the same accumulators via
        `_block_step_accel`, which must be called AFTER this method in the
        same message handler."""
        self._block_sum[kx] += vx
        self._block_sum[ky] += vy
        self._block_sum[kz] += vz
        self._block_n += 1
        if self._block_n >= self.ma_window:
            for k in self._block_sum:
                self._block_out[k] = self._block_sum[k] / self._block_n
                self._block_sum[k] = 0.0
            self._block_n = 0
        return (self._block_out[kx], self._block_out[ky], self._block_out[kz])

    def _block_step_accel(self, ax, ay, az):
        """Add this message's accel to the current block (shared with gyro).
        Block-finalise happens in `_block_step`; here we just contribute and
        read the latest block output."""
        # Only accumulate if a new block is still building (block_n > 0 means
        # _block_step hasn't yet flushed this cycle). If _block_n was just
        # reset to 0 by _block_step (meaning gyro flushed), the accel sample
        # has already been included via the flush — but to keep things
        # symmetric we'll add it to the *next* block instead.
        self._block_sum["ax"] += ax
        self._block_sum["ay"] += ay
        self._block_sum["az"] += az
        return (self._block_out["ax"], self._block_out["ay"], self._block_out["az"])

    def _broadcast_mount_tf(self):
        """Publish base_link → imu_link static TF with the URDF mount xyz
        and identity rotation. Rotation handled dynamically by horizontal
        projection in _publish — NOT baked here."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "imu_link"
        t.transform.translation.x = self.imu_x
        t.transform.translation.y = self.imu_y
        t.transform.translation.z = self.imu_z
        t.transform.rotation.w = 1.0    # identity
        self.tf_static.sendTransform(t)

    # ---- math helpers ----

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quat_to_euler(q) -> Tuple[float, float, float]:
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
    def _rotate_vec_by_quat(qx: float, qy: float, qz: float, qw: float,
                              vx: float, vy: float, vz: float
                              ) -> Tuple[float, float, float]:
        """v_world = q · v_imu · q*, optimised single-vector form."""
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    @staticmethod
    def _avg_quat(qs: List[Tuple[float, float, float, float]]
                   ) -> Tuple[float, float, float, float]:
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
    node = ImuFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
