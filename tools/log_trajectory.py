#!/usr/bin/env python3
"""
Log the composed map→base_link pose at high rate to a CSV — for analyzing
whether the robot is "hopping" at scan-match rate (10 Hz) vs smoothly
dead-reckoning at EKF rate (100 Hz).

Usage:
    python3 tools/log_trajectory.py [output.csv]

Default output: ~/slam_logs/traj_<YYYYMMDD_HHMMSS>.csv

Records 5 columns: t_sec, x, y, yaw_rad, lookup_lag_ms
  - t_sec: wall-clock time of the lookup
  - x, y, yaw_rad: composed pose in map frame
  - lookup_lag_ms: how stale the TF was at lookup time (high = TF chain not
    keeping up)

Stop with Ctrl-C. CSV is closed cleanly on exit.
"""

import csv
import datetime
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


SAMPLE_HZ = 100


class TrajectoryLogger(Node):
    def __init__(self, out_path: str):
        super().__init__("trajectory_logger")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.out_path = out_path
        self.fh = open(out_path, "w", newline="")
        self.csv = csv.writer(self.fh)
        self.csv.writerow(["t_sec", "x", "y", "yaw_rad", "lookup_lag_ms"])
        self.t0 = time.time()
        self.n = 0
        self.create_timer(1.0 / SAMPLE_HZ, self._sample)
        self.create_timer(2.0, self._heartbeat)
        self.get_logger().info(
            f"trajectory_logger up — writing to {out_path} at {SAMPLE_HZ} Hz. "
            f"Ctrl-C to stop.")

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _sample(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", Time(),
                timeout=Duration(seconds=0.02))
        except (LookupException, ExtrapolationException):
            return
        now = time.time()
        # Header stamp → seconds
        stamp_s = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        lag_ms = max(0.0, (now - stamp_s) * 1000.0)
        x = t.transform.translation.x
        y = t.transform.translation.y
        yaw = self._yaw_from_quat(t.transform.rotation)
        self.csv.writerow([f"{now - self.t0:.4f}",
                           f"{x:.5f}", f"{y:.5f}",
                           f"{yaw:.5f}", f"{lag_ms:.2f}"])
        self.n += 1

    def _heartbeat(self):
        elapsed = time.time() - self.t0
        rate = self.n / max(0.001, elapsed)
        self.get_logger().info(
            f"logged {self.n} samples in {elapsed:.1f} s ({rate:.1f} Hz effective)")

    def close(self):
        self.fh.close()
        self.get_logger().info(f"saved {self.n} samples → {self.out_path}")


def main():
    if len(sys.argv) >= 2:
        out_path = sys.argv[1]
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.expanduser("~/slam_logs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_{stamp}.csv")

    rclpy.init()
    node = TrajectoryLogger(out_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
