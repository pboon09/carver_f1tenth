#!/usr/bin/env python3
"""
F1TENTH Launch GUI — Tkinter front-end for launching and building the
workspaces we use on the car.

Each launch row supervises one ros2-launch / ros2-run subprocess. Each
workspace row drives a colcon build (plain `colcon build`, no flags) and
a Clean (rm build/ install/ log/). All stdout+stderr funnel into the same
log pane, prefixed and color-coded by source. Stop / Cancel send SIGINT to
the whole process group; if a process hasn't exited 3 s later, SIGKILL is
sent.

Sourcing: each launch's bash -c already `source`s the workspaces it needs,
so the next Start automatically picks up a fresh build — there is no
separate "source" step.

Hierarchy awareness: some launches transitively spawn others (base ⊇
sensor + uros; sensor ⊇ rplidar + imu; slam ⊇ sensor because
mapping.launch.py includes sensor.launch.py). When a parent starts, every
transitively-included child is marked "running via <parent>" and its Start
button disables; when the parent stops, the children come out of shadow.
Trying to start a parent whose child is already running independently is
blocked with a warning (they'd race on the same USB device).

Run:
    python3 /home/carver/carver_f1tenth/tools/launcher.py
"""

import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import tkinter as tk

# Optional rclpy — used only by the IMU monitor tab. If ROS isn't sourced in
# the GUI's environment the import fails and the IMU tab shows a hint instead
# of crashing the whole launcher.
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
from tkinter import ttk, scrolledtext, messagebox


ROS_SETUP = "source /opt/ros/jazzy/setup.bash"
ROS2_WS   = "source /home/carver/carver_f1tenth/ros2_ws/install/setup.bash"
SENSOR_WS = "source /home/carver/carver_f1tenth/sensor_ws/install/setup.bash"
SLAM_WS   = "source /home/carver/carver_f1tenth/slam_ws/install/setup.bash"
UROS_WS   = "source /home/carver/uros_ws/install/setup.bash"


def chain(*parts: str) -> str:
    return " && ".join(parts)


# (name, color, bash command)
LAUNCHES = [
    ("slam base",      "#1f6feb",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_bringup base_slam.launch.py")),
    ("loc base",       "#0969da",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_bringup base_loc.launch.py")),
    ("sensor launch",  "#0a3069",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS,
              "ros2 launch f1tenth_bringup sensor.launch.py")),
    ("slam launch",    "#1a7f37",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_slam mapping.launch.py")),
    ("loc launch",     "#218bff",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_slam localization.launch.py")),
    ("rplidar launch", "#bf8700",
        chain(ROS_SETUP, SENSOR_WS,
              "ros2 launch rplidar_ros rplidar_c1_launch.py")),
    ("imu launch",     "#8250df",
        chain(ROS_SETUP, SENSOR_WS,
              "ros2 run bno055_usb_stick bno055_usb_stick_node_script.py")),
    ("uros launch",    "#a72b29",
        chain(ROS_SETUP, UROS_WS,
              "ros2 run micro_ros_agent micro_ros_agent "
              "serial --dev /dev/STM32 -b 115200")),
    ("urdf launch",    "#d63384",
        chain(ROS_SETUP, ROS2_WS,
              "ros2 launch f1tenth_urdf display.launch.py")),
    ("joystick launch","#3e8e41",
        chain(ROS_SETUP, ROS2_WS,
              "ros2 launch f1tenth_bringup joystick.launch.py")),
]

# (display name, /dev/ symlink set up by your udev rules)
SENSORS = [
    ("RPLidar",  "/dev/rplidar"),
    ("BNO055",   "/dev/BNO_Stick"),
    ("STM32",    "/dev/STM32"),
    ("VESC",     "/dev/VESC"),
]

# (name, color, absolute path). Build-all walks these in order.
WORKSPACES = [
    ("sensor_ws", "#bf8700", "/home/carver/carver_f1tenth/sensor_ws"),
    ("ros2_ws",   "#1f6feb", "/home/carver/carver_f1tenth/ros2_ws"),
    ("slam_ws",   "#1a7f37", "/home/carver/carver_f1tenth/slam_ws"),
    ("uros_ws",   "#a72b29", "/home/carver/uros_ws"),
]

# Transitive set of launches each parent already brings up. Drives the
# "running via <parent>" shadow markers and the conflict warning when you
# try to start a parent whose child is already running on its own.
CONTAINS: dict[str, list[str]] = {
    "slam base":     ["slam launch", "loc launch", "sensor launch",
                      "rplidar launch", "imu launch", "uros launch", "loc base"],
    "loc base":      ["loc launch", "slam launch", "sensor launch",
                      "rplidar launch", "imu launch", "uros launch", "slam base"],
    "sensor launch": ["rplidar launch", "imu launch"],
    "slam launch":   ["sensor launch", "rplidar launch", "imu launch"],
    "loc launch":    ["sensor launch", "rplidar launch", "imu launch"],
}

# `pkill -9 -f <pattern>` patterns to fire after SIGKILLing the process group,
# so any orphan that escaped the group (e.g. detached child, separate ros2 run
# someone started by hand) still goes away. Patterns match the *full command
# line* — keep them specific enough to not nuke unrelated processes.
KILL_PATTERNS: dict[str, list[str]] = {
    "slam base":       ["base_slam\\.launch\\.py", "base_loc\\.launch\\.py",
                        "mapping\\.launch\\.py", "localization\\.launch\\.py",
                        "rplidar_node", "bno055_usb_stick",
                        "slam_toolbox", "static_transform_publisher",
                        "robot_state_publisher", "joint_state_publisher",
                        "cmd_to_joint_state\\.py", "imu_calibrator\\.py",
                        "imu_filter\\.py",
                        "vesc_velocity\\.py", "ekf_node",
                        "vesc_node\\.py", "micro_ros_agent",
                        "rviz2 .*f1tenth_slam.*slam\\.rviz"],
    "loc base":        ["base_slam\\.launch\\.py", "base_loc\\.launch\\.py",
                        "mapping\\.launch\\.py", "localization\\.launch\\.py",
                        "rplidar_node", "bno055_usb_stick",
                        "slam_toolbox", "static_transform_publisher",
                        "robot_state_publisher", "joint_state_publisher",
                        "cmd_to_joint_state\\.py", "imu_calibrator\\.py",
                        "imu_filter\\.py",
                        "vesc_velocity\\.py", "ekf_node",
                        "vesc_node\\.py", "micro_ros_agent",
                        "rviz2 .*f1tenth_slam.*slam\\.rviz"],
    "sensor launch":   ["sensor\\.launch\\.py", "rplidar_node", "bno055_usb_stick"],
    "slam launch":     ["mapping\\.launch\\.py", "rplidar_node", "bno055_usb_stick",
                        "slam_toolbox", "static_transform_publisher",
                        "cmd_to_joint_state\\.py", "imu_calibrator\\.py",
                        "imu_filter\\.py",
                        "robot_state_publisher", "ekf_node",
                        "rviz2 .*f1tenth_slam.*slam\\.rviz"],
    "loc launch":      ["localization\\.launch\\.py", "rplidar_node", "bno055_usb_stick",
                        "slam_toolbox", "static_transform_publisher",
                        "cmd_to_joint_state\\.py", "imu_calibrator\\.py",
                        "imu_filter\\.py",
                        "robot_state_publisher", "ekf_node",
                        "rviz2 .*f1tenth_slam.*slam\\.rviz"],
    "rplidar launch":  ["rplidar_c1_launch\\.py", "rplidar_node"],
    "imu launch":      ["bno055_usb_stick"],
    "uros launch":     ["micro_ros_agent"],
    "urdf launch":     ["f1tenth_urdf.*display\\.launch\\.py",
                        "robot_state_publisher", "joint_state_publisher_gui",
                        "rviz2 .*f1tenth_urdf.*display\\.rviz"],
    "joystick launch": ["joystick\\.launch\\.py", "joy_node", "f1tenth_joy"],
}

# Used by the "Kill orphans" button — broad sweep of every node we ever spawn.
# `rviz2` is intentionally excluded so we don't nuke an rviz you're using for
# something unrelated.
ORPHAN_PATTERNS = [
    "ros2 launch f1tenth", "ros2 launch rplidar",
    "ros2 run bno055_usb_stick", "ros2 run micro_ros_agent",
    "rplidar_node", "bno055_usb_stick", "vesc_node\\.py", "micro_ros_agent",
    "async_slam_toolbox_node", "localization_slam_toolbox_node", "slam_toolbox",
    "static_transform_publisher",
    "robot_state_publisher", "joint_state_publisher_gui",
    "joy_node", "f1tenth_joy",
]

# Targeted sweep for the joystick subsystem — used by the "Kill joy zombies"
# button. Symptom this fixes: two `joystick.launch.py` trees ended up running
# in parallel, each spawning its own joy_node + joystick.py, both publishing
# /steering_angle simultaneously → STM32 servo whips between them. Hit this
# button, then Start joystick launch fresh.
JOYSTICK_KILL_PATTERNS = [
    "ros2 launch f1tenth_bringup joystick",
    "joystick\\.launch\\.py",
    "f1tenth_joy/lib/f1tenth_joy/joystick\\.py",
    "joy/joy_node",
]


# ---------------------------------------------------------------------------
# Launch supervisor row
# ---------------------------------------------------------------------------

class LaunchRow:
    def __init__(self, parent, row_idx, name, color, cmd, app):
        self.app = app
        self.name = name
        self.color = color
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self.shadowed_by: set[str] = set()

        ttk.Label(parent, text=name, width=16, anchor="w").grid(
            row=row_idx, column=0, padx=4, pady=3, sticky="w")
        self.status_var = tk.StringVar(value="● stopped")
        self.status_lbl = tk.Label(parent, textvariable=self.status_var,
                                   fg="gray", width=40, anchor="w")
        self.status_lbl.grid(row=row_idx, column=1, padx=4, pady=3, sticky="w")
        self.start_btn = ttk.Button(parent, text="Start", width=7, command=self.start)
        self.start_btn.grid(row=row_idx, column=2, padx=2, pady=3)
        self.stop_btn = ttk.Button(parent, text="Stop", width=7,
                                   command=self.stop, state="disabled")
        self.stop_btn.grid(row=row_idx, column=3, padx=2, pady=3)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _refresh(self):
        if self.is_running():
            assert self.proc is not None
            self.status_var.set(f"● running (pid {self.proc.pid})")
            self.status_lbl.configure(fg="#1a7f37")
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        elif self.shadowed_by:
            via = ", ".join(sorted(self.shadowed_by))
            self.status_var.set(f"● running via {via}")
            self.status_lbl.configure(fg="#0a3069")
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")
        else:
            self.status_var.set("● stopped")
            self.status_lbl.configure(fg="gray")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def add_shadow(self, parent_name: str):
        self.shadowed_by.add(parent_name)
        self._refresh()

    def remove_shadow(self, parent_name: str):
        self.shadowed_by.discard(parent_name)
        self._refresh()

    def start(self):
        if self.is_running() or self.shadowed_by:
            return

        conflicts = [c for c in CONTAINS.get(self.name, [])
                     if self.app.row_by_name(c).is_running()]
        if conflicts:
            msg = (f"Can't start '{self.name}' — already running "
                   f"independently: {', '.join(conflicts)}. Stop them first.")
            self.app.log_system(msg)
            messagebox.showwarning("Already running", msg)
            return

        self.proc = subprocess.Popen(
            ["bash", "-c", self.cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            start_new_session=True,
        )
        self.app.log_system(f"[{self.name}] started (pid {self.proc.pid})")

        children = CONTAINS.get(self.name, [])
        if children:
            self.app.log_system(
                f"[{self.name}] also brings up: {', '.join(children)} — "
                f"Start disabled on those rows while '{self.name}' runs.")
            for c in children:
                self.app.row_by_name(c).add_shadow(self.name)

        self._refresh()
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, ""):
            self.app.log_queue.put((self.name, line.rstrip()))
        self.proc.stdout.close()
        rc = self.proc.wait()
        self.app.log_queue.put(("__launch_exit__", (self.name, rc)))

    def stop(self):
        """Hard stop: SIGKILL the process group, then pkill any orphan node
        commands by pattern. No SIGINT grace — fastest possible Stop."""
        if not self.is_running():
            return
        assert self.proc is not None
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.status_var.set("● killing…")
        self.status_lbl.configure(fg="#a72b29")
        self.stop_btn.configure(state="disabled")
        # Catch anything that escaped the process group.
        for pat in KILL_PATTERNS.get(self.name, []):
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.app.log_system(
            f"[{self.name}] SIGKILL + pkill -9 sent "
            f"(patterns: {', '.join(KILL_PATTERNS.get(self.name, ['<group only>']))})")

    def on_exit(self, rc):
        for c in CONTAINS.get(self.name, []):
            self.app.row_by_name(c).remove_shadow(self.name)
        self.proc = None
        if rc in (0, -2, 130, -9, 137):
            self.status_var.set("● stopped")
            self.status_lbl.configure(fg="gray")
        else:
            self.status_var.set(f"● exited ({rc})")
            self.status_lbl.configure(fg="#a72b29")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


# ---------------------------------------------------------------------------
# Sensor connection (USB/TTY) row — read-only, polled
# ---------------------------------------------------------------------------

class SensorCell:
    """Compact one-line status: `RPLidar ● ttyUSB0` packed inline."""

    def __init__(self, parent, name, path, app):
        self.app = app
        self.name = name
        self.path = path
        self.last_state: str | None = None

        self.lbl = tk.Label(parent, text=f"{name}: ●  …",
                            fg="gray", font=("TkDefaultFont", 9), padx=6)
        self.lbl.pack(side="left", padx=4)

    def refresh(self):
        state, text, color = self._probe()
        self.lbl.configure(text=text, fg=color)
        if self.last_state is not None and state != self.last_state:
            self.app.log_system(
                f"[{self.name}] {self.last_state} → {state}  ({self.path})")
        self.last_state = state

    def _probe(self) -> tuple[str, str, str]:
        if not os.path.lexists(self.path):
            return ("disconnected", f"{self.name}: ●  off", "#a72b29")
        try:
            target = (os.readlink(self.path) if os.path.islink(self.path)
                      else os.path.basename(self.path))
        except OSError:
            target = "?"
        if not os.path.exists(os.path.realpath(self.path)):
            return ("dangling", f"{self.name}: ●  dangling→{target}", "#bf8700")
        return ("connected", f"{self.name}: ●  {target}", "#1a7f37")


# ---------------------------------------------------------------------------
# IMU monitor tab — passive read-only view of /imu/data
# ---------------------------------------------------------------------------

class ImuMonitor:
    """Subscribes to /imu/data and shows roll/pitch/yaw + gyro + accel live.

    Designed for ONE specific use case: measuring the BNO055's mount tilt so
    you can bake it into the `base_link → imu_link` static TF. Park the robot
    on a level surface, wait for the values to settle, copy roll & pitch
    (in radians) into mapping.launch.py.

    Does not launch or enable anything. Just attempts a subscription on
    sensor_data QoS; if the IMU node isn't publishing, the tab simply shows
    "waiting…" forever.
    """

    POLL_HZ = 10                          # GUI refresh rate
    STALE_AFTER_S = 0.5                   # message considered stale beyond this

    def __init__(self, parent, root):
        self.root = root
        self._latest = None
        self._latest_ts = 0.0
        self._lock = threading.Lock()
        self.node = None
        self.exec = None

        # --- header / status ---
        ttk.Label(parent, text="IMU monitor — read-only",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w",
                                                            padx=8, pady=(6, 0))
        self.status_lbl = tk.Label(parent, text="initializing…",
                                    fg="gray", font=("monospace", 10),
                                    anchor="w")
        self.status_lbl.pack(anchor="w", padx=8, pady=2, fill="x")

        # --- value grid ---
        grid = ttk.LabelFrame(parent, text="Latest /imu/data", padding=8)
        grid.pack(fill="x", padx=8, pady=6)

        self.labels: dict[str, tk.Label] = {}
        rows = [
            ("Roll",   "roll",  "rad"),
            ("Pitch",  "pitch", "rad"),
            ("Yaw",    "yaw",   "rad"),
            ("Gyro X", "gx",    "rad/s"),
            ("Gyro Y", "gy",    "rad/s"),
            ("Gyro Z", "gz",    "rad/s"),
            ("Accel X", "ax",   "m/s²"),
            ("Accel Y", "ay",   "m/s²"),
            ("Accel Z", "az",   "m/s²"),
        ]
        for i, (label, key, unit) in enumerate(rows):
            ttk.Label(grid, text=label + ":", anchor="e", width=8).grid(
                row=i, column=0, padx=4, pady=1, sticky="e")
            value_lbl = tk.Label(grid, text="—", fg="#dcdcdc",
                                  bg="#1e1e1e", font=("monospace", 11),
                                  width=12, anchor="e", padx=6)
            value_lbl.grid(row=i, column=1, padx=4, pady=1, sticky="w")
            self.labels[key] = value_lbl
            ttk.Label(grid, text=unit, foreground="#888", width=8,
                      anchor="w").grid(row=i, column=2, padx=4, pady=1, sticky="w")
            # extra column showing degrees for orientation rows
            if key in ("roll", "pitch", "yaw"):
                deg_lbl = tk.Label(grid, text="—", fg="#888",
                                    bg="#1e1e1e", font=("monospace", 10),
                                    width=12, anchor="e", padx=6)
                deg_lbl.grid(row=i, column=3, padx=4, pady=1, sticky="w")
                self.labels[key + "_deg"] = deg_lbl
                ttk.Label(grid, text="deg", foreground="#888", width=6,
                          anchor="w").grid(row=i, column=4, padx=4, pady=1,
                                            sticky="w")

        # --- usage hint ---
        hint_text = (
            "To bake the mount tilt into the URDF / static TF:\n"
            "  1. Park the robot on a level surface (a spirit level helps).\n"
            "  2. Wait until Roll and Pitch stop drifting (post-IMU-calibration).\n"
            "  3. Open mapping.launch.py → tf_base_to_imu Node.\n"
            "  4. Replace --roll / --pitch with the Roll / Pitch values above\n"
            "     (in radians, NOT degrees).\n"
            "  5. Rebuild f1tenth_slam, restart slam launch."
        )
        ttk.Label(parent, text=hint_text, foreground="#888",
                  wraplength=520, justify="left",
                  font=("TkDefaultFont", 9)).pack(anchor="w", padx=8, pady=6)

        # --- wire up rclpy subscription (best-effort, never blocks the GUI) ---
        if not RCLPY_AVAILABLE:
            self.status_lbl.configure(
                text="rclpy not importable. Source ROS jazzy + sensor_ws "
                     "in this shell, then restart the GUI.",
                fg="#a72b29")
        else:
            try:
                if not rclpy.ok():
                    rclpy.init(args=None)
                self.node = rclpy.create_node("launcher_imu_monitor")
                self.node.create_subscription(
                    Imu, "/imu_filter", self._on_imu, qos_profile_sensor_data)
                self.exec = SingleThreadedExecutor()
                self.exec.add_node(self.node)
                threading.Thread(target=self._spin, daemon=True).start()
                self.status_lbl.configure(
                    text="subscribed to /imu/data — waiting for first message…",
                    fg="#bf8700")
            except Exception as e:
                self.status_lbl.configure(
                    text=f"rclpy init failed: {e!r}", fg="#a72b29")

        self.root.after(int(1000 / self.POLL_HZ), self._tick)

    def _spin(self):
        try:
            self.exec.spin()
        except Exception:
            pass

    def _on_imu(self, msg):
        with self._lock:
            self._latest = msg
            self._latest_ts = time.time()

    def _tick(self):
        with self._lock:
            msg = self._latest
            ts = self._latest_ts

        if msg is not None:
            roll, pitch, yaw = self._quat_to_euler(msg.orientation)
            self.labels["roll"].configure(text=f"{roll:+.5f}")
            self.labels["pitch"].configure(text=f"{pitch:+.5f}")
            self.labels["yaw"].configure(text=f"{yaw:+.5f}")
            self.labels["roll_deg"].configure(text=f"{math.degrees(roll):+.3f}")
            self.labels["pitch_deg"].configure(text=f"{math.degrees(pitch):+.3f}")
            self.labels["yaw_deg"].configure(text=f"{math.degrees(yaw):+.3f}")
            self.labels["gx"].configure(text=f"{msg.angular_velocity.x:+.5f}")
            self.labels["gy"].configure(text=f"{msg.angular_velocity.y:+.5f}")
            self.labels["gz"].configure(text=f"{msg.angular_velocity.z:+.5f}")
            self.labels["ax"].configure(text=f"{msg.linear_acceleration.x:+.5f}")
            self.labels["ay"].configure(text=f"{msg.linear_acceleration.y:+.5f}")
            self.labels["az"].configure(text=f"{msg.linear_acceleration.z:+.5f}")
            age = time.time() - ts
            if age < self.STALE_AFTER_S:
                self.status_lbl.configure(
                    text=f"receiving /imu/data (frame: {msg.header.frame_id}, "
                         f"latency: {age*1000:.0f} ms)",
                    fg="#1a7f37")
            else:
                self.status_lbl.configure(
                    text=f"STALE — last /imu/data was {age:.1f} s ago",
                    fg="#bf8700")
        self.root.after(int(1000 / self.POLL_HZ), self._tick)

    @staticmethod
    def _quat_to_euler(q):
        # ZYX convention
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = (math.copysign(math.pi / 2, sinp)
                  if abs(sinp) >= 1 else math.asin(sinp))
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def shutdown(self):
        if self.exec is not None:
            try:
                self.exec.shutdown()
            except Exception:
                pass
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Workspace build row
# ---------------------------------------------------------------------------

class BuildRow:
    def __init__(self, parent, row_idx, name, color, path, app):
        self.app = app
        self.name = name
        self.color = color
        self.path = path
        self.proc: subprocess.Popen | None = None

        ttk.Label(parent, text=name, width=12, anchor="w").grid(
            row=row_idx, column=0, padx=4, pady=3, sticky="w")
        self.status_var = tk.StringVar(value=self._initial_status())
        self.status_lbl = tk.Label(parent, textvariable=self.status_var,
                                   fg="gray", width=36, anchor="w")
        self.status_lbl.grid(row=row_idx, column=1, padx=4, pady=3, sticky="w")
        self.build_btn = ttk.Button(parent, text="Build", width=7, command=self.build)
        self.build_btn.grid(row=row_idx, column=2, padx=2, pady=3)
        self.cancel_btn = ttk.Button(parent, text="Cancel", width=7,
                                     command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=row_idx, column=3, padx=2, pady=3)
        self.clean_btn = ttk.Button(parent, text="Clean", width=7, command=self.clean)
        self.clean_btn.grid(row=row_idx, column=4, padx=2, pady=3)

    def _initial_status(self) -> str:
        return "● built" if os.path.isdir(os.path.join(self.path, "install")) else "● not built"

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def build(self):
        if self.is_running():
            return
        if not os.path.isdir(os.path.join(self.path, "src")):
            msg = f"{self.name}: no src/ at {self.path} — skipping."
            self.app.log_system(msg)
            return

        cmd = chain(ROS_SETUP, f"cd {self.path}", "colcon build")
        self.proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            start_new_session=True,
        )
        self.status_var.set(f"● building… (pid {self.proc.pid})")
        self.status_lbl.configure(fg="#bf8700")
        self.build_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.clean_btn.configure(state="disabled")
        self.app.log_system(
            f"[{self.name}] colcon build started in {self.path}")
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, ""):
            self.app.log_queue.put((self.name, line.rstrip()))
        self.proc.stdout.close()
        rc = self.proc.wait()
        self.app.log_queue.put(("__build_exit__", (self.name, rc)))

    def cancel(self):
        """Hard cancel: SIGKILL the colcon process group immediately."""
        if not self.is_running():
            return
        assert self.proc is not None
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.status_var.set("● killing…")
        self.cancel_btn.configure(state="disabled")

    def clean(self):
        if self.is_running():
            return
        targets = [os.path.join(self.path, d) for d in ("build", "install", "log")]
        present = [t for t in targets if os.path.isdir(t)]
        if not present:
            self.app.log_system(f"[{self.name}] nothing to clean.")
            return
        if not messagebox.askyesno(
                "Clean workspace",
                f"Delete build/, install/, log/ in:\n{self.path} ?"):
            return
        for t in present:
            shutil.rmtree(t, ignore_errors=True)
        self.app.log_system(
            f"[{self.name}] removed: {', '.join(os.path.basename(p) for p in present)}")
        self.status_var.set("● cleaned")
        self.status_lbl.configure(fg="gray")

    def on_exit(self, rc):
        self.proc = None
        if rc == 0:
            self.status_var.set("● built ✓")
            self.status_lbl.configure(fg="#1a7f37")
            self.app.log_system(f"[{self.name}] build OK — relaunches will pick it up.")
        elif rc in (-2, 130):
            self.status_var.set("● cancelled")
            self.status_lbl.configure(fg="gray")
            self.app.log_system(f"[{self.name}] build cancelled.")
        else:
            self.status_var.set(f"● build failed ({rc})")
            self.status_lbl.configure(fg="#a72b29")
            self.app.log_system(f"[{self.name}] build FAILED (rc={rc}).")
        self.build_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.clean_btn.configure(state="normal")
        if self.app.build_chain_active:
            self.app.advance_build_chain(rc)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("F1TENTH Launch GUI")
        root.geometry("1600x820")

        self.log_queue: queue.Queue = queue.Queue()
        self.build_chain: list["BuildRow"] = []
        self.build_chain_active = False

        # --- header ---
        header = ttk.Frame(root, padding=(8, 8, 8, 0))
        header.pack(fill="x")
        ttk.Label(header, text="F1TENTH Launch GUI",
                  font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(header,
                  text="Left: launches + sensor status + colcon build per workspace. "
                       "base ⊇ sensor + uros, slam ⊇ sensor, sensor ⊇ rplidar + imu. "
                       "Right: live log.",
                  foreground="#555", wraplength=1560, justify="left").pack(anchor="w")

        # --- body: left column (controls) | right column (log) ---
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # --- left: launches ---
        lframe = ttk.LabelFrame(left, text="Launches", padding=8)
        lframe.pack(fill="x", padx=8, pady=8)

        self.rows: list[LaunchRow] = []
        self._by_name: dict[str, LaunchRow] = {}
        for i, (name, color, cmd) in enumerate(LAUNCHES):
            row = LaunchRow(lframe, i, name, color, cmd, self)
            self.rows.append(row)
            self._by_name[name] = row

        # --- left: sensors (USB/TTY) — one compact horizontal row ---
        sframe = ttk.LabelFrame(left, text="Sensors (USB/TTY)", padding=4)
        sframe.pack(fill="x", padx=8, pady=(0, 6))
        self.sensor_rows: list[SensorCell] = [
            SensorCell(sframe, name, path, self) for name, path in SENSORS
        ]

        # --- left: workspaces ---
        wframe = ttk.LabelFrame(left, text="Workspaces — colcon build", padding=8)
        wframe.pack(fill="x", padx=8, pady=(0, 8))

        self.build_rows: list[BuildRow] = []
        self._build_by_name: dict[str, BuildRow] = {}
        for i, (name, color, path) in enumerate(WORKSPACES):
            row = BuildRow(wframe, i, name, color, path, self)
            self.build_rows.append(row)
            self._build_by_name[name] = row

        # --- left: shared controls ---
        ctrl = ttk.Frame(left, padding=(8, 0, 8, 4))
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Stop all launches",
                   command=self.stop_all).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Kill orphans",
                   command=self.kill_orphans).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Kill joy zombies",
                   command=self.kill_joystick_zombies).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Build all",
                   command=self.build_all).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Clear log",
                   command=self.clear_log).pack(side="left", padx=2)
        self.autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto-scroll log",
                        variable=self.autoscroll_var).pack(side="left", padx=8)

        # --- right: notebook with Log + IMU tabs ---
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True, padx=(0, 8), pady=8)

        # Tab 1 — log
        log_frame = ttk.Frame(notebook, padding=2)
        notebook.add(log_frame, text="Log")
        self.log = scrolledtext.ScrolledText(
            log_frame, wrap="none", font=("monospace", 9),
            background="#1e1e1e", foreground="#dcdcdc", insertbackground="white",
        )
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        for name, color, _ in LAUNCHES:
            self.log.tag_config(name, foreground=color)
        for name, color, _ in WORKSPACES:
            self.log.tag_config(name, foreground=color)
        self.log.tag_config("__system__", foreground="#f0d000",
                            font=("monospace", 9, "bold"))

        # Tab 2 — IMU monitor (read-only, just for measuring mount tilt etc.)
        imu_frame = ttk.Frame(notebook, padding=2)
        notebook.add(imu_frame, text="IMU")
        self.imu_monitor = ImuMonitor(imu_frame, root)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self._drain_log)
        root.after(100, self._refresh_sensors)

    # ----- lookups -----
    def row_by_name(self, name: str) -> LaunchRow:
        return self._by_name[name]

    def build_row_by_name(self, name: str) -> BuildRow:
        return self._build_by_name[name]

    def log_system(self, msg: str):
        self.log_queue.put(("__system__", msg))

    # ----- build-all chain -----
    def build_all(self):
        if self.build_chain_active:
            self.log_system("Build all: already running.")
            return
        if any(r.is_running() for r in self.build_rows):
            self.log_system("Build all: another build is in progress, skipping.")
            return
        self.build_chain = list(self.build_rows)
        self.build_chain_active = True
        self.log_system("Build all → " +
                        " → ".join(r.name for r in self.build_chain))
        self._next_in_chain()

    def _next_in_chain(self):
        if not self.build_chain:
            self.build_chain_active = False
            self.log_system("Build all: complete ✓")
            return
        self.build_chain.pop(0).build()

    def advance_build_chain(self, last_rc: int):
        if last_rc != 0:
            self.build_chain = []
            self.build_chain_active = False
            self.log_system(f"Build all: aborted (last build returned {last_rc}).")
            return
        self._next_in_chain()

    # ----- log pump -----
    def _drain_log(self):
        try:
            while True:
                tag, payload = self.log_queue.get_nowait()
                if tag == "__launch_exit__":
                    name, rc = payload
                    self._append("__system__", f"[{name}] exited with code {rc}")
                    self.row_by_name(name).on_exit(rc)
                elif tag == "__build_exit__":
                    name, rc = payload
                    self._append("__system__",
                                 f"[{name}] colcon build finished, rc={rc}")
                    self.build_row_by_name(name).on_exit(rc)
                elif tag == "__system__":
                    self._append("__system__", payload)
                else:
                    self._append(tag, f"[{tag}] {payload}")
        except queue.Empty:
            pass
        self.root.after(50, self._drain_log)

    def _refresh_sensors(self):
        for row in self.sensor_rows:
            row.refresh()
        self.root.after(2000, self._refresh_sensors)

    def _append(self, tag, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag)
        if self.autoscroll_var.get():
            self.log.see("end")
        self.log.configure(state="disabled")

    # ----- bulk actions -----
    def stop_all(self):
        for row in self.rows:
            row.stop()

    def kill_orphans(self):
        """Broad sweep — pkill -9 every known F1TENTH node pattern. Use after
        a crash / external launch / GUI restart left stuff behind."""
        self.log_system(
            f"Kill orphans: pkill -9 -f on {len(ORPHAN_PATTERNS)} patterns…")
        for pat in ORPHAN_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.log_system("Kill orphans: done.")

    def kill_joystick_zombies(self):
        """Targeted sweep — pkill -9 only joystick subsystem processes
        (joy_node + joystick.py + the launch wrappers). Leaves SLAM, sensors,
        actuators, etc. alone. Use when /steering_angle has multiple publishers
        and the servo is jittering — Start joystick launch fresh afterwards."""
        self.log_system(
            f"Kill joy zombies: pkill -9 -f on "
            f"{len(JOYSTICK_KILL_PATTERNS)} patterns…")
        for pat in JOYSTICK_KILL_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.log_system("Kill joy zombies: done. Start joystick launch fresh.")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def on_close(self):
        # Cancel everything in flight before destroying the window.
        for row in self.rows:
            row.stop()
        for row in self.build_rows:
            row.cancel()
        if hasattr(self, "imu_monitor"):
            self.imu_monitor.shutdown()
        self.root.after(700, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
