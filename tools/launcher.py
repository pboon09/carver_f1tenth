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
from typing import Optional

# Optional rclpy — used only by the IMU monitor tab. If ROS isn't sourced in
# the GUI's environment the import fails and the IMU tab shows a hint instead
# of crashing the whole launcher.
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Path
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
from tkinter import ttk, scrolledtext, messagebox


ROS_SETUP = "source /opt/ros/jazzy/setup.bash"
ROS2_WS   = "source /home/carver/carver_f1tenth/ros2_ws/install/setup.bash"
SENSOR_WS = "source /home/carver/carver_f1tenth/sensor_ws/install/setup.bash"
SLAM_WS   = "source /home/carver/carver_f1tenth/slam_ws/install/setup.bash"
UROS_WS   = "source /home/carver/uros_ws/install/setup.bash"

# Where saved maps live. Used by the Maps tab for save / list / discard.
MAP_DIR = "/home/carver/carver_f1tenth/map"


def chain(*parts: str) -> str:
    return " && ".join(parts)


def clean_env() -> dict:
    """Return a copy of os.environ with /snap/* entries stripped from
    LD_LIBRARY_PATH and PATH. Required when the launcher is started from
    a snap-installed VS Code (or any snap'd terminal) — the snap leaks its
    bundled libpthread.so.0 into LD_LIBRARY_PATH which conflicts with the
    system glibc and crashes RViz with
        "symbol lookup error ... undefined symbol __libc_pthread_init"
    """
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "PATH", "PYTHONPATH",
                "GTK_PATH", "GIO_MODULE_DIR", "LOCPATH"):
        v = env.get(var, "")
        if "/snap/" not in v:
            continue
        clean = ":".join(p for p in v.split(":") if "/snap/" not in p)
        env[var] = clean
    return env


# (name, color, bash command)
LAUNCHES = [
    ("slam base",      "#1f6feb",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_bringup base_slam.launch.py")),
    ("sensor launch",  "#0a3069",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS,
              "ros2 launch f1tenth_bringup sensor.launch.py")),
    ("slam launch",    "#1a7f37",
        chain(ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
              "ros2 launch f1tenth_slam mapping.launch.py")),
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
    "slam base":     ["slam launch", "sensor launch",
                      "rplidar launch", "imu launch", "uros launch"],
    "sensor launch": ["rplidar launch", "imu launch"],
    "slam launch":   ["sensor launch", "rplidar launch", "imu launch"],
}

# Mutually exclusive launches — only conflict checking, NO shadow display.
# Mapping (slam base / slam launch) and localization (owned by MapManager
# in the Maps tab) can't run simultaneously — they fight over the USB
# devices and would both try to publish map→odom. The MapManager-owned
# localize process is checked separately inside LaunchRow.start().
MUTEX: dict[str, list[str]] = {}

# `pkill -9 -f <pattern>` patterns to fire after SIGKILLing the process group,
# so any orphan that escaped the group (e.g. detached child, separate ros2 run
# someone started by hand) still goes away. Patterns match the *full command
# line* — keep them specific enough to not nuke unrelated processes.
KILL_PATTERNS: dict[str, list[str]] = {
    "slam base":       ["base_slam\\.launch\\.py", "base_loc\\.launch\\.py",
                        "mapping\\.launch\\.py", "localization\\.launch\\.py",
                        "rplidar_node", "bno055_usb_stick",
                        "slam_toolbox",
                        "static_transform_publisher",
                        "robot_state_publisher", "joint_state_publisher",
                        "cmd_to_joint_state\\.py", "imu_calibrator\\.py",
                        "imu_filter\\.py",
                        "vesc_velocity\\.py", "ekf_node",
                        "vesc_node\\.py", "micro_ros_agent",
                        "rviz2 .*f1tenth_slam.*\\(slam\\|loc\\)\\.rviz"],
    "sensor launch":   ["sensor\\.launch\\.py", "rplidar_node", "bno055_usb_stick"],
    "slam launch":     ["mapping\\.launch\\.py", "rplidar_node", "bno055_usb_stick",
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
    "nav2_amcl", "nav2_map_server", "nav2_lifecycle_manager",
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
        conflicts += [c for c in MUTEX.get(self.name, [])
                      if self.app.row_by_name(c).is_running()]
        # MapManager owns the localize subprocess (Maps tab). Block any
        # mapping-side launch from starting while it's running.
        if self.name in ("slam base", "slam launch", "sensor launch",
                          "rplidar launch", "imu launch"):
            if (getattr(self.app, "map_manager", None) is not None
                    and self.app.map_manager.is_localizing()):
                conflicts.append("localize (Maps tab)")
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
            env=clean_env(),
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
# Map manager tab — save / discard / list saved maps in MAP_DIR
# ---------------------------------------------------------------------------

class MapManager:
    """Save the live map to disk, discard (reset) the live map in slam_toolbox,
    and list whatever's already saved on disk in MAP_DIR.

    File pairs tracked in the listing:
      .pgm / .yaml         occupancy grid (saved by nav2_map_server map_saver_cli)
      .posegraph / .data   slam_toolbox pose graph (saved via serialize_map svc)
    """

    TRACKED_EXTS = (".pgm", ".yaml", ".posegraph", ".data")

    # Patterns to pkill -9 after stopping the localize process, to mop up
    # nodes that escape the process group.
    LOC_KILL_PATTERNS = [
        "base_loc\\.launch\\.py", "localization\\.launch\\.py",
        "rplidar_node", "bno055_usb_stick",
        "slam_toolbox", "static_transform_publisher",
        "cmd_to_joint_state\\.py", "imu_filter\\.py",
        "robot_state_publisher", "ekf_node",
        "vesc_velocity\\.py", "vesc_node\\.py", "micro_ros_agent",
        "rviz2 .*f1tenth_slam.*loc\\.rviz",
    ]

    # Patterns for the FGM toggle (mixer + gap_follow).
    FGM_KILL_PATTERNS = [
        "f1tenth_mixer/lib/f1tenth_mixer/mixer_node\\.py",
        "f1tenth_controller/lib/f1tenth_controller/gap_follow\\.py",
        "ros2 run f1tenth_mixer mixer_node",
        "ros2 run f1tenth_controller gap_follow",
    ]

    # Patterns for the Pure Pursuit toggle (mixer + pure_pursuit).
    PP_KILL_PATTERNS = [
        "f1tenth_mixer/lib/f1tenth_mixer/mixer_node\\.py",
        "f1tenth_controller/lib/f1tenth_controller/pure_pursuit\\.py",
        "ros2 run f1tenth_mixer mixer_node",
        "ros2 run f1tenth_controller pure_pursuit",
    ]

    # Patterns for the Lattice toggle (mixer + lattice_planner).
    LATTICE_KILL_PATTERNS = [
        "f1tenth_mixer/lib/f1tenth_mixer/mixer_node\\.py",
        "f1tenth_controller/lib/f1tenth_controller/lattice_planner\\.py",
        "ros2 run f1tenth_mixer mixer_node",
        "ros2 run f1tenth_controller lattice_planner",
    ]

    # Union of all controller kill patterns — used to sweep leftover
    # processes from a prior launcher session before spawning a new one.
    # Restarting the launcher while a controller is running otherwise
    # leaves the old mixer+controller alive (start_new_session=True), and
    # the new launcher (which knows nothing of them) happily spawns a
    # second instance that fights for /drive.
    ALL_CONTROLLER_KILL_PATTERNS = [
        "f1tenth_mixer/lib/f1tenth_mixer/mixer_node\\.py",
        "f1tenth_controller/lib/f1tenth_controller/gap_follow\\.py",
        "f1tenth_controller/lib/f1tenth_controller/pure_pursuit\\.py",
        "f1tenth_controller/lib/f1tenth_controller/lattice_planner\\.py",
        "ros2 run f1tenth_mixer mixer_node",
        "ros2 run f1tenth_controller gap_follow",
        "ros2 run f1tenth_controller pure_pursuit",
        "ros2 run f1tenth_controller lattice_planner",
    ]

    # Where path_v_<basename>.yaml racelines live (generated by
    # raceline_generator.py). The Pure Pursuit / Lattice buttons look up
    # path_v_<selected basename>.yaml under here.
    RACELINE_DIR = "/home/carver/carver_f1tenth/ros2_ws/src/f1tenth_controller/path"

    def __init__(self, parent, app):
        self.app = app
        self.map_dir = MAP_DIR
        self.loc_proc: Optional[subprocess.Popen] = None
        self.fgm_proc: Optional[subprocess.Popen] = None
        self.pp_proc: Optional[subprocess.Popen] = None
        self.lattice_proc: Optional[subprocess.Popen] = None

        # On startup, sweep any orphaned controller/mixer processes left by
        # a previous launcher session that crashed or was closed without
        # stopping its controller. Without this, the next click would spawn
        # a second instance fighting the orphan on /drive.
        self._sweep_orphan_controllers(reason="launcher startup")

        # --- Live map (in-memory in slam_toolbox) ----------------------------
        live = ttk.LabelFrame(parent,
                              text="Live map (slam_toolbox in memory)",
                              padding=8)
        live.pack(fill="x", padx=8, pady=(8, 4))

        name_row = ttk.Frame(live)
        name_row.pack(fill="x", pady=(0, 6))
        ttk.Label(name_row, text="Basename:").pack(side="left", padx=4)
        self.name_var = tk.StringVar(value="my_map")
        ttk.Entry(name_row, textvariable=self.name_var, width=24).pack(
            side="left", padx=4)
        ttk.Label(name_row,
                  text=f"  →  {self.map_dir}/<basename>.{{pgm,yaml,posegraph,data}}",
                  foreground="#888").pack(side="left")

        save_row = ttk.Frame(live)
        save_row.pack(fill="x", pady=(0, 4))
        ttk.Button(save_row, text="Save .pgm + .yaml",
                   command=self._save_occgrid).pack(side="left", padx=2)
        ttk.Button(save_row, text="Save .posegraph + .data",
                   command=self._save_posegraph).pack(side="left", padx=2)
        ttk.Button(save_row, text="Save BOTH",
                   command=self._save_both).pack(side="left", padx=2)

        discard_row = ttk.Frame(live)
        discard_row.pack(fill="x", pady=(4, 0))
        ttk.Button(discard_row, text="Discard live map (reset slam_toolbox)",
                   command=self._discard_live).pack(side="left", padx=2)
        ttk.Label(discard_row,
                  text="Wipes the in-memory map + pose graph. Files on disk "
                       "are untouched.",
                  foreground="#888").pack(side="left", padx=6)

        # --- Saved maps on disk ----------------------------------------------
        disk = ttk.LabelFrame(parent, text="Saved maps on disk", padding=4)
        disk.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        top = ttk.Frame(disk)
        top.pack(fill="x", pady=2)
        ttk.Label(top, text="Folder:").pack(side="left", padx=4)
        tk.Label(top, text=self.map_dir, fg="#0a3069",
                 font=("monospace", 10)).pack(side="left", padx=4)
        ttk.Button(top, text="Refresh", width=9,
                   command=self._refresh_list).pack(side="right")

        cols = ("name", "files", "size")
        self.tree = ttk.Treeview(disk, columns=cols, show="headings",
                                  height=8)
        self.tree.heading("name", text="Basename")
        self.tree.heading("files", text="Files present")
        self.tree.heading("size", text="Total size")
        self.tree.column("name", width=180, anchor="w")
        self.tree.column("files", width=260, anchor="w")
        self.tree.column("size", width=100, anchor="e")
        self.tree.pack(fill="both", expand=True, padx=2, pady=2)

        del_row = ttk.Frame(disk)
        del_row.pack(fill="x", pady=2)
        ttk.Button(del_row, text="Delete selected from disk",
                   command=self._delete_selected).pack(side="left", padx=2)
        ttk.Label(del_row,
                  text="Deletes all .pgm/.yaml/.posegraph/.data files for "
                       "the selected basename.",
                  foreground="#888").pack(side="left", padx=6)

        # --- Localize against selected map ----------------------------------
        loc = ttk.LabelFrame(parent,
                             text="Localize against selected map "
                                  "(loc base — slam_toolbox)",
                             padding=8)
        loc.pack(fill="x", padx=8, pady=(4, 8))

        self.loc_status_var = tk.StringVar(value="● stopped")
        self.loc_status_lbl = tk.Label(loc, textvariable=self.loc_status_var,
                                        fg="gray", anchor="w",
                                        font=("monospace", 10))
        self.loc_status_lbl.pack(fill="x", padx=2, pady=(0, 4))

        loc_btns = ttk.Frame(loc)
        loc_btns.pack(fill="x")
        self.loc_start_btn = ttk.Button(
            loc_btns, text="Start loc base (selected map)",
            command=self._start_localize)
        self.loc_start_btn.pack(side="left", padx=2)
        self.loc_stop_btn = ttk.Button(
            loc_btns, text="Stop loc base", state="disabled",
            command=self._stop_localize)
        self.loc_stop_btn.pack(side="left", padx=2)
        ttk.Label(loc_btns,
                  text="Pick a basename above with .posegraph + .data present.",
                  foreground="#888").pack(side="left", padx=6)

        # --- Autonomous toggle: FGM (mixer_node + gap_follow.py) ------------
        # Runs alongside the localize launch. Writes /vesc/cmd and
        # /steering_angle (consumed by the actuator stack + vesc_velocity).
        # No topic conflict with the localize chain as long as joystick.py
        # isn't also running.
        auto = ttk.Frame(loc)
        auto.pack(fill="x", pady=(8, 0))
        self.fgm_btn = tk.Button(
            auto, text="FGM", width=6,
            font=("TkDefaultFont", 10, "bold"),
            command=self._toggle_fgm)
        self.fgm_btn.pack(side="left", padx=2)
        self.fgm_sublabel_var = tk.StringVar(value="")
        self.fgm_sublabel = tk.Label(
            auto, textvariable=self.fgm_sublabel_var,
            fg="#a72b29", font=("TkDefaultFont", 8))
        self.fgm_sublabel.pack(side="left", padx=(6, 0))
        self._fgm_default_bg = self.fgm_btn.cget("bg")
        self._fgm_default_fg = self.fgm_btn.cget("fg")

        # Pure Pursuit toggle — needs a raceline matching the selected map
        # (path_v_<basename>.yaml). Pose is read from /dedicate_odom (map
        # frame, from trajectory_publisher), speed from /odometry/filtered.
        self.pp_btn = tk.Button(
            auto, text="Pure Pursuit", width=12,
            font=("TkDefaultFont", 10, "bold"),
            command=self._toggle_pp)
        self.pp_btn.pack(side="left", padx=2)
        self.pp_sublabel_var = tk.StringVar(value="")
        self.pp_sublabel = tk.Label(
            auto, textvariable=self.pp_sublabel_var,
            fg="#a72b29", font=("TkDefaultFont", 8))
        self.pp_sublabel.pack(side="left", padx=(6, 0))
        self._pp_default_bg = self.pp_btn.cget("bg")
        self._pp_default_fg = self.pp_btn.cget("fg")

        # Lattice toggle — same raceline convention as Pure Pursuit.
        self.lattice_btn = tk.Button(
            auto, text="Lattice", width=8,
            font=("TkDefaultFont", 10, "bold"),
            command=self._toggle_lattice)
        self.lattice_btn.pack(side="left", padx=2)
        self.lattice_sublabel_var = tk.StringVar(value="")
        self.lattice_sublabel = tk.Label(
            auto, textvariable=self.lattice_sublabel_var,
            fg="#a72b29", font=("TkDefaultFont", 8))
        self.lattice_sublabel.pack(side="left", padx=(6, 0))
        self._lattice_default_bg = self.lattice_btn.cget("bg")
        self._lattice_default_fg = self.lattice_btn.cget("fg")

        self._refresh_list()

    # ----- list -----

    def _refresh_list(self):
        os.makedirs(self.map_dir, exist_ok=True)
        basenames = set()
        for f in os.listdir(self.map_dir):
            base, ext = os.path.splitext(f)
            if ext.lower() in self.TRACKED_EXTS:
                basenames.add(base)

        for item in self.tree.get_children():
            self.tree.delete(item)
        for base in sorted(basenames):
            present, total = [], 0
            for ext in self.TRACKED_EXTS:
                p = os.path.join(self.map_dir, base + ext)
                if os.path.exists(p):
                    present.append(ext.lstrip("."))
                    total += os.path.getsize(p)
            self.tree.insert("", "end", values=(
                base, " ".join(present), self._fmt_size(total)))

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024
        return f"{n:.1f} TiB"

    def _selected_basename(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.item(sel[0], "values")[0]

    # ----- discard live (slam_toolbox in-memory reset) -----

    def _discard_live(self):
        if not messagebox.askyesno(
                "Discard live map",
                "This calls /slam_toolbox/reset and wipes the in-memory "
                "pose graph + map. Files on disk are NOT touched. Continue?"):
            return
        cmd = chain(
            ROS_SETUP, SLAM_WS,
            "ros2 service call /slam_toolbox/reset slam_toolbox/srv/Reset '{}'",
        )
        self._run_save_cmd(cmd, "discard live map (slam_toolbox reset)")

    # ----- delete saved files -----

    def _delete_selected(self):
        base = self._selected_basename()
        if not base:
            messagebox.showinfo("No selection", "Pick a map in the list first.")
            return
        if not messagebox.askyesno(
                "Delete from disk",
                f"Delete all files for '{base}' in\n{self.map_dir}/ ?"):
            return
        removed = []
        for ext in self.TRACKED_EXTS:
            p = os.path.join(self.map_dir, base + ext)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(os.path.basename(p))
                except OSError as e:
                    self.app.log_system(f"[map] couldn't delete {p}: {e}")
        self.app.log_system(f"[map] deleted {base}: {', '.join(removed)}")
        self._refresh_list()

    # ----- save -----

    def _validate_name(self) -> Optional[str]:
        name = self.name_var.get().strip()
        if not name or "/" in name or "\\" in name:
            messagebox.showwarning(
                "Invalid name",
                "Basename must be non-empty and have no slashes.")
            return None
        return name

    def _save_occgrid(self):
        name = self._validate_name()
        if name is None:
            return
        path = os.path.join(self.map_dir, name)
        cmd = chain(
            ROS_SETUP, ROS2_WS,
            f"ros2 run nav2_map_server map_saver_cli -f {path}",
        )
        self._run_save_cmd(cmd, f"occupancy grid → {path}.pgm + .yaml")

    def _save_posegraph(self):
        name = self._validate_name()
        if name is None:
            return
        path = os.path.join(self.map_dir, name)
        cmd = chain(
            ROS_SETUP, SLAM_WS,
            f"ros2 service call /slam_toolbox/serialize_map "
            f"slam_toolbox/srv/SerializePoseGraph "
            f"\"{{filename: '{path}'}}\"",
        )
        self._run_save_cmd(cmd, f"pose graph → {path}.posegraph + .data")

    def _save_both(self):
        name = self._validate_name()
        if name is None:
            return
        path = os.path.join(self.map_dir, name)
        cmd = chain(
            ROS_SETUP, ROS2_WS, SLAM_WS,
            f"ros2 run nav2_map_server map_saver_cli -f {path}",
            f"ros2 service call /slam_toolbox/serialize_map "
            f"slam_toolbox/srv/SerializePoseGraph "
            f"\"{{filename: '{path}'}}\"",
        )
        self._run_save_cmd(cmd, f"both → {path}.{{pgm,yaml,posegraph,data}}")

    # ----- localize (MapManager-owned launch supervisor) -----

    def is_localizing(self) -> bool:
        return self.loc_proc is not None and self.loc_proc.poll() is None

    def _refresh_loc_buttons(self):
        if self.is_localizing():
            self.loc_status_var.set(
                f"● running (pid {self.loc_proc.pid})")
            self.loc_status_lbl.configure(fg="#1a7f37")
            self.loc_start_btn.configure(state="disabled")
            self.loc_stop_btn.configure(state="normal")
        else:
            self.loc_status_var.set("● stopped")
            self.loc_status_lbl.configure(fg="gray")
            self.loc_start_btn.configure(state="normal")
            self.loc_stop_btn.configure(state="disabled")

    def _start_localize(self):
        if self.is_localizing():
            return

        # Mutex with slam base / slam launch — running mapping at the
        # same time would fight over USB + map→odom publisher.
        for blocker in ("slam base", "slam launch"):
            row = self.app._by_name.get(blocker)
            if row is not None and row.is_running():
                msg = (f"Can't start localize — '{blocker}' is running. "
                       f"Stop it first.")
                self.app.log_system(msg)
                messagebox.showwarning("Already running", msg)
                return

        base = self._selected_basename()
        if not base:
            messagebox.showinfo("No selection",
                                 "Pick a saved map in the list first.")
            return
        # Must have both pose-graph files.
        for ext in (".posegraph", ".data"):
            if not os.path.exists(os.path.join(self.map_dir, base + ext)):
                messagebox.showerror(
                    "Missing files",
                    f"Selected basename '{base}' is missing "
                    f"{base}{ext}.\nSave the pose graph from slam base "
                    f"first ('Save .posegraph + .data' above).")
                return

        map_path = os.path.join(self.map_dir, base)
        cmd = chain(
            ROS_SETUP, SENSOR_WS, ROS2_WS, SLAM_WS,
            f"ros2 launch f1tenth_bringup base_loc.launch.py "
            f"map_file_name:={map_path}",
        )
        self.loc_proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            start_new_session=True,
            env=clean_env(),
        )
        self.app.log_system(
            f"[loc base] started (pid {self.loc_proc.pid}) "
            f"map = {map_path}")
        self._refresh_loc_buttons()
        threading.Thread(target=self._read_loc_output, daemon=True).start()

    def _read_loc_output(self):
        assert self.loc_proc is not None and self.loc_proc.stdout is not None
        for line in iter(self.loc_proc.stdout.readline, ""):
            self.app.log_queue.put(("loc base", line.rstrip()))
        self.loc_proc.stdout.close()
        rc = self.loc_proc.wait()
        self.app.log_queue.put(("__system__",
                                 f"[loc base] exited with code {rc}"))
        # Drop the handle on the UI thread.
        self.app.root.after(50, self._on_loc_exit)

    def _on_loc_exit(self):
        self.loc_proc = None
        self._refresh_loc_buttons()

    def _stop_localize(self):
        if not self.is_localizing():
            return
        try:
            os.killpg(os.getpgid(self.loc_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pat in self.LOC_KILL_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.app.log_system(
            f"[loc base] SIGKILL + pkill -9 sent "
            f"(patterns: {', '.join(self.LOC_KILL_PATTERNS)})")
        self.loc_status_var.set("● killing…")
        self.loc_status_lbl.configure(fg="#a72b29")
        self.loc_stop_btn.configure(state="disabled")

    # ----- Defensive sweep for orphaned controller/mixer processes ------

    def _sweep_orphan_controllers(self, reason: str = ""):
        """pkill -9 every mixer + controller pattern. Used at launcher
        startup and right before spawning any controller, so a prior
        session's leftover processes can't compete with the new one on
        /drive."""
        killed_any = False
        for pat in self.ALL_CONTROLLER_KILL_PATTERNS:
            rc = subprocess.run(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode
            if rc == 0:
                killed_any = True
        if killed_any:
            self.app.log_system(
                f"[sweep] killed orphan mixer/controller processes "
                f"({reason})" if reason else "[sweep] killed orphan processes")

    # ----- FGM toggle (mixer + gap_follow) ------------------------------

    def is_fgm_running(self) -> bool:
        return self.fgm_proc is not None and self.fgm_proc.poll() is None

    def _refresh_fgm_button(self):
        if self.is_fgm_running():
            self.fgm_btn.configure(bg="#d11a2a", fg="white",
                                    activebackground="#a51220",
                                    activeforeground="white")
            self.fgm_sublabel_var.set("active")
        else:
            self.fgm_btn.configure(bg=self._fgm_default_bg,
                                    fg=self._fgm_default_fg,
                                    activebackground=self._fgm_default_bg,
                                    activeforeground=self._fgm_default_fg)
            self.fgm_sublabel_var.set("")

    def _toggle_fgm(self):
        if self.is_fgm_running():
            self._stop_fgm()
        else:
            self._start_fgm()

    def _start_fgm(self):
        # Mutex with the other autonomous toggles — they all write to /drive
        # (via mixer_node) so only one can run at a time.
        if self.is_pp_running():
            messagebox.showwarning("Pure Pursuit running",
                                    "Stop Pure Pursuit before starting FGM.")
            return
        if self.is_lattice_running():
            messagebox.showwarning("Lattice running",
                                    "Stop Lattice before starting FGM.")
            return
        # Belt-and-suspenders: kill any orphan processes the mutex check
        # missed (e.g. controller from a prior launcher session — our
        # self.*_proc handles are None, so is_*_running() returned False).
        self._sweep_orphan_controllers(reason="pre-FGM")
        # gap_follow defaults to algorithm="bubble"; the bbox (rectangular
        # corridor sweep) variant accounts for both W and L and doesn't
        # freeze on "no gaps" in tight spaces. Flip the -p to "bubble"
        # to use the legacy behavior.
        cmd = chain(
            ROS_SETUP, ROS2_WS,
            "ros2 run f1tenth_mixer mixer_node.py & "
            "ros2 run f1tenth_controller gap_follow.py "
            "--ros-args -p algorithm:=bbox & "
            "wait",
        )
        self.fgm_proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            start_new_session=True,
            env=clean_env(),
        )
        self.app.log_system(
            f"[FGM] started (pid {self.fgm_proc.pid}) — "
            f"mixer_node + gap_follow")
        self._refresh_fgm_button()
        threading.Thread(target=self._read_fgm_output, daemon=True).start()

    def _stop_fgm(self):
        if not self.is_fgm_running():
            return
        try:
            os.killpg(os.getpgid(self.fgm_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pat in self.FGM_KILL_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.app.log_system(
            f"[FGM] SIGKILL + pkill -9 sent "
            f"(patterns: {', '.join(self.FGM_KILL_PATTERNS)})")

    def _read_fgm_output(self):
        assert self.fgm_proc is not None and self.fgm_proc.stdout is not None
        for line in iter(self.fgm_proc.stdout.readline, ""):
            self.app.log_queue.put(("FGM", line.rstrip()))
        self.fgm_proc.stdout.close()
        rc = self.fgm_proc.wait()
        self.app.log_queue.put(("__system__",
                                 f"[FGM] exited with code {rc}"))
        self.app.root.after(50, self._on_fgm_exit)

    def _on_fgm_exit(self):
        self.fgm_proc = None
        self._refresh_fgm_button()

    # ----- Pure Pursuit toggle (mixer + pure_pursuit) -------------------

    def _raceline_for_selected(self) -> Optional[str]:
        """Look up path_v_<basename>.yaml for the map selected in the tree.
        Returns the absolute path, or None (and shows a dialog) if no map
        is selected or the raceline file doesn't exist yet."""
        base = self._selected_basename()
        if not base:
            messagebox.showinfo(
                "No map selected",
                "Pick a saved map in the list first — the raceline "
                "path_v_<basename>.yaml is looked up from it.")
            return None
        raceline = os.path.join(self.RACELINE_DIR, f"path_v_{base}.yaml")
        if not os.path.exists(raceline):
            messagebox.showerror(
                "Missing raceline",
                f"Expected raceline file not found:\n  {raceline}\n\n"
                f"Generate it first with:\n\n"
                f"  python3 ros2_ws/src/f1tenth_controller/scripts/"
                f"raceline_generator.py \\\n"
                f"      --map {self.map_dir}/{base}.yaml --lane mincurv \\\n"
                f"      --out {raceline}")
            return None
        return raceline

    def is_pp_running(self) -> bool:
        return self.pp_proc is not None and self.pp_proc.poll() is None

    def _refresh_pp_button(self):
        if self.is_pp_running():
            self.pp_btn.configure(bg="#1a7f37", fg="white",
                                   activebackground="#136028",
                                   activeforeground="white")
            self.pp_sublabel_var.set("active")
        else:
            self.pp_btn.configure(bg=self._pp_default_bg,
                                   fg=self._pp_default_fg,
                                   activebackground=self._pp_default_bg,
                                   activeforeground=self._pp_default_fg)
            self.pp_sublabel_var.set("")

    def _toggle_pp(self):
        if self.is_pp_running():
            self._stop_pp()
        else:
            self._start_pp()

    def _start_pp(self):
        # Mutex with the other autonomous toggles — they all write to /drive
        # (via mixer_node) so only one can run at a time.
        if self.is_fgm_running():
            messagebox.showwarning("FGM running",
                                    "Stop FGM before starting Pure Pursuit.")
            return
        if self.is_lattice_running():
            messagebox.showwarning("Lattice running",
                                    "Stop Lattice before starting Pure Pursuit.")
            return
        self._sweep_orphan_controllers(reason="pre-PP")
        raceline = self._raceline_for_selected()
        if raceline is None:
            return
        # Topics + speed clamp live in the script defaults
        # (/dedicate_odom, /odometry/filtered, max 0.5 m/s) — only the
        # raceline path is dynamic per-map, so that's all we pass.
        cmd = chain(
            ROS_SETUP, ROS2_WS,
            "ros2 run f1tenth_mixer mixer_node.py & "
            "ros2 run f1tenth_controller pure_pursuit.py "
            f"--ros-args -p waypoints_path:={raceline} & "
            "wait",
        )
        self.pp_proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            start_new_session=True,
            env=clean_env(),
        )
        self.app.log_system(
            f"[PP] started (pid {self.pp_proc.pid}) — "
            f"mixer_node + pure_pursuit, raceline = {raceline}")
        self._refresh_pp_button()
        threading.Thread(target=self._read_pp_output, daemon=True).start()

    def _stop_pp(self):
        if not self.is_pp_running():
            return
        try:
            os.killpg(os.getpgid(self.pp_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pat in self.PP_KILL_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.app.log_system(
            f"[PP] SIGKILL + pkill -9 sent "
            f"(patterns: {', '.join(self.PP_KILL_PATTERNS)})")

    def _read_pp_output(self):
        assert self.pp_proc is not None and self.pp_proc.stdout is not None
        for line in iter(self.pp_proc.stdout.readline, ""):
            self.app.log_queue.put(("PP", line.rstrip()))
        self.pp_proc.stdout.close()
        rc = self.pp_proc.wait()
        self.app.log_queue.put(("__system__",
                                 f"[PP] exited with code {rc}"))
        self.app.root.after(50, self._on_pp_exit)

    def _on_pp_exit(self):
        self.pp_proc = None
        self._refresh_pp_button()

    # ----- Lattice toggle (mixer + lattice_planner) ---------------------

    def is_lattice_running(self) -> bool:
        return self.lattice_proc is not None and self.lattice_proc.poll() is None

    def _refresh_lattice_button(self):
        if self.is_lattice_running():
            self.lattice_btn.configure(bg="#0a3069", fg="white",
                                        activebackground="#082352",
                                        activeforeground="white")
            self.lattice_sublabel_var.set("active")
        else:
            self.lattice_btn.configure(bg=self._lattice_default_bg,
                                        fg=self._lattice_default_fg,
                                        activebackground=self._lattice_default_bg,
                                        activeforeground=self._lattice_default_fg)
            self.lattice_sublabel_var.set("")

    def _toggle_lattice(self):
        if self.is_lattice_running():
            self._stop_lattice()
        else:
            self._start_lattice()

    def _start_lattice(self):
        if self.is_fgm_running():
            messagebox.showwarning("FGM running",
                                    "Stop FGM before starting Lattice.")
            return
        if self.is_pp_running():
            messagebox.showwarning("Pure Pursuit running",
                                    "Stop Pure Pursuit before starting Lattice.")
            return
        self._sweep_orphan_controllers(reason="pre-Lattice")
        raceline = self._raceline_for_selected()
        if raceline is None:
            return
        # Topics + speed clamp live in the script defaults
        # (/dedicate_odom, /odometry/filtered, max 0.5 m/s) — only the
        # raceline path is dynamic per-map, so that's all we pass.
        cmd = chain(
            ROS_SETUP, ROS2_WS,
            "ros2 run f1tenth_mixer mixer_node.py & "
            "ros2 run f1tenth_controller lattice_planner.py "
            f"--ros-args -p waypoints_path:={raceline} & "
            "wait",
        )
        self.lattice_proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            start_new_session=True,
            env=clean_env(),
        )
        self.app.log_system(
            f"[Lattice] started (pid {self.lattice_proc.pid}) — "
            f"mixer_node + lattice_planner, raceline = {raceline}")
        self._refresh_lattice_button()
        threading.Thread(target=self._read_lattice_output, daemon=True).start()

    def _stop_lattice(self):
        if not self.is_lattice_running():
            return
        try:
            os.killpg(os.getpgid(self.lattice_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pat in self.LATTICE_KILL_PATTERNS:
            subprocess.Popen(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.app.log_system(
            f"[Lattice] SIGKILL + pkill -9 sent "
            f"(patterns: {', '.join(self.LATTICE_KILL_PATTERNS)})")

    def _read_lattice_output(self):
        assert self.lattice_proc is not None and self.lattice_proc.stdout is not None
        for line in iter(self.lattice_proc.stdout.readline, ""):
            self.app.log_queue.put(("Lattice", line.rstrip()))
        self.lattice_proc.stdout.close()
        rc = self.lattice_proc.wait()
        self.app.log_queue.put(("__system__",
                                 f"[Lattice] exited with code {rc}"))
        self.app.root.after(50, self._on_lattice_exit)

    def _on_lattice_exit(self):
        self.lattice_proc = None
        self._refresh_lattice_button()

    def _run_save_cmd(self, cmd: str, desc: str):
        os.makedirs(self.map_dir, exist_ok=True)
        self.app.log_system(f"[map] starting save: {desc}")

        def _run():
            try:
                proc = subprocess.Popen(
                    ["bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1, text=True,
                    env=clean_env(),
                )
                for line in iter(proc.stdout.readline, ""):
                    self.app.log_queue.put(("map", line.rstrip()))
                proc.stdout.close()
                rc = proc.wait()
                self.app.log_queue.put(("__system__",
                    f"[map] save done ({desc}) rc={rc}"))
                self.app.root.after(100, self._refresh_list)
            except Exception as e:
                self.app.log_queue.put(("__system__",
                                         f"[map] save failed: {e!r}"))

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Trajectory view tab — live matplotlib plot of /trajectory
# ---------------------------------------------------------------------------

class TrajectoryView:
    """Embedded matplotlib plot of nav_msgs/Path from /trajectory.

    Mirrors the green Trajectory display in RViz but inside the GUI so
    you can glance at the path without alt-tabbing. Subscribes lazily
    via the shared ImuMonitor rclpy node if it exists, otherwise creates
    its own node.
    """

    REDRAW_HZ = 5

    def __init__(self, parent, root, app):
        self.root = root
        self.app = app
        self._lock = threading.Lock()
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._dirty = True
        self.node = None
        self.exec = None

        # --- header / status ---
        ttk.Label(parent, text="Vehicle trajectory (/trajectory)",
                  font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.status_lbl = tk.Label(parent, text="initializing…",
                                    fg="gray",
                                    font=("monospace", 10), anchor="w")
        self.status_lbl.pack(anchor="w", padx=8, pady=2, fill="x")

        btns = ttk.Frame(parent, padding=(8, 0, 8, 4))
        btns.pack(fill="x")
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left",
                                                                   padx=2)
        ttk.Button(btns, text="Autofit", command=self._autofit).pack(
            side="left", padx=2)

        if not MPL_AVAILABLE:
            self.status_lbl.configure(
                text="matplotlib not installed — can't render plot.",
                fg="#a72b29")
            return
        if not RCLPY_AVAILABLE:
            self.status_lbl.configure(
                text="rclpy not importable.",
                fg="#a72b29")
            return

        # --- figure / canvas ---
        self.fig = Figure(figsize=(5, 5), dpi=100,
                           facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111, facecolor="#1e1e1e")
        self.ax.set_aspect("equal")
        self.ax.grid(True, color="#444", linewidth=0.5)
        self.ax.tick_params(colors="#dcdcdc")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#888")
        self.ax.set_xlabel("x (m)", color="#dcdcdc")
        self.ax.set_ylabel("y (m)", color="#dcdcdc")
        (self.line,) = self.ax.plot([], [], "-", color="#19c832",
                                       linewidth=1.6)
        (self.head,) = self.ax.plot([], [], "o", color="#19c832",
                                       markersize=6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=8, pady=(4, 8))

        # --- subscribe ---
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node("launcher_trajectory_view")
            self.node.create_subscription(Path, "/trajectory",
                                            self._on_path, 10)
            self.exec = SingleThreadedExecutor()
            self.exec.add_node(self.node)
            threading.Thread(target=self._spin, daemon=True).start()
            self.status_lbl.configure(
                text="subscribed to /trajectory — waiting for messages…",
                fg="#bf8700")
        except Exception as e:
            self.status_lbl.configure(text=f"rclpy init failed: {e!r}",
                                       fg="#a72b29")

        self.root.after(int(1000 / self.REDRAW_HZ), self._tick)

    def _spin(self):
        try:
            self.exec.spin()
        except Exception:
            pass

    def _on_path(self, msg: "Path"):
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        with self._lock:
            self._xs = xs
            self._ys = ys
            self._dirty = True

    def _tick(self):
        if not self._dirty:
            self.root.after(int(1000 / self.REDRAW_HZ), self._tick)
            return
        with self._lock:
            xs = list(self._xs)
            ys = list(self._ys)
            self._dirty = False
        if xs:
            self.line.set_data(xs, ys)
            self.head.set_data([xs[-1]], [ys[-1]])
            self.status_lbl.configure(
                text=f"{len(xs)} poses | head ({xs[-1]:+.2f}, {ys[-1]:+.2f}) m",
                fg="#1a7f37")
            self._autofit()
        self.canvas.draw_idle()
        self.root.after(int(1000 / self.REDRAW_HZ), self._tick)

    def _autofit(self):
        with self._lock:
            xs, ys = list(self._xs), list(self._ys)
        if not xs:
            return
        pad = 1.0
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)
        self.canvas.draw_idle()

    def _clear(self):
        with self._lock:
            self._xs = []
            self._ys = []
            self._dirty = True
        self.line.set_data([], [])
        self.head.set_data([], [])
        self.status_lbl.configure(text="cleared (waiting for new poses…)",
                                    fg="gray")
        self.canvas.draw_idle()

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
            env=clean_env(),
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
        self.log.tag_config("map", foreground="#8e44ad")

        # Tab 2 — IMU monitor (read-only, just for measuring mount tilt etc.)
        imu_frame = ttk.Frame(notebook, padding=2)
        notebook.add(imu_frame, text="IMU")
        self.imu_monitor = ImuMonitor(imu_frame, root)

        # Tab 3 — Map manager (save / discard / list maps on disk + localize)
        map_frame = ttk.Frame(notebook, padding=2)
        notebook.add(map_frame, text="Maps")
        self.map_manager = MapManager(map_frame, self)

        # Tab 4 — Trajectory plot (live /trajectory)
        traj_frame = ttk.Frame(notebook, padding=2)
        notebook.add(traj_frame, text="Trajectory")
        self.trajectory_view = TrajectoryView(traj_frame, root, self)

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
        if hasattr(self, "map_manager") and self.map_manager.is_localizing():
            self.map_manager._stop_localize()

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
        if hasattr(self, "map_manager") and self.map_manager.is_localizing():
            self.map_manager._stop_localize()
        if hasattr(self, "imu_monitor"):
            self.imu_monitor.shutdown()
        if hasattr(self, "trajectory_view"):
            self.trajectory_view.shutdown()
        self.root.after(700, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
