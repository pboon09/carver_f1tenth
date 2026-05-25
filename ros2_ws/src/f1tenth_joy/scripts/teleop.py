#!/usr/bin/env python3

from __future__ import annotations

import sys
import termios
import tty

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node


KEY_BINDINGS = {
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (0.0, 1.0),
    "d": (0.0, -1.0),
    "q": (1.0, 1.0),
    "e": (1.0, -1.0),
    "x": (0.0, 0.0),
}

DEFAULT_MAX_SPEED = 2.0
DEFAULT_MAX_STEER = 0.4189
DEFAULT_SPEED_STEP = 0.1
DEFAULT_STEER_STEP = 0.05


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("teleop_node")

        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("max_steer", DEFAULT_MAX_STEER)
        self.declare_parameter("speed_step", DEFAULT_SPEED_STEP)
        self.declare_parameter("steer_step", DEFAULT_STEER_STEP)

        drive_topic = self.get_parameter("drive_topic").value
        self.max_speed = self.get_parameter("max_speed").value
        self.max_steer = self.get_parameter("max_steer").value
        self.speed_step = self.get_parameter("speed_step").value
        self.steer_step = self.get_parameter("steer_step").value

        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.speed = 0.0
        self.steer = 0.0

        self.print_usage()

    def print_usage(self) -> None:
        msg = (
            "\n"
            "F1TENTH Keyboard Teleop\n"
            "-----------------------\n"
            "  W/S : increase/decrease speed\n"
            "  A/D : steer left/right\n"
            "  Q/E : forward-left / forward-right\n"
            "  X   : full stop\n"
            "  ESC : quit\n"
            f"\n"
            f"  Max speed : {self.max_speed:.1f} m/s\n"
            f"  Max steer : {self.max_steer:.2f} rad\n"
        )
        print(msg)

    def publish_drive(self) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steer
        self.publisher.publish(msg)

    def process_key(self, key: str) -> bool:
        if key == "\x1b":
            return False

        if key not in KEY_BINDINGS:
            return True

        speed_dir, steer_dir = KEY_BINDINGS[key]

        if key == "x":
            self.speed = 0.0
            self.steer = 0.0
        else:
            if speed_dir != 0.0:
                self.speed += speed_dir * self.speed_step
                self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
            if steer_dir != 0.0:
                self.steer += steer_dir * self.steer_step
                self.steer = max(-self.max_steer, min(self.max_steer, self.steer))

        self.publish_drive()
        print(
            f"\r  speed: {self.speed:+.2f} m/s  |  steer: {self.steer:+.2f} rad   ",
            end="",
            flush=True,
        )
        return True

    def run(self) -> None:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                key = sys.stdin.read(1)
                if not self.process_key(key):
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.speed = 0.0
            self.steer = 0.0
            self.publish_drive()
            print("\nStopped.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
