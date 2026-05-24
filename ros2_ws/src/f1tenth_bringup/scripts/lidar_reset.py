#!/usr/bin/env python3
"""
RPLidar C1 pre-launch reset.

Symptom this fixes:
    [rplidar_node]: RPLidar health status : 2
    [rplidar_node]: Error, RPLidar internal error detected. Please reboot
                    the device to retry.

Why it happens: previous rplidar_node was SIGKILL'd (e.g. by the GUI Stop
button) before it could send STOP_MOTOR to the device. The motor keeps
spinning, the internal state machine gets confused, and the next start
sees a `health = 2` error code and refuses to scan.

What this script does: opens /dev/rplidar at the C1 baudrate, sends:
  - STOP            (0xA5 0x25)   stop scan/motor
  - RESET           (0xA5 0x40)   software reboot
…then waits ~0.8 s for the lidar's internal MCU to come back. After that
the rplidar_node can launch cleanly.

Usage:
    lidar_reset.py [device]      device defaults to /dev/rplidar

Exit codes:
    0  reset sent (or device already clear)
    1  serial port unavailable (device unplugged / wrong path)
"""

import sys
import time

try:
    import serial
except ImportError:
    print("python3-serial not installed; skipping lidar reset.", file=sys.stderr)
    sys.exit(0)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/rplidar"
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 460800

    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except (serial.SerialException, OSError) as e:
        print(f"[lidar_reset] could not open {port}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # STOP — quiets scan + motor
        ser.write(bytes([0xA5, 0x25]))
        ser.flush()
        time.sleep(0.1)
        # RESET — full software reboot of the lidar's MCU
        ser.write(bytes([0xA5, 0x40]))
        ser.flush()
        # MCU reboot takes ~800 ms; do not slam the next driver into it.
        time.sleep(0.9)
        # Drain anything the lidar said while booting so the next driver
        # doesn't see boot-up garbage.
        ser.reset_input_buffer()
        print(f"[lidar_reset] reset OK on {port}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()