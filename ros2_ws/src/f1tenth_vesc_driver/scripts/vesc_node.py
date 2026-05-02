#!/usr/bin/env python3

import serial
import struct
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import JointState


def crc16_ccitt(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


def make_packet(payload):
    n = len(payload)
    if n < 256:
        header = bytes([0x02, n])
    else:
        header = bytes([0x03, (n >> 8) & 0xFF, n & 0xFF])
    crc = crc16_ccitt(payload)
    return header + payload + bytes([(crc >> 8) & 0xFF, crc & 0xFF, 0x03])


def read_packet(ser, timeout=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b[0] == 0x02:
            l = ser.read(1)
            if len(l) < 1:
                return None
            length = l[0]
        elif b[0] == 0x03:
            l = ser.read(2)
            if len(l) < 2:
                return None
            length = (l[0] << 8) | l[1]
        else:
            continue
        payload = ser.read(length)
        crc_bytes = ser.read(2)
        stop = ser.read(1)
        if len(payload) != length or len(crc_bytes) != 2 or len(stop) != 1 or stop[0] != 0x03:
            return None
        crc_recv = (crc_bytes[0] << 8) | crc_bytes[1]
        if crc16_ccitt(payload) != crc_recv:
            return None
        return bytes(payload)
    return None


def parse_values(payload):
    if len(payload) < 54 or payload[0] != 4:
        return None
    return {
        'temp_fet': struct.unpack('>h', payload[1:3])[0] / 10.0,
        'temp_motor': struct.unpack('>h', payload[3:5])[0] / 10.0,
        'i_motor': struct.unpack('>i', payload[5:9])[0] / 100.0,
        'i_in': struct.unpack('>i', payload[9:13])[0] / 100.0,
        'duty': struct.unpack('>h', payload[21:23])[0] / 1000.0,
        'rpm': struct.unpack('>i', payload[23:27])[0],
        'v_in': struct.unpack('>h', payload[27:29])[0] / 10.0,
        'fault': payload[53],
    }


class VescNode(Node):
    def __init__(self):
        super().__init__('vesc_node')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 230400)
        self.declare_parameter('cmd_rate', 50.0)

        self.serial_port = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        cmd_rate = self.get_parameter('cmd_rate').value

        self.subscription = self.create_subscription(
            Int32,
            '/vesc/cmd',
            self.cmd_callback,
            10
        )

        self.state_publisher = self.create_publisher(
            JointState,
            '/vesc/state',
            10
        )

        self.ser = None
        self.cmd_rpm = 0
        self.last_query = 0.0
        self.last_connection_attempt = 0.0
        self.connection_failed_logged = False

        self.connect_to_vesc()

        timer_period = 1.0 / cmd_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def connect_to_vesc(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.05)
            self.get_logger().info(f'Connected to VESC on {self.serial_port}')
            self.connection_failed_logged = False
        except Exception as e:
            if not self.connection_failed_logged:
                self.get_logger().warn(f'Failed to open serial port: {e}')
                self.connection_failed_logged = True

    def cmd_callback(self, msg):
        self.cmd_rpm = msg.data

    def timer_callback(self):
        if self.ser is None:
            if time.time() - self.last_connection_attempt > 2.0:
                self.connect_to_vesc()
                self.last_connection_attempt = time.time()
            return

        try:
            payload = struct.pack('>Bi', 8, int(self.cmd_rpm))
            self.ser.write(make_packet(payload))

            if time.time() - self.last_query > 0.05:
                self.ser.write(make_packet(bytes([4])))
                pkt = read_packet(self.ser, timeout=0.05)
                if pkt is not None:
                    values = parse_values(pkt)
                    if values:
                        state = JointState()
                        state.header.stamp = self.get_clock().now().to_msg()
                        state.name = ['rpm', 'temp_fet', 'temp_motor', 'i_motor', 'i_in', 'duty', 'v_in', 'fault']
                        state.position = [
                            float(values['rpm']),
                            values['temp_fet'],
                            values['temp_motor'],
                            values['i_motor'],
                            values['i_in'],
                            values['duty'],
                            values['v_in'],
                            float(values['fault'])
                        ]
                        self.state_publisher.publish(state)
                self.last_query = time.time()
        except Exception as e:
            self.get_logger().error(f'Error communicating with VESC: {e}')
            self.ser = None

    def destroy_node(self):
        if self.ser is not None:
            payload = struct.pack('>Bi', 8, 0)
            self.ser.write(make_packet(payload))
            time.sleep(0.1)
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VescNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
