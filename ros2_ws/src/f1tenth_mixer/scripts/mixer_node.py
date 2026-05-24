#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np


class MixerNode(Node):
    def __init__(self):
        super().__init__('mixer_node')
        
        self.vesc_cmd_publisher = self.create_publisher(Int32, '/vesc/cmd', 10)
        self.steering_publisher = self.create_publisher(Float32, '/steering_angle', 10)

        self.ackermann_subscription = self.create_subscription(AckermannDriveStamped, '/drive', self.ackermann_callback, 10)

        self.declare_parameter('max_rpm', 10000)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('wheel_radius', 0.06)

        self.max_rpm = int(self.get_parameter("max_rpm").value)
        self.max_steering_angle = float(self.get_parameter("max_steering_angle").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)

        self.get_logger().info('Mixer Node Started')
        self.get_logger().info(f'Max RPM: {self.max_rpm}, Max Steering: {self.max_steering_angle:.4f} rad')        

    def ackermann_callback(self, msg):
        
        speed = msg.drive.speed
        steering_angle = max(min(msg.drive.steering_angle, self.max_steering_angle), -self.max_steering_angle) * -1.0
        rpm_command = min(max(int(60.0 * speed * 29.75 / (2.0 * np.pi * self.wheel_radius)), -self.max_rpm), self.max_rpm)
        
        self.vesc_cmd_publisher.publish(Int32(data=rpm_command))
        self.steering_publisher.publish(Float32(data=steering_angle))

def main(args=None):
    rclpy.init(args=args)
    node = MixerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
