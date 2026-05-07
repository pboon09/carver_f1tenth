#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32
from std_msgs.msg import Float32
import time

class JoystickNode(Node):
    def __init__(self):
        super().__init__('joystick_node')

        self.vesc_cmd_publisher = self.create_publisher(Int32, '/vesc/cmd', 10)
        self.steering_publisher = self.create_publisher(Float32, '/steering_angle', 10)

        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.declare_parameter('max_rpm', 10000)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('deadzone', 0.1)

        self.max_rpm = self.get_parameter('max_rpm').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.deadzone = self.get_parameter('deadzone').value

        self.axis_neutral_positions = {
            0: 0.0,
            1: 0.0,
            3: 0.0,
        }

        self.current_joy = None
        self.last_joy_time = time.time()
        self.joy_timeout = 0.5

        self.create_timer(0.01, self.publish_velocity)
        self.create_timer(0.1, self.check_joy_timeout)

        self.get_logger().info('VESC Joystick Node Started')
        self.get_logger().info(f'Max RPM: {self.max_rpm}, Max Steering: {self.max_steering_angle:.4f} rad')
        self.get_logger().info('Control: Left stick Y=RPM, Right stick X=Steering')

    def joy_callback(self, msg):
        """Process joystick input"""
        self.current_joy = msg
        self.last_joy_time = time.time()

    def apply_axis_deadzone(self, axis_index, raw_value):
        """Apply deadzone relative to the axis's neutral position"""
        if axis_index not in self.axis_neutral_positions:
            return 0.0

        neutral_pos = self.axis_neutral_positions[axis_index]
        deviation = raw_value - neutral_pos

        if abs(deviation) < self.deadzone:
            return 0.0

        sign = 1 if deviation >= 0 else -1
        scaled_deviation = (abs(deviation) - self.deadzone) / (1.0 - self.deadzone)
        return sign * scaled_deviation

    def publish_velocity(self):
        """Publish RPM and steering commands based on current joystick state"""
        rpm_cmd = Int32()
        steering_cmd = Float32()

        if self.current_joy is None:
            self.vesc_cmd_publisher.publish(rpm_cmd)
            self.steering_publisher.publish(steering_cmd)
            return

        if len(self.current_joy.axes) < 4:
            self.vesc_cmd_publisher.publish(rpm_cmd)
            self.steering_publisher.publish(steering_cmd)
            return

        left_stick_y = self.apply_axis_deadzone(1, self.current_joy.axes[1])
        right_stick_x = self.apply_axis_deadzone(3, self.current_joy.axes[3])

        rpm_cmd.data = int(left_stick_y * self.max_rpm)
        steering_cmd.data = right_stick_x * self.max_steering_angle

        self.vesc_cmd_publisher.publish(rpm_cmd)
        self.steering_publisher.publish(steering_cmd)

        # self.get_logger().info(
        #         f'RPM: {rpm_cmd.data}, Steering: {steering_cmd.data:.4f}',
        #         throttle_duration_sec=1.0
        #     )

    def check_joy_timeout(self):
        """Stop robot if no joystick messages received recently"""
        if time.time() - self.last_joy_time > self.joy_timeout:
            self.vesc_cmd_publisher.publish(Int32(data=0))
            self.steering_publisher.publish(Float32(data=0.0))
            if self.current_joy is not None:
                self.get_logger().warn('Joystick timeout - stopping robot')
                self.current_joy = None

def main(args=None):
    rclpy.init(args=args)
    node = JoystickNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.vesc_cmd_publisher.publish(Int32(data=0))
        node.steering_publisher.publish(Float32(data=0.0))
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()