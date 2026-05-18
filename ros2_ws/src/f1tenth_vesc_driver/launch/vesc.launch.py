from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM1',
        description='Serial port for VESC'
    )

    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='115200',
        description='Baudrate for VESC serial connection'
    )

    vesc_node = Node(
        package='f1tenth_vesc_driver',
        executable='vesc_node.py',
        name='vesc_node',
        parameters=[
            {'serial_port': LaunchConfiguration('serial_port')},
            {'baudrate': LaunchConfiguration('baudrate')},
        ],
        output='screen',
    )

    return LaunchDescription([
        serial_port_arg,
        baudrate_arg,
        vesc_node,
    ])
