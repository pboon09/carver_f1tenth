import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
    )

    joystick = Node(
        package="f1tenth_joy",
        executable="joystick.py",
        name="joystick_node",
        output="screen",
        parameters=[
            {"max_rpm": 10000},
            {"max_steering_angle": 0.4189},
            {"deadzone": 0.1},
        ],
    )

    return LaunchDescription([
        joy_node,
        joystick,
    ])