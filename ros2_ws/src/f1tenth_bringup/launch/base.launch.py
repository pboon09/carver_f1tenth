import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    actuator_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"), "launch"
    )
    sensor_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"), "launch"
    )

    actuator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(actuator_launch_dir, "actuator.launch.py")
        ),
    )

    sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensor_launch_dir, "sensor.launch.py")
        ),
    )

    return LaunchDescription([
        actuator_launch,
        sensor_launch,
    ])