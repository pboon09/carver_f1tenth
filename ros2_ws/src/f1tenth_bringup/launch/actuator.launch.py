import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    vesc_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_vesc_driver"), "launch"
    )

    vesc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vesc_launch_dir, "vesc.launch.py")
        ),
        launch_arguments={
            "serial_port": "/dev/VESC",
        }.items(),
    )

    micro_ros_agent = Node(
        package="micro_ros_agent",
        executable="micro_ros_agent",
        name="micro_ros_agent",
        output="screen",
        arguments=["serial", "--dev", "/dev/STM32", "-b", "115200"],
    )

    return LaunchDescription([
        vesc_launch,
        micro_ros_agent,
    ])