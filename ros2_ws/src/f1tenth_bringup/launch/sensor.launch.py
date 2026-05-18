import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    rplidar_launch_dir = os.path.join(
        get_package_share_directory("rplidar_ros"), "launch"
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_launch_dir, "rplidar_c1_launch.py")
        ),
    )

    bno_stick = Node(
        package="bno055_usb_stick",
        executable="bno055_usb_stick_node_script.py",
        output="screen",
    )

    return LaunchDescription([
        rplidar_launch,
        bno_stick,
    ])