import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """
    On-car bringup in SLAM (mapping) mode: sensors + slam_toolbox(mapping)
    + actuators. For localization mode, use `base_loc.launch.py` instead.
    """
    bringup_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"), "launch"
    )
    slam_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_slam"), "launch"
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_launch_dir, "mapping.launch.py")
        ),
    )

    actuator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch_dir, "actuator.launch.py")
        ),
    )
    delayed_actuator = TimerAction(period=5.0, actions=[actuator_launch])

    return LaunchDescription([
        LogInfo(msg="[base_slam.launch.py] mode = SLAM (mapping)"),
        slam_launch,
        delayed_actuator,
    ])
