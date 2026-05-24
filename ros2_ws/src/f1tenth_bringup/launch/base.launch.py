import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """
    Full on-car bringup: sensors + actuators + SLAM + RViz.

    Pulls in `f1tenth_slam/mapping.launch.py` for the SLAM half — which
    itself includes `sensor.launch.py` from this package, so we don't
    include it again here (would race for the USB devices). Then layers
    in `actuator.launch.py` (VESC + micro-ROS bridge) on a 5 s delay so
    the lidar has time to finish negotiating its USB serial link first.

    Requires slam_ws to be sourced at runtime — there's no exec_depend on
    f1tenth_slam in this package.xml because that would create a circular
    package dep (f1tenth_slam already exec_depends on f1tenth_bringup).
    The GUI launcher and the README handle the sourcing.
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
        slam_launch,
        delayed_actuator,
    ])
