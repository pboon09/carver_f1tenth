import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """
    On-car bringup in LOCALIZATION mode: sensors + slam_toolbox(localization)
    + actuators. Loads a pre-built pose graph from the path baked into
    `slam_toolbox_localization.yaml` (default: /home/carver/carver_f1tenth/map/my_map).

    Mirror of base.launch.py except it includes `localization.launch.py`
    instead of `mapping.launch.py`. Two separate files (rather than a
    `slam_mode:=` arg on base.launch.py) so the dispatch is trivial and
    impossible to mis-route.
    """
    bringup_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"), "launch"
    )
    slam_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_slam"), "launch"
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_launch_dir, "localization.launch.py")
        ),
    )

    actuator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch_dir, "actuator.launch.py")
        ),
    )
    delayed_actuator = TimerAction(period=5.0, actions=[actuator_launch])

    return LaunchDescription([
        LogInfo(msg="[base_loc.launch.py] mode = LOCALIZATION"),
        localization_launch,
        delayed_actuator,
    ])
