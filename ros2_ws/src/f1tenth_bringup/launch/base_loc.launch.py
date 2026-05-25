import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             LogInfo, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    On-car bringup in LOCALIZATION mode (slam_toolbox flavour).

    Picks which pose graph to load via:
        ros2 launch f1tenth_bringup base_loc.launch.py \
            map_file_name:=/home/carver/carver_f1tenth/map/<basename>

    The Maps tab in tools/launcher.py constructs this command with the
    basename you select.
    """
    bringup_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"), "launch"
    )
    slam_launch_dir = os.path.join(
        get_package_share_directory("f1tenth_slam"), "launch"
    )

    declare_map = DeclareLaunchArgument(
        "map_file_name",
        default_value="/home/carver/carver_f1tenth/map/my_map",
        description="Pose-graph basename (no extension) for slam_toolbox "
                    "localization to load.",
    )
    map_file_name = LaunchConfiguration("map_file_name")

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_launch_dir, "localization.launch.py")
        ),
        launch_arguments={"map_file_name": map_file_name}.items(),
    )

    actuator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch_dir, "actuator.launch.py")
        ),
    )
    delayed_actuator = TimerAction(period=5.0, actions=[actuator_launch])

    return LaunchDescription([
        declare_map,
        LogInfo(msg=["[base_loc.launch.py] mode = LOCALIZATION, "
                     "map_file_name = ", map_file_name]),
        localization_launch,
        delayed_actuator,
    ])
