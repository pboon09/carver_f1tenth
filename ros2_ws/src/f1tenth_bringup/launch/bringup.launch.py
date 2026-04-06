import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_dir = get_package_share_directory("f1tenth_bringup")
    launch_dir = os.path.join(bringup_dir, "launch")

    map_arg = DeclareLaunchArgument(
        "map",
        default_value="Spielberg_map",
        description="Map name without extension (e.g. Spielberg_map, Spielberg_map_easy)",
    )

    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="manual",
        description="Controller mode: manual or auto",
    )

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "simulation.launch.py")
        ),
        launch_arguments={"map": LaunchConfiguration("map")}.items(),
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "controller.launch.py")
        ),
        launch_arguments={"mode": LaunchConfiguration("mode")}.items(),
    )

    return LaunchDescription(
        [
            map_arg,
            mode_arg,
            simulation_launch,
            controller_launch,
        ]
    )
