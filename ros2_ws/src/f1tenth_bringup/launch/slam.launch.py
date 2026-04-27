import os
import tempfile

import yaml
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    gym_ros_dir  = get_package_share_directory("f1tenth_gym_ros")
    bringup_dir  = get_package_share_directory("f1tenth_bringup")

    # --- patch sim.yaml with the correct map path ---
    config_path = os.path.join(gym_ros_dir, "config", "sim.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    params   = config_dict["bridge"]["ros__parameters"]
    maps_dir = os.path.join(gym_ros_dir, "maps")
    map_name = LaunchConfiguration("map").perform(context)
    map_path = os.path.realpath(os.path.join(maps_dir, map_name))
    params["map_path"] = map_path

    patched_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="sim_slam_", delete=False
    )
    yaml.dump(config_dict, patched_config, default_flow_style=False)
    patched_config.close()

    # --- nodes ---
    bridge_node = Node(
        package="f1tenth_gym_ros",
        executable="gym_bridge",
        name="bridge",
        parameters=[patched_config.name],
    )

    ego_robot_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ego_robot_state_publisher",
        parameters=[
            {
                "robot_description": Command(
                    ["xacro ", os.path.join(gym_ros_dir, "launch", "ego_racecar.xacro")]
                )
            }
        ],
        remappings=[("/robot_description", "ego_robot_description")],
    )

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(bringup_dir, "config", "slam_toolbox_params.yaml"),
        ],
    )

    # Stanley controller runs alongside SLAM; publishes to /stanley/drive
    # so the GUI can relay it to /drive when autopilot is toggled ON
    stanley_node = Node(
        package="f1tenth_controller",
        executable="stanley_avoidance.py",
        name="stanley_avoidance_node",
        output="screen",
        parameters=[
            {"velocity": 3.0},
            {"K_E": 1.0},
            {"K_H": 0.5},
            {"K_p": 0.5},
            {"K_p_obstacle": 0.8},
            {"min_lookahead": 1.0},
            {"max_lookahead": 3.0},
            {"min_lookahead_speed": 3.0},
            {"max_lookahead_speed": 6.0},
            {"velocity_percentage": 0.5},
            {"velocity_min": 1.0},
            {"velocity_max": 2.0},
            {"steering_limit": 25.0},
            {"grid_width_meters": 6.0},
            {"cells_per_meter": 20},
        ],
        remappings=[("/drive", "/stanley/drive")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=["-d", os.path.join(bringup_dir, "rviz", "slam.rviz")],
        additional_env={"DISPLAY": os.environ.get("DISPLAY", ":0")},
    )

    slam_gui_node = Node(
        package="f1tenth_joy",
        executable="slam_gui.py",
        name="slam_gui",
        output="screen",
        additional_env={"DISPLAY": os.environ.get("DISPLAY", ":0")},
    )

    return [
        bridge_node,
        ego_robot_publisher,
        slam_toolbox_node,
        stanley_node,
        rviz_node,
        slam_gui_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="Spielberg_map",
                description="Gym map name (physics only; SLAM builds its own map)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
