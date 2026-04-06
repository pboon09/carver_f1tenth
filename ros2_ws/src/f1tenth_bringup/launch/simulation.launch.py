import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


MAPS_DIR = os.path.expanduser(
    "~/carver_f1tenth/ros2_ws/src/f1tenth_gym_ros/maps"
)


def launch_setup(context):
    gym_ros_dir = get_package_share_directory("f1tenth_gym_ros")
    config_path = os.path.join(gym_ros_dir, "config", "sim.yaml")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    params = config_dict["bridge"]["ros__parameters"]
    has_opp = params["num_agent"] > 1

    map_name = LaunchConfiguration("map").perform(context)
    map_path = os.path.realpath(os.path.join(MAPS_DIR, map_name))
    params["map_path"] = map_path

    patched_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="sim_", delete=False
    )
    yaml.dump(config_dict, patched_config, default_flow_style=False)
    patched_config.close()

    bridge_node = Node(
        package="f1tenth_gym_ros",
        executable="gym_bridge",
        name="bridge",
        parameters=[patched_config.name],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=[
            "-d",
            os.path.join(gym_ros_dir, "launch", "gym_bridge.rviz"),
        ],
    )

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        parameters=[
            {"yaml_filename": map_path + ".yaml"},
            {"topic": "map"},
            {"frame_id": "map"},
            {"output": "screen"},
            {"use_sim_time": True},
        ],
    )

    nav_lifecycle_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    ego_robot_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ego_robot_state_publisher",
        parameters=[
            {
                "robot_description": Command(
                    [
                        "xacro ",
                        os.path.join(gym_ros_dir, "launch", "ego_racecar.xacro"),
                    ]
                )
            }
        ],
        remappings=[("/robot_description", "ego_robot_description")],
    )

    opp_robot_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="opp_robot_state_publisher",
        parameters=[
            {
                "robot_description": Command(
                    [
                        "xacro ",
                        os.path.join(gym_ros_dir, "launch", "opp_racecar.xacro"),
                    ]
                )
            }
        ],
        remappings=[("/robot_description", "opp_robot_description")],
    )

    nodes = [
        bridge_node,
        rviz_node,
        map_server_node,
        nav_lifecycle_node,
        ego_robot_publisher,
    ]

    if has_opp:
        nodes.append(opp_robot_publisher)

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="Spielberg_map",
                description="Map name without extension (e.g. Spielberg_map, Spielberg_map_easy)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
