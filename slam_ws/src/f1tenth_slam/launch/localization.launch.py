#!/usr/bin/env python3
"""
F1TENTH SLAM localization bringup — slam_toolbox flavour, map-file selectable.

Same shape as mapping.launch.py except slam_toolbox runs in localization
mode against a pre-built pose graph. The pose graph basename is passed in
via the `map_file_name:=<path>` launch argument (default below). The GUI's
"Localize against selected map" button in the Maps tab sets this arg from
whichever saved map you pick.

Required files at <map_file_name>.posegraph and .data — save them while
mapping with:
    ros2 service call /slam_toolbox/serialize_map \
        slam_toolbox/srv/SerializePoseGraph \
        "{filename: '<map_file_name>'}"

Run directly:
    ros2 launch f1tenth_slam localization.launch.py
    ros2 launch f1tenth_slam localization.launch.py map_file_name:=/path/to/foo
"""

import datetime
import os

import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState


def generate_launch_description():
    pkg = get_package_share_directory("f1tenth_slam")
    slam_params = os.path.join(pkg, "config", "slam_toolbox_localize_param.yaml")
    ekf_params = os.path.join(pkg, "config", "ekf_imu.yaml")
    rviz_config = os.path.join(pkg, "rviz", "loc.rviz")

    declare_map = DeclareLaunchArgument(
        "map_file_name",
        default_value="/home/carver/carver_f1tenth/map/my_map",
        description="Pose-graph basename for slam_toolbox to load "
                    "(no extension — .posegraph and .data are appended).",
    )
    map_file_name = LaunchConfiguration("map_file_name")

    urdf_path = os.path.join(
        get_package_share_directory("f1tenth_urdf"),
        "urdf", "simplify_fullremake.urdf",
    )
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("f1tenth_bringup"),
                "launch", "sensor.launch.py",
            )
        ),
    )

    tf_base_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_base_to_laser",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0.0771",
            "--yaw", "0", "--pitch", "0", "--roll", "0",
            "--frame-id", "base_link", "--child-frame-id", "laser",
        ],
    )

    tf_odom_to_base = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params],
    )

    # slam_toolbox in localization mode. We pass the YAML file PLUS a
    # one-key dict that overrides `map_file_name` at runtime — that's how
    # the launch-arg gets injected into slam_toolbox's parameter set.
    slam_toolbox_node = LifecycleNode(
        package="slam_toolbox",
        executable="localization_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        parameters=[
            slam_params,
            {"map_file_name": map_file_name},
        ],
    )

    configure_slam = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    activate_slam = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_toolbox_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                ))
            ],
        )
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher = Node(
        package="f1tenth_joy",
        executable="cmd_to_joint_state.py",
        name="cmd_to_joint_state",
        output="screen",
        parameters=[{"gear_ratio": 29.75}],
    )

    imu_filter = Node(
        package="f1tenth_joy",
        executable="imu_filter.py",
        name="imu_filter",
        output="screen",
        parameters=[{
            "calibration_seconds": 5.0,
            "gyro_median_window": 1,
            "gyro_ma_window": 8,
            "gyro_ma_block_mode": True,
        }],
    )

    vesc_velocity = Node(
        package="f1tenth_vesc_driver",
        executable="vesc_velocity.py",
        name="vesc_velocity",
        output="screen",
        parameters=[{
            "wheel_radius": 0.0594,
            "gear_ratio": 29.75,
            "wheelbase": 0.30,           # URDF: 0.192757 − (−0.10719)
            "publish_rate": 100.0,
            "base_frame": "basefootprint",
            # Loose — this car slips a lot. IMU gyro + slam scan-match
            # are the trustworthy signals; wheel odom is dead-reckon filler.
            "vx_covariance": 2.0,        # 1σ ≈ 1.4 m/s
            "vyaw_covariance": 1.0,      # 1σ ≈ 57°/s
        }],
    )

    trajectory_publisher = Node(
        package="f1tenth_joy",
        executable="trajectory_publisher.py",
        name="trajectory_publisher",
        output="screen",
        parameters=[{
            "parent_frame": "map",
            "child_frame": "base_link",
            "rate": 100.0,            # sample TF at 100 Hz — matches EKF + map→odom rate
            "publish_rate": 100.0,    # republish full path at 100 Hz
            "max_poses": 5000,
            "min_distance": 0.02,
        }],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription([
        declare_map,
        LogInfo(msg=["[localization.launch.py] map_file_name = ", map_file_name]),
        sensor_launch,
        tf_base_to_laser,
        tf_odom_to_base,
        robot_state_publisher,
        joint_state_publisher,
        imu_filter,
        vesc_velocity,
        slam_toolbox_node,
        activate_slam,
        configure_slam,
        trajectory_publisher,
        rviz_node,
    ])
