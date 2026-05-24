#!/usr/bin/env python3
"""
F1TENTH localization bringup — Nav2 AMCL flavour.

Stack:
  sensors (rplidar + bno055)            → /scan, /imu/data
  imu_filter                             → /imu_filter
  vesc_velocity                          → /vesc/twist
  robot_localization EKF                 → odom → base_link TF
  map_server   (loads my_map.pgm/.yaml)  → /map
  amcl         (particle filter)         → map → odom TF
  lifecycle_manager                       → activates map_server + amcl
  rviz

What's NOT here vs the mapping launch:
  - slam_toolbox is gone — replaced by AMCL
  - the EKF is kept (AMCL needs odom→base_link from somewhere)

Run:
    ros2 launch f1tenth_slam localization.launch.py
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("f1tenth_slam")
    amcl_params = os.path.join(pkg, "config", "amcl_params.yaml")
    ekf_params = os.path.join(pkg, "config", "ekf_imu.yaml")
    rviz_config = os.path.join(pkg, "rviz", "slam.rviz")
    map_yaml = "/home/carver/carver_f1tenth/map/my_map.yaml"

    # Fail fast — AMCL with a missing map yaml is just as confusing as the
    # old slam_toolbox-missing-posegraph failure.
    if not os.path.exists(map_yaml):
        raise FileNotFoundError(
            f"Map yaml missing: {map_yaml}\n"
            f"Save the map first with:\n"
            f"  ros2 run nav2_map_server map_saver_cli -f "
            f"/home/carver/carver_f1tenth/map/my_map"
        )

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

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params],
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
            "publish_rate": 100.0,
            "base_frame": "basefootprint",
            # AMCL needs a real translation prior on odom→base_link. Loose
            # vx (20.0 m²/s²) basically pinned odom at the origin in xy →
            # particles never tracked the car. Tightened to 0.5 → 1σ ≈ 0.7
            # m/s, so the EKF actually integrates wheel velocity into xy.
            "vx_covariance": 0.5,
        }],
    )

    # --- Nav2 localization stack: map_server + amcl + lifecycle_manager ---
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[amcl_params],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[amcl_params],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[amcl_params],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=["-d", rviz_config],
        output="screen",
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.expanduser(f"~/slam_logs/loc_{stamp}")
    os.makedirs(os.path.dirname(bag_dir), exist_ok=True)
    bag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "-o", bag_dir,
            "/scan", "/imu/data", "/imu_filter",
            "/odometry/filtered", "/tf", "/tf_static",
            "/vesc/state", "/vesc/twist", "/vesc/slip",
            "/map", "/amcl_pose", "/particle_cloud", "/initialpose",
        ],
        output="screen",
    )
    delayed_bag = TimerAction(period=5.0, actions=[bag_record])

    return LaunchDescription([
        sensor_launch,
        tf_base_to_laser,
        ekf_node,
        robot_state_publisher,
        joint_state_publisher,
        imu_filter,
        vesc_velocity,
        map_server,
        amcl,
        lifecycle_manager,
        rviz_node,
        delayed_bag,
    ])
