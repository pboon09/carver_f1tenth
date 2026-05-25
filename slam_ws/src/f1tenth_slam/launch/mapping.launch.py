#!/usr/bin/env python3
"""
F1TENTH SLAM mapping bringup.

Brings up sensors + the SLAM layer in one launch. The sensor drivers
themselves live in `sensor_ws` (rplidar_ros, bno055_usb_stick); we don't
copy them here. Instead we include `f1tenth_bringup`'s `sensor.launch.py`
from the main ros2_ws, which is the canonical entry point for /scan and
/imu/data on the car.

So slam_ws stays minimal — just the `f1tenth_slam` package — and the
SLAM stack reuses the same sensor launch every other component uses.

Pipeline (single terminal):
    ros2 launch f1tenth_slam mapping.launch.py

If you want manual driving alongside SLAM, run joystick + actuators
separately from the main workspace:
    ros2 launch f1tenth_bringup joystick.launch.py

Brings up:
  - sensor.launch.py   → /scan, /imu/data (rplidar + bno055)
  - slam_toolbox       → /map (subscribes to /scan, publishes map → odom TF)
  - Static transforms  → base_link → laser, base_link → imu_link
                       → odom → base_link (identity placeholder until wheel
                         odometry from the VESC is available)
  - rviz2

Save the map with:
    ros2 run nav2_map_server map_saver_cli -f ~/my_map
"""

import datetime
import os

import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState


def generate_launch_description():
    pkg = get_package_share_directory("f1tenth_slam")
    slam_params  = os.path.join(pkg, "config", "slam_toolbox_params.yaml")
    rviz_config  = os.path.join(pkg, "rviz", "slam.rviz")

    # Load URDF so robot_state_publisher can broadcast the model's TF chain
    # (base_link → lidar / imu / wheels / steering hubs) and so RViz can draw
    # the car around the lidar scan / map. Frame names from the URDF do NOT
    # collide with the static TFs below — URDF uses `lidar` for the visual
    # link, the rplidar publishes /scan in frame `laser`, and slam_toolbox
    # consumes the static `base_link → laser` TF.
    urdf_path = os.path.join(
        get_package_share_directory("f1tenth_urdf"),
        "urdf", "simplify_fullremake.urdf",
    )
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    # Sensors come from f1tenth_bringup (which itself depends on rplidar_ros
    # and bno055_usb_stick from sensor_ws). Reusing this launch instead of
    # re-declaring the drivers keeps slam_ws minimal.
    sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("f1tenth_bringup"),
                "launch", "sensor.launch.py",
            )
        ),
    )

    # base_link → laser  (planar / 2D — yaw only)
    # The 180° rotation is now handled inside the rplidar driver via
    # flip_x_axis:=true (see rplidar_c1_launch.py), so the scan already
    # has the arrow direction at +x. This TF is identity in rotation;
    # only z lifts the laser frame to the actual mount height.
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

    # base_link → imu_link is now broadcast by `imu_calibrator` itself —
    # it measures the actual mount tilt from the BNO055's quaternion during
    # the 5 s idle window and bakes that into the static TF. Mount POSITION
    # (xyz) is passed in below; only the rotation is measured live.

    # odom → basefootprint via robot_localization EKF, fed by the BNO055
    # gyro Z (yaw rate only — absolute yaw and accel produce too much drift
    # on this unit, see ekf_imu.yaml comments). EKF gives slam_toolbox a
    # non-zero motion delta each scan so it actually runs scan-match.
    ekf_params = os.path.join(pkg, "config", "ekf_imu.yaml")
    tf_odom_to_base = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params],
    )

    # `async_slam_toolbox_node` is a managed lifecycle node — it boots into
    # `unconfigured` and stays there until somebody transitions it through
    # `configure` → `activate`. Without that it never subscribes to /scan, never
    # loads parameters, never publishes the map→odom TF. The two events below
    # do the dance automatically so the launch is fire-and-forget.
    slam_toolbox_node = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        parameters=[slam_params],
    )

    # 1. as soon as the node is running, push it to `configure`
    configure_slam = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    # 2. when configure finishes (state goes configuring → inactive), activate
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

    # cmd_to_joint_state turns joystick / autopilot commands (/vesc/cmd RPM
    # and /steering_angle rad) into JointState messages so the URDF model
    # in RViz animates as we drive — wheels spin, Ackermann hubs rotate.
    # Robot pose on the map still comes from SLAM via map → odom → base_link;
    # this only animates the joint angles. With no joystick running, every
    # joint sits at 0 (model is static).
    joint_state_publisher = Node(
        package="f1tenth_joy",
        executable="cmd_to_joint_state.py",
        name="cmd_to_joint_state",
        output="screen",
        parameters=[{
            # match the physical drivetrain so the URDF wheel-spin in RViz
            # looks right at speed: (51/12) × 7 = 29.75
            "gear_ratio": 29.75,
        }],
    )

    # imu_calibrator samples /imu/data for the first 5 s after the IMU comes
    # up (robot must be IDLE), measures the actual noise variance per channel,
    # and republishes /imu/data_calibrated with those variances baked into
    # the message covariance fields. The EKF subscribes to that topic.
    # imu_filter — single canonical IMU-processing node. Eats /imu/data,
    # spits out /imu_filter (which the EKF and anything else should read).
    # See imu_filter.py for the full filter pipeline.
    imu_filter = Node(
        package="f1tenth_joy",
        executable="imu_filter.py",
        name="imu_filter",
        output="screen",
        parameters=[{
            "calibration_seconds": 5.0,
            # Moving-average filter on each gyro + accel axis.
            "gyro_median_window": 1,    # median stage OFF — pure MA
            "gyro_ma_window": 8,
            # BLOCK mode: accumulate N samples, output their mean, reset.
            # Each block is independent — no spike carries over to the next.
            # Output is a 100/N Hz staircase (12.5 Hz at N=8).
            "gyro_ma_block_mode": True,
        }],
    )

    # vesc_velocity bridges /vesc/state RPM → /vesc/twist (TwistWithCov) at
    # 100 Hz with a high vx covariance, so the EKF gets a real translation
    # source that's automatically down-weighted during wheel slip. Also
    # publishes /vesc/slip = wheel_velocity − EKF velocity for diagnostics.
    vesc_velocity = Node(
        package="f1tenth_vesc_driver",
        executable="vesc_velocity.py",
        name="vesc_velocity",
        output="screen",
        parameters=[{
            "wheel_radius": 0.0594,
            # Physical drivetrain: (51/12) × 7 = 29.75 motor revs per wheel rev
            # (spur:pinion 51:12  ×  differential/final 7:1).
            "gear_ratio": 29.75,
            "publish_rate": 100.0,
            "base_frame": "basefootprint",
            # 0.5 m²/s² → 1σ ≈ 0.7 m/s. Tight enough that the EKF integrates
            # wheel velocity into odom xy at 100 Hz — gives smooth dead-
            # reckoning between scan-match corrections (lidar 10 Hz). Loose
            # enough to absorb wheel slip without locking in a wrong velocity.
            "vx_covariance": 0.5,
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
            "rate": 50.0,
            "publish_rate": 20.0,
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
        sensor_launch,
        tf_base_to_laser,
        tf_odom_to_base,
        robot_state_publisher,
        joint_state_publisher,
        imu_filter,                 # /imu/data → /imu_filter + base_link→imu_link TF
        vesc_velocity,
        slam_toolbox_node,
        activate_slam,    # register handler BEFORE we emit the configure event
        configure_slam,
        trajectory_publisher,
        rviz_node,
    ])
