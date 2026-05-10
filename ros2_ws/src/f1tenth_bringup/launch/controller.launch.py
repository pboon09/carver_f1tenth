import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    _pkg = get_package_share_directory("f1tenth_controller")

    mode_arg = DeclareLaunchArgument(
        "mode", default_value="manual",
        description="Controller mode: manual or auto"
    )

    algorithm_arg = DeclareLaunchArgument(
        "algorithm", default_value="stanley",
        description="Control algorithm: stanley, gap_follow, pure_pursuit, or lattice"
    )

    # ── Single place to switch the raceline ──────────────────────────────────
    raceline_arg = DeclareLaunchArgument(
        "raceline", default_value=os.path.join(_pkg, "path", "path_v_mincurv.yaml"),
        description="Absolute path to the waypoints yaml used by all controllers and viz"
    )

    teleop_node = Node(
        package="f1tenth_joy",
        executable="teleop.py",
        name="teleop_node",
        output="screen",
        prefix="xterm -e",
        parameters=[
            {"drive_topic": "/drive"},
            {"max_speed": 2.0},
            {"max_steer": 0.4189},
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'manual'"])
        ),
    )

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
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto' and '", LaunchConfiguration("algorithm"), "' == 'stanley'"])
        ),
    )

    gap_follow_node = Node(
        package="f1tenth_controller",
        executable="gap_follow.py",
        name="gap_follow_node",
        output="screen",
        parameters=[
            {"min_speed": 0.5},
            {"max_speed": 4.5},
            {"max_steering_angle": 0.35},
            {"steering_smoothing": 0.4},
            {"car_width": 0.2},
            {"min_distance_threshold": 2.5},
            {"obstacle_inflated": 30},
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto' and '", LaunchConfiguration("algorithm"), "' == 'gap_follow'"])
        ),
    )

    purepursuit_node = Node(
        package="f1tenth_controller",
        executable="pure_pursuit.py",
        name="pure_pursuit_node",
        output="screen",
        parameters=[
            {"waypoints_path": LaunchConfiguration("raceline")},
            {"velocity": 5.0},
            {"min_lookahead": 0.6},
            {"max_lookahead": 2.5},
            {"max_gain": 0.8},
            {"D": 2.0},
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto' and '", LaunchConfiguration("algorithm"), "' == 'pure_pursuit'"])
        ),
    )

    lattice_node = Node(
        package="f1tenth_controller",
        executable="lattice_planner.py",
        name="lattice_planner_node",
        output="screen",
        parameters=[
            {"waypoints_path":          LaunchConfiguration("raceline")},
            {"max_offset":              1.0},
            {"safety_radius":           0.35},
            {"track_half_width":        0.70},
            {"plan_horizon":            5.0},
            {"num_offsets":             17},
            {"w_deviation":             1.0},
            {"w_continuity":            1.5},
            {"min_lookahead":           0.8},
            {"max_lookahead":           1.8},
            {"speed_gain":              0.40},
            {"steer_gain":              0.8},
            {"steer_limit":             0.41},
            {"imminent_dist":           0.40},
            {"avoidance_speed_scale":   0.85},
            {"replan_hold_ticks":       20},
            {"clear_hold_ticks":        15},
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto' and '", LaunchConfiguration("algorithm"), "' == 'lattice'"])
        ),
    )

    viz_node = Node(
        package="f1tenth_viz",
        executable="viz.py",
        name="viz_node",
        output="screen",
        parameters=[
            {"waypoints_path": LaunchConfiguration("raceline")},
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto'"])
        ),
    )

    return LaunchDescription([
        mode_arg,
        algorithm_arg,
        raceline_arg,
        teleop_node,
        stanley_node,
        gap_follow_node,
        purepursuit_node,
        lattice_node,
        viz_node,
    ])
