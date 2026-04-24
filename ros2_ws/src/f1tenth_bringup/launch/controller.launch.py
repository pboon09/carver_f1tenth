from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode", default_value="manual",
        description="Controller mode: manual or auto"
    )

    algorithm_arg = DeclareLaunchArgument(
        "algorithm", default_value="stanley",
        description="Control algorithm: stanley or gap_follow"
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

    viz_node = Node(
        package="f1tenth_viz",
        executable="stanley_viz.py",
        name="stanley_viz_node",
        output="screen",
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mode"), "' == 'auto'"])
        ),
    )

    return LaunchDescription([
        mode_arg,
        algorithm_arg,
        teleop_node,
        stanley_node,
        gap_follow_node,
        viz_node,
    ])
