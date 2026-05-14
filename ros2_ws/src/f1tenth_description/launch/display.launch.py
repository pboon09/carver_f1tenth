#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
import xacro    
    
def generate_launch_description():

    pkg = get_package_share_directory('f1tenth_description')
    rviz_config = os.path.join(pkg, 'config', 'display.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config],
        output='screen')

    path_description = os.path.join(pkg,'urdf','f1tenth_urdf.urdf')
    with open(path_description, 'r') as f:
        robot_desc_xml = f.read()
    
    parameters = [{'robot_description':robot_desc_xml}]
    #parameters.append({'frame_prefix':namespace+'/'})
    robot_state_publisher = Node(package='robot_state_publisher',
                                  executable='robot_state_publisher',
                                  output='screen',
                                  parameters=parameters
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    launch_description = LaunchDescription()

    launch_description.add_action(robot_state_publisher)
    launch_description.add_action(joint_state_publisher_gui)
    launch_description.add_action(rviz)
    
    return launch_description