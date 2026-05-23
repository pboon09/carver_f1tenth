import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    rplidar_launch_dir = os.path.join(
        get_package_share_directory("rplidar_ros"), "launch"
    )
    bringup_lib_dir = os.path.join(
        get_package_share_directory("f1tenth_bringup"),
        "..", "..", "lib", "f1tenth_bringup",
    )
    lidar_reset_path = os.path.abspath(
        os.path.join(bringup_lib_dir, "lidar_reset.py")
    )

    # Pre-flight reset of the RPLidar C1 over its serial port so a previous
    # SIGKILL'd session doesn't leave it stuck in health=2. Runs synchronously
    # for ~1 s, then the actual driver launches.
    lidar_reset = ExecuteProcess(
        cmd=["python3", lidar_reset_path, "/dev/rplidar", "460800"],
        output="screen",
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_launch_dir, "rplidar_c1_launch.py")
        ),
    )
    # Give the lidar's MCU time to finish its software reset before the driver
    # opens the port. ~1.2 s is enough headroom over the 0.9 s sleep in
    # lidar_reset.py.
    delayed_rplidar = TimerAction(period=1.2, actions=[rplidar_launch])

    bno_stick = Node(
        package="bno055_usb_stick",
        executable="bno055_usb_stick_node_script.py",
        output="screen",
    )
    # Delay the BNO055 so it doesn't fight the RPLidar for the USB bus during
    # init — the C1 needs ~2.5 s to negotiate, and concurrent IMU register
    # traffic can starve it into SL_RESULT_OPERATION_TIMEOUT.
    delayed_bno_stick = TimerAction(period=4.0, actions=[bno_stick])

    return LaunchDescription([
        lidar_reset,
        delayed_rplidar,
        delayed_bno_stick,
    ])
