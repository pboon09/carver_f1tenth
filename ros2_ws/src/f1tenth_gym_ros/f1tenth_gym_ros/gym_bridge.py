# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

import numpy as np
from transforms3d import euler
from f110_gym.envs import F110Env

class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')

        self.declare_parameter('ego_namespace')
        self.declare_parameter('ego_odom_topic')
        self.declare_parameter('ego_opp_odom_topic')
        self.declare_parameter('ego_scan_topic')
        self.declare_parameter('ego_drive_topic')
        self.declare_parameter('opp_namespace')
        self.declare_parameter('opp_odom_topic')
        self.declare_parameter('opp_ego_odom_topic')
        self.declare_parameter('opp_scan_topic')
        self.declare_parameter('opp_drive_topic')
        self.declare_parameter('scan_distance_to_base_link')
        self.declare_parameter('scan_fov')
        self.declare_parameter('scan_beams')
        self.declare_parameter('map_path')
        self.declare_parameter('map_img_ext')
        self.declare_parameter('num_agent')
        self.declare_parameter('sx')
        self.declare_parameter('sy')
        self.declare_parameter('stheta')
        self.declare_parameter('sx1')
        self.declare_parameter('sy1')
        self.declare_parameter('stheta1')
        self.declare_parameter('kb_teleop')

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # env backend
        self.scan_max_range = 40.0
        self.env = F110Env(
                            map=self.get_parameter('map_path').value,
                            map_ext=self.get_parameter('map_img_ext').value,
                            num_agents=num_agents,
                            lidar_dist=self.get_parameter("scan_distance_to_base_link").value,
                            num_beams=self.get_parameter('scan_beams').value,
                            fov=self.get_parameter('scan_fov').value,
                            max_range=self.scan_max_range,
                            )

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_prev_speed = [0.0, 0.0, 0.0]  # for IMU accel differentiation

        # BNO055 noise parameters (datasheet Table 0-2, fusion / NDOF mode)
        # Accelerometer: noise density 150 µg/√Hz, BW ≈ 63 Hz in NDOF
        self._accel_noise_std = 150e-6 * 9.80665 * (63.0 ** 0.5)   # ≈ 0.01168 m/s²
        # Gyroscope: 0.014 °/s/√Hz, BW = 47 Hz → σ = 1.675e-3 rad/s
        self._gyro_noise_std  = 0.014 * (np.pi / 180.0) * (47.0 ** 0.5)
        # Orientation (fusion): heading ±2.5°, roll/pitch ±1° (treated as 1-sigma)
        self._orient_std_yaw  = 2.5  * (np.pi / 180.0)   # rad
        self._orient_std_rp   = 1.0  * (np.pi / 180.0)   # rad

        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_odom_topic').value
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        
        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])

            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_odom_topic').value
            opp_drive_topic = self.get_parameter('opp_drive_topic').value

            ego_opp_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_ego_odom_topic').value
        else:
            self.has_opp = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        # sim physical step timer
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)
        # odom + tf publishing timer (250 Hz)
        self.timer = self.create_timer(0.004, self.timer_callback)
        # LiDAR timer: RPLidar S3 scan rate = 10 Hz
        self.scan_timer = self.create_timer(0.1, self.scan_timer_callback)
        # IMU timer: BNO055 NDOF fusion output rate = 100 Hz
        self.imu_timer = self.create_timer(0.01, self.imu_timer_callback)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.ego_imu_pub  = self.create_publisher(Imu, self.ego_namespace + '/imu', 10)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            10)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)


    def drive_callback(self, drive_msg):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta], opp_pose]))
        else:
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta]]))
        self._update_sim_state()

    def opp_reset_callback(self, pose_msg):
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            self.obs, _ , self.done, _ = self.env.reset(np.array([list(self.ego_pose), [rx, ry, rtheta]]))
            self._update_sim_state()

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.ego_drive_published and not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        self._update_sim_state()

    def scan_timer_callback(self):
        """Publish LiDAR scans at 10 Hz — RPLidar S3 scan rate."""
        ts = self.get_clock().now().to_msg()

        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.05
        scan.range_max = 40.
        ego_ranges = np.array(self.ego_scan)
        ego_ranges[ego_ranges >= self.scan_max_range - 1.0] = float('inf')
        scan.ranges = ego_ranges.tolist()
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_ranges = np.array(self.opp_scan)
            opp_ranges[opp_ranges >= self.scan_max_range - 1.0] = float('inf')
            opp_scan.ranges = opp_ranges.tolist()
            self.opp_scan_pub.publish(opp_scan)

    def timer_callback(self):
        ts = self.get_clock().now().to_msg()

        # pub tf
        self._publish_odom(ts)
        self._publish_transforms(ts)
        self._publish_laser_transforms(ts)
        self._publish_wheel_transforms(ts)

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0] = self.obs['poses_x'][1]
            self.opp_pose[1] = self.obs['poses_y'][1]
            self.opp_pose[2] = self.obs['poses_theta'][1]
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

        

    def imu_timer_callback(self):
        ts = self.get_clock().now().to_msg()
        self._publish_imu(ts)

    def _publish_imu(self, ts):
        """Publish simulated IMU data with BNO055 noise characteristics (100 Hz)."""
        dt = 0.01  # BNO055 NDOF output period (100 Hz)
        theta = self.ego_pose[2]

        # --- Orientation (yaw-only for planar car, fusion mode accuracy) ---
        noisy_yaw = theta + np.random.normal(0.0, self._orient_std_yaw)
        imu = Imu()
        imu.header.stamp = ts
        imu.header.frame_id = self.ego_namespace + '/imu'
        imu.orientation.x = 0.0
        imu.orientation.y = 0.0
        imu.orientation.z = float(np.sin(noisy_yaw / 2.0))
        imu.orientation.w = float(np.cos(noisy_yaw / 2.0))
        ov = self._orient_std_rp ** 2
        yv = self._orient_std_yaw ** 2
        imu.orientation_covariance = [ov, 0., 0.,   # roll
                                      0., ov, 0.,   # pitch
                                      0., 0., yv]   # yaw

        # --- Angular velocity (only z from sim; x/y are zero + noise) ---
        gv = self._gyro_noise_std ** 2
        imu.angular_velocity.x = float(np.random.normal(0.0, self._gyro_noise_std))
        imu.angular_velocity.y = float(np.random.normal(0.0, self._gyro_noise_std))
        imu.angular_velocity.z = float(self.ego_speed[2] + np.random.normal(0.0, self._gyro_noise_std))
        imu.angular_velocity_covariance = [gv, 0., 0.,
                                           0., gv, 0.,
                                           0., 0., gv]

        # --- Linear acceleration (differentiate world-frame velocity → body frame) ---
        ax_w = (self.ego_speed[0] - self.ego_prev_speed[0]) / dt
        ay_w = (self.ego_speed[1] - self.ego_prev_speed[1]) / dt
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        ax_b =  cos_t * ax_w + sin_t * ay_w
        ay_b = -sin_t * ax_w + cos_t * ay_w
        av = self._accel_noise_std ** 2
        imu.linear_acceleration.x = float(ax_b + np.random.normal(0.0, self._accel_noise_std))
        imu.linear_acceleration.y = float(ay_b + np.random.normal(0.0, self._accel_noise_std))
        imu.linear_acceleration.z = float(9.80665  + np.random.normal(0.0, self._accel_noise_std))  # gravity
        imu.linear_acceleration_covariance = [av, 0., 0.,
                                              0., av, 0.,
                                              0., 0., av]

        self.ego_prev_speed = list(self.ego_speed)
        self.ego_imu_pub.publish(imu)

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'odom'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'odom'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = euler.euler2quat(0.0, 0.0, self.opp_pose[2], axes='sxyz')
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        # ego_scan_ts.transform.translation.z = 0.04+0.1+0.025
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        self.br.sendTransform(ego_scan_ts)

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = self.scan_distance_to_base_link
            opp_scan_ts.transform.rotation.w = 1.
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            self.br.sendTransform(opp_scan_ts)

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()