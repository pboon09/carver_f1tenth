#!/usr/bin/python3

import math
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")

        # Speed
        self.declare_parameter("velocity", 1.5)

        # Lookahead — scales linearly with speed between min and max
        self.declare_parameter("min_lookahead", 0.5)
        self.declare_parameter("max_lookahead", 2.0)
        self.declare_parameter("min_lookahead_speed", 0.0)
        self.declare_parameter("max_lookahead_speed", 7.0)

        # Steering gain (P) — decreases at higher speed to avoid oscillation
        self.declare_parameter("min_gain", 0.4)
        self.declare_parameter("max_gain", 0.7)
        self.declare_parameter("gain_speed_scale", 7.0)

        # Derivative gain for damping
        self.declare_parameter("D", 2.0)

        self.declare_parameter("steering_limit", 24.0)   # degrees
        self.declare_parameter("lookahead_window", 50)   # waypoints to search ahead

        # Read parameters
        self.odom_topic  = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.velocity    = float(self.get_parameter("velocity").value)

        self.min_lookahead       = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead       = float(self.get_parameter("max_lookahead").value)
        self.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)

        self.min_gain        = float(self.get_parameter("min_gain").value)
        self.max_gain        = float(self.get_parameter("max_gain").value)
        self.gain_speed_scale = float(self.get_parameter("gain_speed_scale").value)
        self.D               = float(self.get_parameter("D").value)

        self.steering_limit   = float(self.get_parameter("steering_limit").value)
        self.lookahead_window = int(self.get_parameter("lookahead_window").value)

        # Load waypoints (also sets self.waypoint_velocities)
        self.waypoint_velocities = None
        self.waypoints = self._load_waypoints()
        self.num_pts   = len(self.waypoints)
        self.get_logger().info(f"Loaded {self.num_pts} waypoints")

        # State
        self.current_speed    = 0.0
        self.prev_nearest_idx = None
        self.prev_steer_error = 0.0
        self.prev_d_item      = 0.0
        self.prev_steer       = 0.0
        self.target_point     = None

        # Subscriptions / publications
        self.odom_sub  = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.target_pub = self.create_publisher(Marker, "/pure_pursuit/target", 10)
        self.path_pub   = self.create_publisher(Marker, "/pure_pursuit/path", 10)

        # Publish path every second so RViz always receives it
        self.create_timer(1.0, self._publish_path)

    # ------------------------------------------------------------------
    # Waypoint loading
    # ------------------------------------------------------------------

    def _load_waypoints(self):
        path = str(self.get_parameter("waypoints_path").value)
        if not path:
            pkg_share = get_package_share_directory("f1tenth_controller")
            path = os.path.join(pkg_share, "path", "path_v.yaml")

        self.get_logger().info(f"Loading waypoints from: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        wp_list = data["waypoints"]
        xy = np.array([[wp["x"], wp["y"]] for wp in wp_list])
        # Load per-waypoint velocity if present, otherwise use fixed param
        if "v" in wp_list[0]:
            self.waypoint_velocities = np.array([wp["v"] for wp in wp_list])
            self.get_logger().info("Using per-waypoint speed profile from path_v.yaml")
        else:
            self.waypoint_velocities = None
        return xy

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        pose  = msg.pose.pose
        twist = msg.twist.twist

        curr_x   = pose.position.x
        curr_y   = pose.position.y
        curr_pos = np.array([curr_x, curr_y])
        self.current_speed = twist.linear.x

        q = pose.orientation
        curr_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

        # 1. Find nearest waypoint using local window
        nearest_idx = self._find_nearest(curr_pos)

        # 2. Walk forward along arc to find lookahead point
        L = self._get_lookahead(self.current_speed)
        target_global = self._find_lookahead_point(curr_pos, nearest_idx, L)
        if target_global is None:
            return
        self.target_point = target_global

        # 3. Transform target to vehicle frame
        R_mat = np.array([
            [ np.cos(curr_yaw), np.sin(curr_yaw)],
            [-np.sin(curr_yaw), np.cos(curr_yaw)]
        ])
        local = R_mat @ (target_global - curr_pos)
        target_y = local[1]

        # 4. Pure pursuit steering
        L_actual = max(np.linalg.norm(curr_pos - target_global), 1e-6)
        error    = (2.0 * target_y) / (L_actual ** 2)
        steer    = self._get_steering(self.current_speed, error)
        steer    = 0.3 * steer + 0.7 * self.prev_steer
        self.prev_steer = steer

        # 5. Publish
        # Use per-waypoint speed profile if available, else fixed velocity
        if self.waypoint_velocities is not None:
            target_speed = float(self.waypoint_velocities[nearest_idx])
        else:
            target_speed = self.velocity

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed           = target_speed
        drive_msg.drive.steering_angle  = steer
        self.drive_pub.publish(drive_msg)

        self._draw_marker(self.target_point, self.target_pub, color="yellow")

    # ------------------------------------------------------------------
    # Nearest waypoint (local window to avoid confusion on closed loops)
    # ------------------------------------------------------------------

    def _find_nearest(self, curr_pos):
        if self.prev_nearest_idx is None:
            idx = int(np.argmin(np.linalg.norm(self.waypoints - curr_pos, axis=1)))
        else:
            window_idxs = [(self.prev_nearest_idx + i) % self.num_pts
                           for i in range(self.lookahead_window)]
            dists = np.linalg.norm(self.waypoints[window_idxs] - curr_pos, axis=1)
            idx   = window_idxs[int(np.argmin(dists))]
        self.prev_nearest_idx = idx
        return idx

    # ------------------------------------------------------------------
    # Arc-length lookahead (handles sharp turns correctly)
    # ------------------------------------------------------------------

    def _find_lookahead_point(self, curr_pos, nearest_idx, L):
        arc = 0.0
        idx = nearest_idx
        for _ in range(self.num_pts):
            next_idx = (idx + 1) % self.num_pts
            arc += np.linalg.norm(self.waypoints[next_idx] - self.waypoints[idx])
            idx  = next_idx
            if arc >= L:
                break
        return self.waypoints[idx].copy()

    # ------------------------------------------------------------------
    # Speed-dependent lookahead distance
    # ------------------------------------------------------------------

    def _get_lookahead(self, speed):
        t = (speed - self.min_lookahead_speed) / max(
            self.max_lookahead_speed - self.min_lookahead_speed, 1e-6
        )
        t = np.clip(t, 0.0, 1.0)
        return self.min_lookahead + t * (self.max_lookahead - self.min_lookahead)

    # ------------------------------------------------------------------
    # Speed-dependent PD steering
    # ------------------------------------------------------------------

    def _get_steering(self, speed, error):
        t = speed / max(self.gain_speed_scale, 1e-6)
        P = self.max_gain - np.clip(t, 0.0, 1.0) * (self.max_gain - self.min_gain)

        d_error = error - self.prev_steer_error
        if d_error == 0.0:
            d_error = self.prev_d_item
        else:
            self.prev_d_item      = d_error
            self.prev_steer_error = error

        steer = P * error + self.D * d_error
        return float(np.clip(steer, -np.radians(self.steering_limit),
                                     np.radians(self.steering_limit)))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _publish_path(self):
        from geometry_msgs.msg import Point
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns     = "pure_pursuit_path"
        marker.id     = 0
        marker.type   = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 0.8
        marker.color.g = 1.0
        marker.pose.orientation.w = 1.0

        for wp in np.vstack([self.waypoints, self.waypoints[:1]]):  # close loop
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            marker.points.append(p)

        self.path_pub.publish(marker)

    def _draw_marker(self, position, publisher, color="red"):
        if position is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id     = 0
        marker.ns     = "pure_pursuit_target"
        marker.type   = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.a = 1.0
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "yellow":
            marker.color.r = 1.0
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime.nanosec = int(1e8)
        publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
