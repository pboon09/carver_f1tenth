# carver_f1tenth

## Usage

### Launch simulation

Obstacle-free:
```bash
ros2 launch f1tenth_bringup bringup.launch.py
```

With obstacles:
```bash
ros2 launch f1tenth_bringup bringup.launch.py map:=levine_easy
```

### Available maps

| Map | Obstacles | Description |
|-----|-----------|-------------|
| `levine` | 0 | Base track (default) |
| `levine_easy` | 1 | Single obstacle |
| `levine_medium` | 3 | Spread across corridors |
| `levine_hard` | 5 | Both corridors + turns |
| `levine_chicane` | 4 | Alternating left-right |
| `levine_dense` | 10 | All four corridors |

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Drive command (publish to control car) |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan |
| `/ego_racecar/odom` | `nav_msgs/Odometry` | Odometry |
| `/map` | `nav_msgs/OccupancyGrid` | Map from map_server |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Reset car pose |

Note: The gym simulator does not provide IMU data, so you can use orientation information from `/ego_racecar/odom`.
