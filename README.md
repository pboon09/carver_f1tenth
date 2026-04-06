# carver_f1tenth

## Usage

### Launch simulation

Manual drive (default):
```bash
ros2 launch f1tenth_bringup bringup.launch.py
```

Auto drive:
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto
```

With obstacle map:
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto map:=Spielberg_map_easy
```

### Available maps

| Map | Obstacles | Description |
|-----|-----------|-------------|
| `Spielberg_map` | 0 | Base track (default) |
| `Spielberg_map_easy` | 2 | Two obstacles spread apart |
| `Spielberg_map_medium` | 4 | Four corners of track |
| `Spielberg_map_hard` | 8 | All sections covered |
| `Spielberg_map_chicane` | 6 | Spread around full loop |
| `Spielberg_map_dense` | 15 | Every section of track |

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Drive command (publish to control car) |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan |
| `/ego_racecar/odom` | `nav_msgs/Odometry` | Odometry |
| `/map` | `nav_msgs/OccupancyGrid` | Map from map_server |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Reset car pose |

Note: The gym simulator does not provide IMU data, so you can use orientation information from `/ego_racecar/odom`.
