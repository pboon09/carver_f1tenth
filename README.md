# carver_f1tenth

## Usage

### Launch simulation

Manual drive (default):
```bash
ros2 launch f1tenth_bringup bringup.launch.py
```

Auto drive (Stanley controller):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley
```

Auto drive (Gap Follow controller):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow
```

### Testing Controllers

**Stanley Controller** (Path-following with waypoints):
```bash
# Open track
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley

# With obstacles (requires waypoint planning)
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_easy
```

**Gap Follow Controller** (Reactive obstacle avoidance):
```bash
# Open track (baseline)
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow

# 2 obstacles (Q2 qualifying)
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_easy

# 4 obstacles (H2H race)
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_medium

# Extreme test (15 obstacles)
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_dense
```

### Adjust parameters

Gap Follow speed:
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow max_speed:=3.5
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
