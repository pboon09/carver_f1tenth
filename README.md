# carver_f1tenth

## Usage

### Quick Start

**Manual drive (joystick control):**
```bash
ros2 launch f1tenth_bringup bringup.launch.py
```

**Autonomous drive - choose an algorithm:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=pure_pursuit
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice
```

**With obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_easy
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_dense
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_dense
```

### Testing Controllers

**1. Stanley Controller** (Path-following with grid-based obstacle avoidance):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_dense
```

**2. Gap Follow Controller** (Reactive obstacle avoidance):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_dense
```

**3. Pure Pursuit Controller** (Simple waypoint-following, baseline):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=pure_pursuit
```

**4. Lattice Planner** (Advanced Pure Pursuit with planning-based obstacle avoidance):
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_dense
```

### Algorithm Comparison

| Algorithm | Type | Obstacle Handling | Best For | Notes |
|-----------|------|-------------------|----------|-------|
| **Stanley** | Path-following | Grid-based detection | Waypoint-heavy tracks | Uses occupancy grid for avoidance |
| **Gap Follow** | Reactive | Reactive gaps | Open/sparse obstacles | Purely reactive, no planning |
| **Pure Pursuit** | Waypoint-following | None | Baseline comparison | Simple path tracking, no obstacle avoidance |
| **Lattice** | Planning-based | Multi-candidate planning | Dense obstacles | Advanced: plans alternative paths, uses pure pursuit for tracking |

**Pure Pursuit Hierarchy:**
- `pure_pursuit.py` — Basic implementation (no obstacles)
- `lattice_planner.py` — Advanced pure pursuit with planning (handles obstacles via lateral path candidates)

### Complete Command Reference

#### Stanley Controller (Path-following + Grid-based Avoidance)

**Without obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley
```

**With obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_easy      # 2 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_medium    # 4 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_hard      # 8 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=stanley map:=Spielberg_map_dense     # 15 obstacles
```

#### Gap Follow Controller (Reactive Obstacle Avoidance)

**Without obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow
```

**With obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_easy      # 2 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_medium    # 4 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_hard      # 8 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_chicane   # 6 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow map:=Spielberg_map_dense     # 15 obstacles (extreme)
```

**Adjust speed:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow max_speed:=3.5
```

#### Pure Pursuit Controller (Simple Waypoint Following - Baseline)

**Without obstacles (baseline for comparison):**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=pure_pursuit
```

**With obstacles (will crash - no avoidance):**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=pure_pursuit map:=Spielberg_map_easy
```

#### Lattice Planner (Advanced Pure Pursuit with Planning-based Obstacle Avoidance)

**Without obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice
```

**With obstacles:**
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_easy      # 2 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_medium    # 4 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_hard      # 8 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_chicane   # 6 obstacles
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=lattice map:=Spielberg_map_dense     # 15 obstacles (ultimate test)
```

### Adjust parameters

Gap Follow speed:
```bash
ros2 launch f1tenth_bringup bringup.launch.py mode:=auto algorithm:=gap_follow max_speed:=3.5
```

### Build & Setup Commands

**Build the workspace:**
```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-ignore f110_gym f1tenth_gym_ros
source install/setup.bash
```

**Generate racing lines for pure pursuit / lattice:**
```bash
python3 -m f1tenth_controller.raceline_generator \
    --map ros2_ws/src/f1tenth_gym_ros/maps/Spielberg_map.yaml \
    --out ros2_ws/src/f1tenth_controller/path/path_v.yaml \
    --lane centerline --vmax 4.0
```

**Available raceline options:**
```bash
--lane centerline   # Track centerline (default)
--lane mincurv      # Minimum-curvature racing line (fastest)
--lane inner        # Offset toward inner wall
--lane outer        # Offset toward outer wall
--vmax 4.0          # Maximum velocity (m/s)
--alat 4.0          # Max lateral acceleration (m/s²)
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
