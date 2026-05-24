# carver_f1tenth — SLAM workspace

Dedicated workspace for building 2D occupancy maps of the F1TENTH track using
`slam_toolbox` driven by the on-car RPLidar C1 and BNO055 IMU.

## Packages

| Package | Purpose |
|---|---|
| `f1tenth_slam` | slam_toolbox bringup: params, static TFs, mapping launch, rviz config |

Sensor drivers (`rplidar_ros`, `bno055_usb_stick`) live in `sensor_ws`.
`mapping.launch.py` includes `f1tenth_bringup/sensor.launch.py`, which itself
runs those drivers — so nothing is duplicated here.

## Build

```bash
cd slam_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## Run mapping

First source all three workspaces (sensor drivers + bringup + slam):

```bash
source /opt/ros/jazzy/setup.bash
source ~/carver_f1tenth/sensor_ws/install/setup.bash
source ~/carver_f1tenth/ros2_ws/install/setup.bash
source ~/carver_f1tenth/slam_ws/install/setup.bash
```

Then one launch is enough for SLAM:

```bash
ros2 launch f1tenth_slam mapping.launch.py
```

That brings up the sensors (via `f1tenth_bringup/sensor.launch.py`) plus
the SLAM stack — slam_toolbox, static TFs, rviz. Drive the car around the
track (joystick + actuators from another terminal if needed) and the map
will build live in rviz.

For manual driving alongside mapping, in another terminal:

```bash
ros2 launch f1tenth_bringup joystick.launch.py
```

> ⚠️ Don't run `base.launch.py` *and* `mapping.launch.py` simultaneously —
> they'd both try to open the lidar / IMU. Pick one entry point.

### Save the map

In another terminal:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map
```

Produces `my_map.pgm` and `my_map.yaml`.

## Configuration files

| File | What it controls |
|---|---|
| `f1tenth_slam/config/slam_toolbox_params.yaml` | slam_toolbox tuning (loop closure, scan matching, map resolution) |
| `f1tenth_slam/config/rplidar_params.yaml` | Lidar serial port, baudrate, frame_id, scan mode |
| `f1tenth_slam/config/bno055_params.yaml` | IMU port + calibration constants (currently documentation only — driver hard-codes the values; update both if you recalibrate) |

## TF tree

```
map ── (slam_toolbox) ── odom ── (static, identity) ── base_link ┬── laser
                                                                  └── imu_link
```

The `odom → base_link` static transform is a placeholder. Replace it with a
real wheel-odometry publisher (e.g. VESC telemetry → `/odom`) when one is
available, and remove the `tf_odom_to_base` node from `mapping.launch.py`.
