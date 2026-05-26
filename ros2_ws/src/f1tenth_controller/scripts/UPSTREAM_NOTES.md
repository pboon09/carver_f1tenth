# Why upstream `stanley_avoidance` doesn't work for our setup

Reference: `f1tenth_ws/src/stanley_avoidance/` (the original we ported from).

## TL;DR

Upstream was designed for fast racing in wide hallways (`e7_floor5_biggest`):
3–6 m/s with 1.5–2.5 m dynamic lookahead through corridors with ≥1 m of
path-to-wall clearance and corner radii in the meters. Our setup is
0.5 m/s with 0.3–0.8 m lookahead through a 1 m wide corridor with
0.20–0.40 m clearance and R=0.81 m corners. The geometry and dynamics
mismatch breaks five specific assumptions in the upstream algorithm.

## Side-by-side conditions

| Property                | Upstream         | Ours              |
|-------------------------|------------------|-------------------|
| Track                   | wide hallways    | tight corridor    |
| Path-to-wall clearance  | ≥1 m             | 0.20–0.40 m       |
| Cruise speed            | 3–6 m/s          | 0.5 m/s           |
| Dynamic lookahead L     | 1.5–2.5 m        | 0.3–0.8 m         |
| Tightest corner radius  | >5 m             | 0.81 m            |
| Chord-cut at corners    | ~6 cm            | ~30 cm at 1 m chord |

## 5 places upstream specifically breaks

### 1. Single car→goal chord is too short on our setup

Upstream collision check: `_check_collision(current_pos, goal_pos, margin)`.
`goal_pos` lives at the *dynamic* lookahead which scales with speed. At
their speeds the chord spans 2.5 m forward — crosses obstacles on the
path. At our speeds the chord is only 0.3 m — obstacles 1+ m ahead are
never on the chord. By the time the chord finds the obstacle, the car
is 0.3 m from impact.

**Fix in our port:** Replaced single chord with **path-walking** detector
(`_path_blocked`) that walks each 0.3 m waypoint segment up to
`obstacle_detect_lookahead` meters. Per-segment chord-cut is ~1 cm —
no false positives — and detection range is decoupled from drive lookahead.

### 2. Chord-cut destroys correctness at our corners

At R=0.81 m with a 1.5 m chord, the chord deviates ~30 cm from the actual
arc. Our path has walls 20 cm away. Chord intersects walls → false
obstacle detection every time we approach a corner.

**Fix:** Same as #1 — path-walking has tiny per-segment chord deviation.

### 3. Pure-pursuit gain doesn't translate to low speed

`angle = K_p × 2y / L²`.

| | L (m) | y (m) | K_p | angle |
|---|-------|-------|-----|-------|
| Upstream | 2.5 | 0.25 | 0.8 | 0.064 rad = 3.7° (gentle) |
| Ours (with upstream K_p) | 0.5 | 0.25 | 0.8 | 1.6 rad = 92° → saturated |

`L²` in the denominator means the gain is 25× more aggressive at our
shorter lookahead.

**Fix in our port:** `K_p_obstacle = 0.1–0.3` (was 0.8) + a separate
`avoidance_steer_limit = 0.175 rad` (~10°) as a hard cap that's tighter
than the global `steering_limit`.

### 4. Velocity table is inverted at our scale

Upstream avoidance velocity table:
```python
velocity_min = 1.0
velocity_max = 2.0    # < cruise = 4.0
```
So during avoidance the car SLOWS to 1–2 m/s. Avoidance == slow down.

Our defaults (initially):
```python
velocity_min = 0.5
velocity_max = 0.7    # > cruise = 0.5
```
So during avoidance the car SPEEDS UP. Opposite intent.

**Fix:** Cap velocity to cruise during avoidance:
```python
velocity = min(velocity_max, cruise)
```

### 5. Shift candidate sequence biased to one side

Upstream:
```python
shifts = [i * (-1 if i % 2 else 1) for i in range(1, 21)]
# = [-1, +2, -3, +4, -5, +6, ...]
```
Odd magnitudes always go negative (right), even always positive (left).
Tries `-1` (right) first.

At high speed this barely matters — the arc to the shifted goal is so
shallow that you just need any clear chord. At low speed, picking the
wrong direction first means the car commits hard the wrong way in one
tick — straight into the obstacle.

**Fix in our port:**
1. Detect which side the obstacle is on
2. Build shift list that ONLY tries the safe side: `[+1, +2, +3, ...]`
3. Lock the side for the duration of the maneuver (release after
   N consecutive clear ticks)

## What we KEEP from upstream

These are sound primitives that work regardless of speed/track:

- `drive_to_target_stanley` — the stanley controller itself (great tracking)
- `drive_to_target` — pure-pursuit formula (works with proper gain)
- `_check_collision` / `_traverse_grid` — Bresenham with lateral margin
- `_check_area` — area check around a goal cell
- `_convolve_grid` — 2×2 dilation
- 3-tier fallback (shift → midpoint → loose check)
- Grid in car frame
- Direction-flip on startup (we ported this idea + use yaml theta)

## Catalogue of adaptations (each marked `[ADAPT-N]` in the code)

| # | Adaptation | Type | Reason |
|---|------------|------|--------|
| 1 | `wheelbase` 0.33 → 0.30 m | bug fix | URDF says 0.300; upstream default was wrong |
| 2 | `steering_limit` in radians (0.4189) instead of degrees (25) | bug fix | mixer expects radians; degree-clip then `np.radians()` was unintuitive |
| 3 | Startup path-direction auto-flip | new feature | upstream stanley_avoidance has none; ported from upstream pure_pursuit |
| 4 | Stanley `path_heading` from yaml `theta` | structural | spacing ≈ wheelbase makes upstream's front/rear-axle derivation unreliable |
| 5 | Path-walking obstacle detection (replaces single chord) | structural | tight corners cause severe chord-cut; per-segment chord-cut is ~1 cm |
| 6 | Side detection from actual obstacle cell, relative to path | structural | centroid pollutes with walls on narrow corridor |
| 7 | Velocity during avoidance clamped to `min(table, cruise)` | bug fix | upstream defaults expected `velocity_max < cruise`; ours had inversion |
| 8 | Separate `avoidance_steer_limit` (tighter than `steering_limit`) | structural | pure-pursuit saturates at low L; need tighter cap during avoidance |
| 9 | E-stop uses `lidar_to_nose` offset | new feature | upstream e-stop didn't exist; nose-relative threshold needed |
| 10 | Safe (0, 0) on SIGINT/SIGTERM/SIGTSTP | operational | prevents VESC holding last command on exit |
| 11 | Adaptive `shift_base` (5 cells past obstacle) | structural | upstream's shift_base = goal_pos at dynamic_L gives angles too small at our scale |
| 12 | Diagnostic logging | development | for tuning and writeup |
| 14 | Shift perpendicular to path direction (not car heading) | structural | upstream shifts in car-frame lateral; this is wrong when the path curves — the shifted target ends up at an oblique angle to the actual path. Using the blocked segment's tangent gives a shifted target on a path parallel to the original raceline |
| 17 | Robot-centric side selection: probe both perpendicular sides of the blocked segment, pick the larger gap, center the offset in it | structural | side-from-path tangent (ADAPT-6) was unreliable when `_path_blocked` returns prev/next cells in distance order rather than path-index order — the (di,dj) vector could flip, mis-classifying the obstacle side and steering the car straight into the obstacle. The largest-gap probe ignores path direction entirely: gap measurement is symmetric. Offset = safe_gap / 2 self-caps to the available corridor and won't push past walls. Computed once at lock |

**Hysteresis on avoidance state.** path_blocked detection flickers
True/False between scan ticks when the obstacle sits right at the
MARGIN edge — causing the car to oscillate between avoidance and
stanley. Mitigation: once the avoidance lock is set (first detection),
stay in avoidance mode until `obstacle_clear_threshold` consecutive
clear ticks (default 15 ≈ 1.5 s at 10 Hz). During held-but-not-blocked
ticks, the last computed avoidance target is reused.

**Notes on the avoidance mechanism.** Upstream's distinguishing feature
is the *chord-shift* fallback: try lateral offsets of the lookahead
waypoint until one is collision-free, then drive pure-pursuit to that
shifted target. This is **path-biased** — the car tries to keep
approximately the same forward direction, just nudged sideways.

A Follow-the-Gap approach was prototyped (largest-gap angular center,
disparity-corrected) but **rejected** because it loses the path-bias
property — it picks whichever direction has the most space, regardless
of where the raceline goes. The repo has a separate `gap_follow.py` for
pure gap-following. This file keeps the chord-shift mechanism that
identifies it as `stanley_avoidance`.

## Open issues (in progress)

- Direction lock can flip during sharp swerves because the path appears
  to rotate in car frame as the car yaws. Current mitigation: relock if
  measurement disagrees with lock by side. A more robust fix would store
  the obstacle's WORLD position once and lock on that.

- Even with all adaptations, very-close-and-on-centerline obstacles (< 0.4 m
  from the nose when first detected) can't be physically avoided at any
  steering — e-stop is the only safety. To detect earlier, place the
  obstacle ≥ 1.5 m ahead of where the car starts.

- Tight-corner avoidance is largely impossible: a 0.20 m wall clearance
  has no room to swerve around a 0.20 m ball. The car will stop instead.

## URDF / hardware notes

- URDF uses SolidWorks axes (X=lateral, Y=longitudinal, Z=up) — NOT ROS
  convention. Wheelbase = 0.300 m derived from wheel-joint positions.
- Lidar is mounted at base_link's xy (≈0 offset). The gym xacro's
  `laser_distance_from_base_link = 0.275` does NOT apply to our real car.
- Lidar to front bumper ≈ 0.264 m (URDF) / 0.28 m (measured).
- E-stop uses `lidar_to_nose` to translate lidar range to nose-relative
  distance.
