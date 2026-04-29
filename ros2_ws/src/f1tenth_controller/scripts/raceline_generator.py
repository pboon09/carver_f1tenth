#!/usr/bin/env python3
"""
Raceline Generator for F1TENTH
================================
Pipeline (matches the project proposal):
  Step 1 – Extract centerline from map (contour-based: midpoint of inner/outer walls)
  Step 2 – Generate racing line with chosen lane option:
              centerline : raw track midline
              mincurv    : minimum-curvature path (QP, cuts corners)
              inner/outer: fixed offset toward a boundary
  Step 3 – Compute curvature-based speed profile  v ≤ sqrt(a_lat / κ)
  Step 4 – Save path_v.yaml  {x, y, theta, v} consumed by pure_pursuit.py

Usage:
    python3 raceline_generator.py \
        --map   /path/to/Spielberg_map.yaml \
        --lane  mincurv \
        --vmax  4.0 \
        --alat  4.0 \
        --out   /path/to/path_v.yaml

Dependencies:  numpy, scipy, opencv-python, pyyaml, Pillow
               trajectory-planning-helpers (optional, for mincurv)
               quadprog (optional, for mincurv)
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter1d

try:
    import trajectory_planning_helpers as tph
    _TPH = True
except ImportError:
    _TPH = False


# ── Map loading ────────────────────────────────────────────────────────────────

def load_map(yaml_path: str):
    """Return (gray_image_flipped, resolution, origin_x, origin_y, meta)."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
    res  = float(meta["resolution"])
    ox   = float(meta["origin"][0])
    oy   = float(meta["origin"][1])
    neg  = int(meta.get("negate", 0))

    img = np.array(Image.open(img_path).convert("L"))   # uint8
    if neg:
        img = 255 - img
    img = np.flipud(img)   # row 0 = y_min
    return img, res, ox, oy, meta


def pixels_to_world(pts_rc, res, ox, oy):
    """(row, col) ndarray → (x_m, y_m) world coordinates."""
    rows = pts_rc[:, 0].astype(float)
    cols = pts_rc[:, 1].astype(float)
    return np.column_stack([cols * res + ox, rows * res + oy])


def world_to_pixels(xy_m, res, ox, oy, shape):
    """(x_m, y_m) → (row, col) pixel indices (clipped to image bounds)."""
    cols = ((xy_m[:, 0] - ox) / res).astype(int)
    rows = ((xy_m[:, 1] - oy) / res).astype(int)
    rows = np.clip(rows, 0, shape[0] - 1)
    cols = np.clip(cols, 0, shape[1] - 1)
    return np.column_stack([rows, cols])


# ── Contour-based centerline extraction ───────────────────────────────────────

def _order_contour_pts(pts: np.ndarray) -> np.ndarray:
    """
    Order a set of (x, y) contour points as a closed loop
    using a greedy nearest-neighbour walk.
    """
    pts = pts.tolist()
    path = [pts.pop(0)]
    while pts:
        last = path[-1]
        dists = [(p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 for p in pts]
        i = int(np.argmin(dists))
        path.append(pts.pop(i))
    return np.array(path)


def extract_centerline(map_yaml: str, spacing: float = 0.3,
                        wall_thresh: int = 100):
    """
    Extract the track centerline by:
      1. Thresholding the map image to find wall pixels
      2. Finding the two largest wall contours (outer wall, inner wall)
      3. For each point on one contour, finding the nearest point on the other
      4. Taking the midpoint → centerline
      5. Computing half-track-width from the contour spacing

    Returns (xy [N,2], w_right [N], w_left [N]).
    """
    img, res, ox, oy, meta = load_map(map_yaml)
    occ_thr = float(meta.get("occupied_thresh", 0.65))
    print(f"Map: {img.shape}, res={res} m/px, origin=({ox:.2f},{oy:.2f}), "
          f"occ_thresh={occ_thr}")

    # Wall mask: pixels below occupied threshold (dark = wall)
    wall_mask = (img < int(occ_thr * 255)).astype(np.uint8)
    print(f"  Wall pixels: {wall_mask.sum()}")

    # Find contours with full hierarchy
    contours, hierarchy = cv2.findContours(wall_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        sys.exit("ERROR: fewer than 2 wall contours found — check --map and thresholds")

    # Sort by perimeter (descending) and print info
    c_info = sorted(enumerate(contours),
                    key=lambda ic: cv2.arcLength(ic[1], True), reverse=True)
    for rank, (i, c) in enumerate(c_info[:6]):
        p = cv2.arcLength(c, True) * res
        par = hierarchy[0][i][3]
        print(f"  [{rank}] contour {i}: {len(c)} pts, perimeter={p:.1f} m, parent={par}")

    # The track corridor lies between:
    #   contour rank 1 (inner edge of outer wall)
    #   contour rank 2 (outer edge of inner wall)
    # This is the standard hierarchy for a closed track:
    #   rank0=outer-outer, rank1=outer-inner, rank2=inner-outer, rank3=inner-inner
    if len(c_info) >= 4:
        contours = [c_info[1][1], c_info[2][1]]   # inner-of-outer, outer-of-inner
        print(f"  Using contours ranked 1 and 2 as track boundaries")
    else:
        contours = [c_info[0][1], c_info[1][1]]
        print(f"  Using contours ranked 0 and 1 as track boundaries")

    # Convert contours to (x_m, y_m) world coordinates
    # cv2 contour points are (col, row) in the flipped (flipud) image
    def contour_to_world(c):
        pts_cr = c[:, 0, :]          # shape (N, 2): col, row
        pts_rc = pts_cr[:, ::-1]     # → row, col
        return pixels_to_world(pts_rc, res, ox, oy)

    c0_world = contour_to_world(contours[0])  # outer wall
    c1_world = contour_to_world(contours[1])  # inner wall

    # ── Two-phase centerline extraction ──────────────────────────────────────
    # Phase 1: Arc-length matching (winding-corrected) gives correctly ORDERED
    #          estimate points that are roughly in the corridor center.
    # Phase 2: EDT gradient ascent from those estimates snaps each point to the
    #          TRUE ridge (equidistant from both walls) in ≤30 steps, since the
    #          estimates are already inside the corridor (not on the walls).
    # This avoids: exterior escape (estimates are interior), ordering errors
    # (arc-matching preserves loop order), smoothing oscillations (short ascent
    # gives distinct ridge pixels with no large duplicate clusters).

    from scipy.ndimage import gaussian_filter

    def resample_contour(pts, sp):
        out = [pts[0]]; acc = 0.0
        for i in range(1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[i - 1])
            acc += d
            if acc >= sp:
                out.append(pts[i]); acc = 0.0
        return np.array(out)

    def signed_area(pts):
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    def arc_lengths_closed(pts):
        closed = np.vstack([pts, pts[0]])
        d = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(d)])

    # ── Phase 1: arc-length-matched midpoints as ordered seed estimates ───────
    fine_sp = spacing * 0.5          # finer than output so we don't lose detail
    c0_s = resample_contour(c0_world, fine_sp)
    c1_s = resample_contour(c1_world, fine_sp)

    sa0 = signed_area(c0_s); sa1 = signed_area(c1_s)
    print(f"  Winding: c0={'CCW' if sa0>0 else 'CW'}, c1={'CCW' if sa1>0 else 'CW'}")
    if np.sign(sa0) != np.sign(sa1):
        c1_s = c1_s[::-1]

    # Align c1 start to c0 start
    c1_s = np.roll(c1_s, -int(np.argmin(np.linalg.norm(c1_s - c0_s[0], axis=1))), axis=0)

    # Interpolate both contours at N evenly-spaced arc-length fractions
    N = min(len(c0_s), len(c1_s))
    s0 = arc_lengths_closed(c0_s); s0n = np.append(s0[:-1]/s0[-1], 1.0)
    s1 = arc_lengths_closed(c1_s); s1n = np.append(s1[:-1]/s1[-1], 1.0)
    c0e = np.vstack([c0_s, c0_s[0]]); c1e = np.vstack([c1_s, c1_s[0]])
    t = np.linspace(0, 1, N, endpoint=False)
    c0m = np.column_stack([np.interp(t, s0n, c0e[:,0]), np.interp(t, s0n, c0e[:,1])])
    c1m = np.column_stack([np.interp(t, s1n, c1e[:,0]), np.interp(t, s1n, c1e[:,1])])
    estimates = (c0m + c1m) / 2.0   # midpoints: inside corridor, roughly ordered

    # Subsample estimates to the requested spacing (reduces Phase 2 work)
    estimates = resample_contour(estimates, spacing)
    print(f"  Phase 1: {len(estimates)} ordered estimates")

    # ── Phase 2: EDT gradient ascent to snap each estimate to the true ridge ──
    free_mask = (wall_mask == 0).astype(np.uint8)
    edt_raw = distance_transform_edt(free_mask)
    # Light blur to break ties on flat ridges without creating large plateaus
    edt = gaussian_filter(edt_raw, sigma=0.5)

    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    H, W_px = img.shape
    n_est = len(estimates)
    xy_raw = np.zeros((n_est, 2))
    w_raw  = np.zeros(n_est)

    print(f"  Phase 2: snapping {n_est} estimates to EDT ridge …")
    for i in range(n_est):
        rc = world_to_pixels(estimates[i:i+1], res, ox, oy, img.shape)[0]
        r, c_px = int(rc[0]), int(rc[1])
        # Short ascent: estimate is already inside the corridor,
        # so the ridge is at most half-track-width (~43 px) away.
        for _ in range(100):
            best_val = edt[r, c_px]
            br, bc = r, c_px
            for dr, dc in neighbours:
                nr, nc = r + dr, c_px + dc
                if 0 <= nr < H and 0 <= nc < W_px and edt[nr, nc] > best_val:
                    best_val = edt[nr, nc]; br, bc = nr, nc
            if br == r and bc == c_px:
                break
            r, c_px = br, bc
        xy_raw[i] = pixels_to_world(np.array([[r, c_px]]), res, ox, oy)[0]
        w_raw[i]  = np.clip(edt_raw[r, c_px] * res, 0.1, 5.0)

    xy = xy_raw.copy()
    w  = w_raw.copy()

    # Light smoothing only — short ascent from interior estimates produces
    # much cleaner ridge positions than wall-seeded ascent, so sigma can be small.
    sigma = 2.0
    xy[:, 0] = gaussian_filter1d(xy[:, 0], sigma=sigma, mode="wrap")
    xy[:, 1] = gaussian_filter1d(xy[:, 1], sigma=sigma, mode="wrap")
    w = gaussian_filter1d(w, sigma=sigma, mode="wrap")

    print(f"  Centerline: {len(xy)} waypoints, "
          f"half-width [{w.min():.2f}, {w.max():.2f}] m")
    gap = np.linalg.norm(xy[-1] - xy[0])
    print(f"  Loop closure gap: {gap:.2f} m")

    return xy, w, w


# ── Normal vectors ─────────────────────────────────────────────────────────────

def calc_normals(xy: np.ndarray) -> np.ndarray:
    """Unit left-pointing normal at each waypoint of a closed loop."""
    n = len(xy)
    normals = np.zeros((n, 2))
    for i in range(n):
        fwd = xy[(i + 1) % n] - xy[(i - 1) % n]
        normals[i] = np.array([-fwd[1], fwd[0]])
        norm = np.linalg.norm(normals[i])
        if norm > 1e-9:
            normals[i] /= norm
    return normals


# ── Lane options ───────────────────────────────────────────────────────────────

def lane_centerline(xy, **_):
    return xy.copy()


def lane_offset(xy, normals, w_right, w_left, fraction: float):
    """fraction > 0 → left wall, fraction < 0 → right wall."""
    offsets = fraction * np.minimum(w_right, w_left)
    return xy + normals * offsets[:, np.newaxis]


def lane_mincurv(xy, normals, w_right, w_left, width_opt: float = 0.3):
    """
    Minimum-curvature racing line via QP.
    Falls back to centerline if tph / quadprog are unavailable.
    """
    if not _TPH:
        print("WARNING: trajectory_planning_helpers not installed, "
              "falling back to centerline")
        return xy.copy()

    reftrack = np.column_stack([
        xy,
        np.clip(w_right - width_opt / 2, 0.05, 99),
        np.clip(w_left  - width_opt / 2, 0.05, 99),
    ])
    refpath_cl = np.vstack([xy, xy[0:1]])

    try:
        print("Computing splines for mincurv …")
        coeffs_x, coeffs_y, a_interp, normvec = \
            tph.calc_splines.calc_splines(path=refpath_cl)

        print("Running minimum-curvature QP …")
        alpha_opt, _ = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack,
            normvectors=normvec[:-1],
            A=None,
            kappa_bound=0.0,
            w_veh=width_opt,
            print_debug=True,
            closed=True)

        raceline = xy + normvec[:-1] * alpha_opt[:, np.newaxis]
        print(f"Optimisation done. Max offset: {np.abs(alpha_opt).max():.3f} m")
        return raceline

    except Exception as e:
        print(f"WARNING: mincurv failed ({e}), falling back to centerline")
        return xy.copy()


# ── Curvature & speed profile ──────────────────────────────────────────────────

def calc_curvature(xy: np.ndarray) -> np.ndarray:
    """Menger curvature (3-point) for each waypoint of a closed loop."""
    n = len(xy)
    kappa = np.zeros(n)
    for i in range(n):
        a = xy[(i - 1) % n]
        b = xy[i]
        c = xy[(i + 1) % n]
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        denom = ab * bc * ac
        kappa[i] = 2.0 * cross / denom if denom > 1e-9 else 0.0
    return kappa


def _limit_curvature(xy: np.ndarray, kappa_max: float,
                     max_iters: int = 200) -> np.ndarray:
    """
    Iteratively smooth sections of the path where curvature exceeds kappa_max.
    This ensures the path is physically driveable given the car's steering limit.

    F1TENTH physical limits:
      wheelbase L  ≈ 0.33 m
      max steer δ  ≈ 0.41 rad
      R_min = L / tan(δ) ≈ 0.76 m  →  kappa_max = 1/R_min ≈ 1.32 rad/m
    """
    xy = xy.copy()
    n  = len(xy)
    for iteration in range(max_iters):
        kappa    = calc_curvature(xy)
        bad_mask = kappa > kappa_max
        if not bad_mask.any():
            print(f"  Curvature OK after {iteration} iterations "
                  f"(max κ = {kappa.max():.3f})")
            break
        # Smooth only the offending points and their neighbours
        for i in np.where(bad_mask)[0]:
            for idx in [(i - 1) % n, i, (i + 1) % n]:
                prev_i = (idx - 1) % n
                next_i = (idx + 1) % n
                xy[idx] = (xy[prev_i] + xy[idx] + xy[next_i]) / 3.0
    else:
        kappa = calc_curvature(xy)
        print(f"  Warning: curvature still {kappa.max():.3f} after {max_iters} iters "
              f"({bad_mask.sum()} points above limit)")
    return xy


def calc_speed_profile(xy: np.ndarray, v_max: float,
                        a_lat: float, a_lon: float = 2.0) -> np.ndarray:
    """
    Step A – lateral limit:   v_lat[i] = sqrt(a_lat / max(κ_i, ε))
    Step B – forward pass:    v[i] ≤ sqrt(v[i-1]² + 2*a_lon*ds)  (acceleration)
    Step C – backward pass:   v[i] ≤ sqrt(v[i+1]² + 2*a_lon*ds)  (braking)
    """
    kappa = calc_curvature(xy)
    # Smooth curvature to avoid spikes
    kappa = gaussian_filter1d(kappa, sigma=3, mode="wrap")
    kappa_safe = np.maximum(kappa, 1e-4)

    v = np.minimum(np.sqrt(a_lat / kappa_safe), v_max)

    n = len(xy)
    ds = np.array([np.linalg.norm(xy[(i + 1) % n] - xy[i]) for i in range(n)])
    ds = np.maximum(ds, 1e-6)

    for i in range(1, n):          # forward pass
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_lon * ds[i - 1]))
    for i in range(n - 2, -1, -1): # backward pass
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_lon * ds[i]))

    return v


# ── Heading ────────────────────────────────────────────────────────────────────

def calc_heading(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    theta = np.zeros(n)
    for i in range(n):
        nxt = xy[(i + 1) % n]
        theta[i] = math.atan2(float(nxt[1] - xy[i][1]),
                               float(nxt[0] - xy[i][0]))
    return theta


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a F1TENTH racing line from a map.")
    parser.add_argument("--map",     required=True,
                        help="Path to map .yaml file")
    parser.add_argument("--out",     required=True,
                        help="Output path_v.yaml file")
    parser.add_argument("--lane",    default="mincurv",
                        choices=["centerline", "mincurv", "inner", "outer"],
                        help="Lane / raceline type (default: mincurv)")
    parser.add_argument("--vmax",    type=float, default=4.0,
                        help="Maximum speed m/s (default: 4.0)")
    parser.add_argument("--alat",    type=float, default=4.0,
                        help="Max lateral acceleration m/s² (default: 4.0)")
    parser.add_argument("--alon",    type=float, default=2.0,
                        help="Max longitudinal acceleration m/s² (default: 2.0)")
    parser.add_argument("--spacing", type=float, default=0.3,
                        help="Waypoint spacing m (default: 0.3)")
    parser.add_argument("--width-opt", type=float, default=0.3,
                        help="Vehicle width safety margin for mincurv (default: 0.3)")
    parser.add_argument("--kappa-max", type=float, default=1.0,
                        help="Max path curvature = 1/R_min (default: 1.0 rad/m "
                             "≈ R_min 1.0m, matches F1TENTH steering limit)")
    args = parser.parse_args()

    # ── Step 1: Extract centerline ────────────────────────────────────────────
    print("\n── Step 1: Centerline extraction (contour-based) ──")
    xy_center, w_right, w_left = extract_centerline(
        args.map, spacing=args.spacing)

    # ── Step 2: Generate racing line ──────────────────────────────────────────
    print(f"\n── Step 2: Lane = '{args.lane}' ──")
    normals = calc_normals(xy_center)

    if args.lane == "centerline":
        xy_race = lane_centerline(xy_center)
    elif args.lane == "mincurv":
        xy_race = lane_mincurv(xy_center, normals, w_right, w_left,
                                width_opt=args.width_opt)
    elif args.lane == "inner":
        xy_race = lane_offset(xy_center, normals, w_right, w_left, fraction=-0.5)
    elif args.lane == "outer":
        xy_race = lane_offset(xy_center, normals, w_right, w_left, fraction=+0.5)

    # ── Step 2b: Curvature limiting (respect car's steering ability) ─────────
    # F1TENTH: wheelbase L≈0.33m, max steer δ≈0.41rad → R_min = L/tan(δ) ≈ 0.76m
    # κ_max = 1/R_min ≈ 1.32 rad/m  (use --kappa-max to tune)
    # Any section tighter than this is physically undriveable — smooth it out.
    print(f"\n── Step 2b: Curvature limit (κ_max={args.kappa_max:.2f} rad/m) ──")
    xy_race = _limit_curvature(xy_race, kappa_max=args.kappa_max)

    # ── Step 3: Speed profile ─────────────────────────────────────────────────
    print("\n── Step 3: Speed profile (curvature-based) ──")
    v = calc_speed_profile(xy_race, v_max=args.vmax,
                            a_lat=args.alat, a_lon=args.alon)
    print(f"  v: min={v.min():.2f}  mean={v.mean():.2f}  max={v.max():.2f} m/s")

    theta = calc_heading(xy_race)

    # ── Step 4: Save path_v.yaml ────────────────────────────────────────────────
    print("\n── Step 4: Saving ──")
    waypoints = [
        {"x": round(float(xy_race[i, 0]), 6),
         "y": round(float(xy_race[i, 1]), 6),
         "theta": round(float(theta[i]),  6),
         "v": round(float(v[i]),          4)}
        for i in range(len(xy_race))
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump({"waypoints": waypoints}, f, default_flow_style=False)

    print(f"Saved {len(waypoints)} waypoints → {args.out}")
    print(f"  x: [{xy_race[:,0].min():.1f}, {xy_race[:,0].max():.1f}]  "
          f"y: [{xy_race[:,1].min():.1f}, {xy_race[:,1].max():.1f}]")


if __name__ == "__main__":
    main()
