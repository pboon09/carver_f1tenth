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
  Step 4 – Save path.yaml  {x, y, theta, v} consumed by pure_pursuit.py

Usage:
    python3 raceline_generator.py \
        --map   /path/to/Spielberg_map.yaml \
        --lane  mincurv \
        --vmax  4.0 \
        --alat  4.0 \
        --out   /path/to/path.yaml

Dependencies:  numpy, scipy, opencv-python, pyyaml, Pillow
               trajectory-planning-helpers (optional, for mincurv)
               quadprog (optional, for mincurv)
"""

import argparse
import math
import os
import sys
import traceback

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

def extract_centerline(map_yaml: str, spacing: float = 0.3,
                        wall_thresh: int = 100,
                        contour_ids: list = None,
                        skip_outer: int = 1):
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

    # Print a sorted-by-perimeter summary with depth in the hierarchy so the
    # user can pick contour IDs manually via --contour-ids if the auto-pick
    # is wrong (common on noisy slam_toolbox maps).
    h = hierarchy[0]

    def depth(i: int) -> int:
        d = 0
        cur = int(h[i][3])
        while cur != -1:
            d += 1
            cur = int(h[cur][3])
        return d

    c_info = sorted(enumerate(contours),
                    key=lambda ic: cv2.arcLength(ic[1], True), reverse=True)
    print(f"  Contours found ({len(contours)}, top 8 by perimeter):")
    for rank, (i, c) in enumerate(c_info[:8]):
        p = cv2.arcLength(c, True) * res
        par = int(h[i][3])
        print(f"    [rank {rank}] id={i}  depth={depth(i)}  pts={len(c)}  "
              f"perimeter={p:.1f} m  parent={par}")

    if contour_ids is not None:
        # Explicit override: user supplied --contour-ids OUTER,INNER.
        try:
            outer_id, inner_id = contour_ids
            contours = [contours[outer_id], contours[inner_id]]
            print(f"  Using user-specified contour ids: "
                  f"outer={outer_id}, inner={inner_id}")
        except (IndexError, ValueError, TypeError):
            sys.exit(f"ERROR: --contour-ids {contour_ids} invalid for "
                     f"{len(contours)} contours")
    else:
        # Hierarchy-aware default: SLAM maps usually have an outermost noisy
        # hull (the entire scanned region). The actual track is one or more
        # depth levels in. Pick the largest contour at depth = skip_outer,
        # and its largest descendant — that's the track outer+inner walls.
        by_depth: dict = {}
        for i, c in enumerate(contours):
            by_depth.setdefault(depth(i), []).append((i, c))
        for d in by_depth:
            by_depth[d].sort(key=lambda ic: cv2.arcLength(ic[1], True),
                             reverse=True)

        outer_depth = skip_outer
        inner_depth = skip_outer + 1
        if outer_depth in by_depth and inner_depth in by_depth:
            outer_idx, outer_c = by_depth[outer_depth][0]
            inner_idx, inner_c = by_depth[inner_depth][0]
            contours = [outer_c, inner_c]
            print(f"  Auto-picked (skip_outer={skip_outer}): "
                  f"outer=id{outer_idx} (depth {outer_depth}), "
                  f"inner=id{inner_idx} (depth {inner_depth})")
        else:
            # Fallback: no nested structure — use legacy 2-largest heuristic.
            print(f"  WARNING: no nested contours at depth {outer_depth}/"
                  f"{inner_depth}; falling back to two-largest heuristic. "
                  f"Pass --contour-ids OUTER,INNER to override.")
            if len(c_info) >= 4:
                contours = [c_info[1][1], c_info[2][1]]
            else:
                contours = [c_info[0][1], c_info[1][1]]

    def contour_to_world(c):
        pts_cr = c[:, 0, :]
        pts_rc = pts_cr[:, ::-1]
        return pixels_to_world(pts_rc, res, ox, oy)

    c0_world = contour_to_world(contours[0])
    c1_world = contour_to_world(contours[1])

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

    # ── Centerline via medial axis of the corridor mask ─────────────────────
    # Build a binary mask of the corridor (between the two walls), then take
    # its medial axis — the locus of points equidistant from the two walls.
    # This is the true centerline and is robust to wall irregularity (SLAM
    # noise, jagged inner-island outlines, etc.), because it operates on the
    # filled corridor region rather than on individual wall outlines.
    from skimage.morphology import medial_axis

    H, W_px = img.shape
    # world_to_pixels returns (N,2) as (row,col). cv2.fillPoly wants (col,row).
    outer_rc = world_to_pixels(c0_world, res, ox, oy, img.shape).astype(np.int32)
    inner_rc = world_to_pixels(c1_world, res, ox, oy, img.shape).astype(np.int32)
    outer_pix_xy = outer_rc[:, ::-1]
    inner_pix_xy = inner_rc[:, ::-1]

    outer_fill = np.zeros((H, W_px), dtype=np.uint8)
    inner_fill = np.zeros((H, W_px), dtype=np.uint8)
    cv2.fillPoly(outer_fill, [outer_pix_xy], 1)
    cv2.fillPoly(inner_fill, [inner_pix_xy], 1)
    # Corridor = inside outer wall, outside inner island, AND not a wall pixel
    # itself — so every original wall pixel is an edge of the corridor mask.
    corridor_mask = (outer_fill.astype(bool) & ~inner_fill.astype(bool)
                     & ~wall_mask.astype(bool))
    print(f"  Corridor pixels: {int(corridor_mask.sum())} "
          f"({int(corridor_mask.sum()) * res * res:.1f} m²)")

    skeleton, dist = medial_axis(corridor_mask, return_distance=True)

    # Prune spur branches: iteratively delete endpoints (skeleton pixels with
    # ≤1 skeleton neighbor) until only the closed loop remains.
    sk = skeleton.copy()
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _neighbor_count(sk_arr, r, c):
        n = 0
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W_px and sk_arr[rr, cc]:
                n += 1
        return n

    pruned = 0
    while True:
        rows, cols = np.where(sk)
        to_remove = [(int(r), int(c)) for r, c in zip(rows, cols)
                     if _neighbor_count(sk, int(r), int(c)) <= 1]
        if not to_remove:
            break
        for r, c in to_remove:
            sk[r, c] = False
        pruned += len(to_remove)

    # Small wall fragments inside the corridor (slam noise islands) create
    # their own little medial-axis loops next to the main one. The walker
    # only follows the loop containing its start pixel, so keep only the
    # LARGEST connected component of the pruned skeleton — that's the
    # one true loop around the inner island.
    sk_u8 = sk.astype(np.uint8)
    n_sk, sk_labels = cv2.connectedComponents(sk_u8, connectivity=8)
    if n_sk >= 2:
        sk_sizes = np.bincount(sk_labels.flatten())
        sk_sizes[0] = 0
        largest_lbl = int(np.argmax(sk_sizes))
        kept = int(sk_sizes[largest_lbl])
        sk = (sk_labels == largest_lbl)
        if n_sk - 1 > 1:
            print(f"  Skeleton: pruned {pruned} spurs, dropped {n_sk - 2} "
                  f"satellite loops (wall noise islands); kept main loop "
                  f"with {kept} pixels")
        else:
            print(f"  Skeleton: pruned {pruned} spurs; "
                  f"{kept} pixels remain in the closed loop")
    else:
        print(f"  Skeleton: pruned {pruned} spurs; "
              f"{int(sk.sum())} pixels remain in the closed loop")

    rows, cols = np.where(sk)
    if len(rows) < 4:
        sys.exit("ERROR: medial-axis loop too small — check the corridor mask. "
                 "Try a different --contour-ids pair.")

    # The skeleton may still contain T-junctions where a small parasitic
    # loop (from wall noise) attaches to the main loop. A greedy walker
    # picks one branch and dead-ends. Use networkx to enumerate cycles in
    # the skeleton graph and pick the LONGEST one — that's the main loop
    # around the inner island.
    import networkx as nx
    G = nx.Graph()
    pixel_list = [(int(r), int(c)) for r, c in zip(rows, cols)]
    G.add_nodes_from(pixel_list)
    for r, c in pixel_list:
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if (0 <= rr < H and 0 <= cc < W_px and sk[rr, cc]
                    and (rr, cc) > (r, c)):
                G.add_edge((r, c), (rr, cc))

    basis = nx.cycle_basis(G)
    if not basis:
        sys.exit("ERROR: skeleton has no closed cycle — corridor topology bad.")
    # Pick longest cycle by node count (≈ perimeter, since all pixels are
    # adjacent unit-spaced).
    ordered_pix = max(basis, key=len)
    print(f"  Skeleton graph: {len(G.nodes)} nodes, {len(G.edges)} edges, "
          f"{len(basis)} cycle(s) in basis; main loop = {len(ordered_pix)} px")

    # World coordinates + half-width from the medial-axis distance.
    rc_arr = np.array(ordered_pix, dtype=float)
    centerline_world = pixels_to_world(rc_arr, res, ox, oy)
    half_widths = np.array([dist[r, c] * res for r, c in ordered_pix])

    # Resample at the target spacing along the loop.
    closed = np.vstack([centerline_world, centerline_world[:1]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(s[-1])
    n_target = max(int(round(total / spacing)), 12)
    s_targets = np.linspace(0.0, total, n_target, endpoint=False)
    xs = np.interp(s_targets, s, np.append(centerline_world[:, 0],
                                           centerline_world[0, 0]))
    ys = np.interp(s_targets, s, np.append(centerline_world[:, 1],
                                           centerline_world[0, 1]))
    xy = np.column_stack([xs, ys])
    hw_closed = np.append(half_widths, half_widths[0])
    w = np.interp(s_targets, s, hw_closed)
    print(f"  Resampled to {len(xy)} waypoints at spacing={spacing} m "
          f"(perimeter={total:.1f} m)")

    # Perpendicular-balance refinement: medial axis = locus of max-inscribed
    # circle, which on asymmetric corridors sits closer to the side with the
    # bigger "open" wall (further from a bump on the other side). That's
    # geometrically the "max clearance" path but it looks off-center to the
    # eye. Iteratively nudge each waypoint along its perpendicular so the
    # ray-cast distances to both walls are equal — gives a visually centered
    # path with gap_L ≈ gap_R, while still living inside the corridor.
    def _ray_distance(p, direction, max_steps=80):
        for k in range(1, max_steps + 1):
            q = p + direction * (k * res)
            col = int((q[0] - ox) / res)
            row = int((q[1] - oy) / res)
            if (not (0 <= row < H and 0 <= col < W_px)
                    or not corridor_mask[row, col]):
                return k * res
        return max_steps * res

    n = len(xy)
    step_cap = 8 * res   # cap per-iteration move to avoid overshoot in narrows
    for _it in range(5):
        new_xy = xy.copy()
        moved = 0.0
        for i in range(n):
            tangent = xy[(i + 1) % n] - xy[(i - 1) % n]
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                continue
            tangent /= t_norm
            normal = np.array([-tangent[1], tangent[0]])
            d_plus  = _ray_distance(xy[i], normal)
            d_minus = _ray_distance(xy[i], -normal)
            offset = 0.5 * (d_plus - d_minus)
            offset = max(-step_cap, min(step_cap, offset))
            new_xy[i] = xy[i] + normal * offset
            moved += abs(offset)
        xy = new_xy
        if moved / n < 0.01:   # converged: average move < 1 cm/waypoint
            break
    print(f"  Perp-balanced in {_it + 1} iters; mean |offset| now ≈ "
          f"{moved / n * 1000:.1f} mm/waypoint")

    sigma = 2.0
    xy[:, 0] = gaussian_filter1d(xy[:, 0], sigma=sigma, mode="wrap")
    xy[:, 1] = gaussian_filter1d(xy[:, 1], sigma=sigma, mode="wrap")
    w = gaussian_filter1d(w, sigma=sigma, mode="wrap")
    w = np.clip(w, 0.1, 5.0)

    print(f"  Centerline: {len(xy)} waypoints, "
          f"half-width [{w.min():.2f}, {w.max():.2f}] m")
    gap = np.linalg.norm(xy[-1] - xy[0])
    print(f"  Loop closure gap: {gap:.2f} m")

    # EDT for the Phase 3 ray-cast hard-cap below (kept from previous algo).
    free_mask = (wall_mask == 0).astype(np.uint8)
    edt_raw = distance_transform_edt(free_mask)
    w_raw = np.array([
        edt_raw[int((y - oy) / res), int((x - ox) / res)] * res
        for x, y in xy
    ])

    # ── Phase 3: ray-cast actual left/right widths along normals ────────────
    # Walks each perpendicular ray until it hits a wall pixel. No EDT cap —
    # the previous version capped both rays at the nearest-wall distance,
    # which on asymmetric corridors silently clipped the FAR wall to look
    # equal to the NEAR wall. Now the blue corridor envelope reaches the
    # actual wall on each side independently.
    normals_tmp = np.zeros((len(xy), 2))
    for i in range(len(xy)):
        fwd = xy[(i + 1) % len(xy)] - xy[(i - 1) % len(xy)]
        n_vec = np.array([-fwd[1], fwd[0]])
        norm = np.linalg.norm(n_vec)
        normals_tmp[i] = n_vec / norm if norm > 1e-9 else n_vec

    step_m    = res
    max_steps = int(4.0 / step_m)   # safety: never search further than 4 m
    H_px, W_px2 = wall_mask.shape
    w_left  = np.full(len(xy), 4.0)
    w_right = np.full(len(xy), 4.0)

    for i in range(len(xy)):
        x0, y0 = float(xy[i, 0]), float(xy[i, 1])
        nx, ny = float(normals_tmp[i, 0]), float(normals_tmp[i, 1])
        for k in range(1, max_steps + 1):
            col = int((x0 + k * step_m * nx - ox) / res)
            row = int((y0 + k * step_m * ny - oy) / res)
            if (not (0 <= row < H_px and 0 <= col < W_px2)
                    or wall_mask[row, col]):
                w_left[i] = k * step_m
                break
        for k in range(1, max_steps + 1):
            col = int((x0 - k * step_m * nx - ox) / res)
            row = int((y0 - k * step_m * ny - oy) / res)
            if (not (0 <= row < H_px and 0 <= col < W_px2)
                    or wall_mask[row, col]):
                w_right[i] = k * step_m
                break

    w_left  = gaussian_filter1d(w_left,  sigma=2.0, mode="wrap")
    w_right = gaussian_filter1d(w_right, sigma=2.0, mode="wrap")
    print(f"  Left wall:  [{w_left.min():.2f}, {w_left.max():.2f}] m")
    print(f"  Right wall: [{w_right.min():.2f}, {w_right.max():.2f}] m")

    return xy, w_right, w_left


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


def lane_mincurv(xy: np.ndarray, normals: np.ndarray,
                 w_right: np.ndarray, w_left: np.ndarray,
                 width_opt: float = 0.3,
                 wall_margin: float = 0.4,
                 corner_margin: float = 0.7,
                 kappa_start: float = 0.3,
                 kappa_corner: float = 0.6,
                 max_offset: float = 0.5,
                 smooth_buffer: float = 0.05,
                 kappa_bound: float = 0.0):
    """
    Minimum-curvature racing line via QP (trajectory_planning_helpers).

    Physical safety is enforced INSIDE the QP via per-waypoint wall margins
    that depend on local curvature — there is no post-hoc clamping needed.

    wall_margin    — base clearance (m) car-edge-to-wall on straights.
    corner_margin  — clearance (m) at tight corners (curvature ≥ kappa_corner).
                     Linearly interpolated between kappa_start and kappa_corner.
    smooth_buffer  — extra margin (m) added to absorb downstream Gaussian-smooth
                     drift, so the *final* path still respects wall_margin /
                     corner_margin after smoothing. Default 0.05 m.
    width_opt      — vehicle total width (m); tph subtracts width_opt/2 per side.
    """
    if not _TPH:
        print("WARNING: trajectory_planning_helpers not installed, "
              "falling back to centerline")
        return xy.copy()

    N = len(xy)

    # ── Adaptive margin & blend ───────────────────────────────────────────────
    # Two-threshold design:
    #   κ < kappa_start  → t = 0.0  → pure mincurv, base wall_margin
    #   κ > kappa_corner → t = 1.0  → centerline, full corner_margin
    #   between          → linear interpolation
    # Sliding-window MAX spreads the effect to the corner entry/exit too.
    from scipy.ndimage import maximum_filter1d
    kappa_cl = calc_curvature(xy)
    kappa_cl = gaussian_filter1d(kappa_cl, sigma=3, mode="wrap")
    # window=20 waypoints at spacing=0.3m covers ±3 m before/after apex
    kappa_spread = maximum_filter1d(kappa_cl, size=20, mode="wrap")
    span = max(kappa_corner - kappa_start, 1e-6)
    t_curve = np.clip((kappa_spread - kappa_start) / span, 0.0, 1.0)
    # Add smooth_buffer so the final path (after Gaussian smoothing) still
    # respects the requested clearance — no need for any post-hoc clamp.
    adaptive_margin = wall_margin + (corner_margin - wall_margin) * t_curve + smooth_buffer
    corner_pts = (t_curve > 0.05).sum()
    print(f"  CL κ: max={kappa_cl.max():.3f} rad/m  "
          f"blend range=[{kappa_start:.2f}, {kappa_corner:.2f}] rad/m")
    print(f"  QP wall margin (clearance + {smooth_buffer:.2f}m smooth-buffer): "
          f"[{adaptive_margin.min():.3f}, {adaptive_margin.max():.3f}] m  "
          f"(corner region: {corner_pts} / {N} pts)")

    # Shrink per-waypoint widths before handing to tph.
    # tph additionally subtracts width_opt/2 for the vehicle half-width.
    wr = np.clip(w_right - adaptive_margin, 0.10, 99.0)
    wl = np.clip(w_left  - adaptive_margin, 0.10, 99.0)

    # reftrack: [x, y, w_right, w_left]  (N, 4)
    reftrack = np.column_stack([xy, wr, wl])

    # Closed path for spline fitting: append the first point
    path_cl = np.vstack([xy, xy[0:1]])   # (N+1, 2)

    # ── FIX 1: compute element lengths (arc-length of each segment) ──────────
    # tph.calc_splines uses these for correct parameterisation.
    # Without them it assumes unit spacing → wrong curvature → zero QP solution.
    el_lengths_cl = np.linalg.norm(np.diff(path_cl, axis=0), axis=1)  # (N,)

    try:
        print("  Computing splines …")
        # calc_splines returns:
        #   coeffs_x  (N, 4)
        #   coeffs_y  (N, 4)
        #   M         (N, N) — used internally for curvature
        #   normvec_normalized  (N+1, 2) — one extra row for the closed loop
        coeffs_x, coeffs_y, M, normvec_cl = tph.calc_splines.calc_splines(
            path=path_cl,
            el_lengths=el_lengths_cl,
            psi_s=None,   # free heading at start
            psi_e=None,   # free heading at end
            use_dist_scaling=True,
        )

        # calc_splines returns exactly N normals for an N-segment closed path.
        normvec = normvec_cl   # (N, 2)

        # Sanity check
        assert normvec.shape[0] == N, \
            f"normvec row mismatch: {normvec.shape[0]} vs {N}"
        assert coeffs_x.shape == (N, 4), \
            f"coeffs_x shape mismatch: {coeffs_x.shape} vs ({N}, 4)"

        # ── FIX 3: kappa_bound should reflect physical steering limit ─────────
        # 0.0 = no curvature constraint → QP ignores sharp corners.
        # A reasonable value is 1/R_min ≈ 1.32 rad/m for F1TENTH.
        # Passed in from the caller (--kappa-max CLI arg).
        print(f"  Running minimum-curvature QP (κ_bound={kappa_bound:.3f}) …")
        alpha_opt, _ = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack,
            normvectors=normvec,
            A=M,            # FIX 4: pass the spline matrix, NOT None
            kappa_bound=kappa_bound,
            w_veh=width_opt,
            print_debug=True,
            closed=True,
        )

        max_off = np.abs(alpha_opt).max()
        mean_off = np.abs(alpha_opt).mean()
        print(f"  QP done. offset max={max_off:.3f} m  mean={mean_off:.3f} m")

        if max_off < 1e-4:
            print("  WARNING: all offsets ≈ 0 — mincurv produced no improvement.")

        # Blend alpha toward 0 (centerline) at tight corners.
        # t_curve = 0 on straights (keep QP offset), 1 at corners (force centerline).
        # max_offset additionally hard-caps the straight-section deviation.
        if max_offset > 0:
            alpha_opt = np.clip(alpha_opt, -max_offset, max_offset)
        blend = 1.0 - t_curve   # 1.0 on straights, 0.0 at tight corners
        alpha_opt = alpha_opt * blend
        print(f"  After corner blend: offset max={np.abs(alpha_opt).max():.3f} m  "
              f"mean={np.abs(alpha_opt).mean():.3f} m")

        raceline = xy + normvec * alpha_opt[:, np.newaxis]
        return raceline

    except Exception:
        # ── FIX 5: print the real traceback so you can debug ─────────────────
        print("ERROR: mincurv optimisation failed with the following traceback:")
        traceback.print_exc()
        print("Falling back to centerline.")
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

    F1TENTH physical limits:
      wheelbase L  ≈ 0.33 m
      max steer δ  ≈ 0.41 rad
      R_min = L / tan(δ) ≈ 0.76 m  →  kappa_max = 1/R_min ≈ 1.32 rad/m

    If max_iters runs out without converging (e.g. the input has a corner
    too tight to fix by triple-point averaging), the *input* path is
    returned with a warning — repeated averaging on a non-convergent path
    eventually collapses the loop toward its centroid, which destroys the
    centerline. Better to drive the unfixed path cautiously than a tight
    O-ring around nothing.
    """
    original = xy.copy()
    xy = xy.copy()
    n  = len(xy)
    for iteration in range(max_iters):
        kappa    = calc_curvature(xy)
        bad_mask = kappa > kappa_max
        if not bad_mask.any():
            print(f"  Curvature OK after {iteration} iterations "
                  f"(max κ = {kappa.max():.3f})")
            return xy
        for i in np.where(bad_mask)[0]:
            for idx in [(i - 1) % n, i, (i + 1) % n]:
                prev_i = (idx - 1) % n
                next_i = (idx + 1) % n
                xy[idx] = (xy[prev_i] + xy[idx] + xy[next_i]) / 3.0
    kappa_now = calc_curvature(xy)
    kappa_orig = calc_curvature(original)
    print(f"  Warning: curvature still {kappa_now.max():.3f} after {max_iters} "
          f"iters (input had max κ = {kappa_orig.max():.3f}). Reverting to "
          f"input path — raise --kappa-max or pass --no-curvature-limit if you "
          f"intend to drive a corner this tight slowly.")
    return original


def calc_speed_profile(xy: np.ndarray, v_max: float,
                        a_lat: float, a_lon: float = 2.0,
                        v_min: float = 0.0) -> np.ndarray:
    """
    Step A – lateral limit:   v_lat[i] = sqrt(a_lat / max(κ_i, ε))
    Step B – forward pass:    v[i] ≤ sqrt(v[i-1]² + 2*a_lon*ds)  (acceleration)
    Step C – backward pass:   v[i] ≤ sqrt(v[i+1]² + 2*a_lon*ds)  (braking)
    Step D – floor at v_min   (so even hairpins don't drop below crawl speed).
    """
    kappa = calc_curvature(xy)
    kappa = gaussian_filter1d(kappa, sigma=3, mode="wrap")
    kappa_safe = np.maximum(kappa, 1e-4)

    v = np.minimum(np.sqrt(a_lat / kappa_safe), v_max)

    n = len(xy)
    ds = np.array([np.linalg.norm(xy[(i + 1) % n] - xy[i]) for i in range(n)])
    ds = np.maximum(ds, 1e-6)

    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_lon * ds[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_lon * ds[i]))

    if v_min > 0.0:
        v = np.maximum(v, v_min)

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


# ── PNG Visualisation ─────────────────────────────────────────────────────────

def save_png(map_yaml: str,
             xy_center: np.ndarray,
             xy_race: np.ndarray,
             v: np.ndarray,
             w_right: np.ndarray,
             w_left: np.ndarray,
             lane: str,
             out_png: str) -> None:
    """
    Render a diagnostic PNG showing:
      • the map (greyscale, flipped to match world coords)
      • track half-width corridor shaded around the centerline
      • centerline in white dashes
      • raceline coloured by speed (blue=slow → red=fast)
      • direction arrows every ~20 waypoints
      • colour-bar legend for speed
      • title with lane name and key stats
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    img, res, ox, oy, meta = load_map(map_yaml)
    H, W = img.shape

    # World extents of the image
    x_min = ox
    x_max = ox + W * res
    y_min = oy
    y_max = oy + H * res

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                              gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]

    # ── Background map ────────────────────────────────────────────────────────
    ax.imshow(img, cmap="gray", origin="lower",
              extent=[x_min, x_max, y_min, y_max],
              vmin=0, vmax=255, alpha=0.55)

    # ── Track corridor (half-width envelope around centerline) ────────────────
    normals = calc_normals(xy_center)
    # Use the true ray-cast distances independently on each side, so the
    # blue corridor envelope follows the real wall (not a symmetric
    # min-of-both, which underplots wherever the corridor isn't centered
    # to start with).
    left_bd  = xy_center + normals * w_left[:, np.newaxis]
    right_bd = xy_center - normals * w_right[:, np.newaxis]

    # Fill corridor as a closed polygon (left boundary + reversed right boundary)
    corridor_x = np.concatenate([left_bd[:, 0],  right_bd[::-1, 0], [left_bd[0, 0]]])
    corridor_y = np.concatenate([left_bd[:, 1],  right_bd[::-1, 1], [left_bd[0, 1]]])
    ax.fill(corridor_x, corridor_y, color="cyan", alpha=0.08, zorder=1)
    ax.plot(np.append(left_bd[:, 0],  left_bd[0, 0]),
            np.append(left_bd[:, 1],  left_bd[0, 1]),
            color="cyan", lw=0.6, alpha=0.4, zorder=2)
    ax.plot(np.append(right_bd[:, 0], right_bd[0, 0]),
            np.append(right_bd[:, 1], right_bd[0, 1]),
            color="cyan", lw=0.6, alpha=0.4, zorder=2)

    # ── Centerline (white dashed) ─────────────────────────────────────────────
    cx = np.append(xy_center[:, 0], xy_center[0, 0])
    cy = np.append(xy_center[:, 1], xy_center[0, 1])
    ax.plot(cx, cy, color="white", lw=1.2, ls="--", alpha=0.6,
            zorder=3, label="Centerline")

    # ── Raceline coloured by speed ────────────────────────────────────────────
    v_min, v_max_val = v.min(), v.max()
    norm  = Normalize(vmin=v_min, vmax=v_max_val)
    cmap  = cm.get_cmap("RdYlGn")   # green=fast, red=slow

    # Build line segments: each segment connects waypoint i → i+1
    N = len(xy_race)
    pts   = np.vstack([np.append(xy_race[:, 0], xy_race[0, 0]),
                       np.append(xy_race[:, 1], xy_race[0, 1])]).T
    segs  = np.stack([pts[:-1], pts[1:]], axis=1)
    v_seg = np.append(v, v[0])
    v_mid = (v_seg[:-1] + v_seg[1:]) / 2.0

    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.5, zorder=5)
    lc.set_array(v_mid)
    ax.add_collection(lc)

    # ── Direction arrows ──────────────────────────────────────────────────────
    step = max(1, N // 25)
    for i in range(0, N, step):
        nxt = (i + 1) % N
        dx  = xy_race[nxt, 0] - xy_race[i, 0]
        dy  = xy_race[nxt, 1] - xy_race[i, 1]
        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            continue
        ax.annotate("",
                    xy=(xy_race[i, 0] + dx / mag * 0.15,
                        xy_race[i, 1] + dy / mag * 0.15),
                    xytext=(xy_race[i, 0], xy_race[i, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=0.8, mutation_scale=8),
                    zorder=6)

    # ── Start marker ─────────────────────────────────────────────────────────
    ax.plot(xy_race[0, 0], xy_race[0, 1], "w*", ms=12, zorder=7, label="Start")

    # Offset comparison: how far is the raceline from the centerline on average?
    offsets = np.linalg.norm(xy_race - xy_center, axis=1)
    mean_off = offsets.mean()
    max_off  = offsets.max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    title = (f"F1TENTH Raceline  |  lane={lane}\n"
             f"N={N} pts   v∈[{v_min:.2f}, {v_max_val:.2f}] m/s   "
             f"mean_v={v.mean():.2f} m/s\n"
             f"offset from CL: mean={mean_off:.3f} m  max={max_off:.3f} m"
             + ("  ← if max≈0 mincurv produced no shift!" if max_off < 0.02 and lane=="mincurv" else ""))
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#222", edgecolor="#555", labelcolor="white")

    # ── Colour bar ────────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Speed (m/s)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # ── Right panel: curvature + speed profiles ───────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    kappa = calc_curvature(xy_race)
    kappa_sm = gaussian_filter1d(kappa, sigma=3, mode="wrap")
    s_arr = np.cumsum(np.linalg.norm(np.diff(
        np.vstack([xy_race, xy_race[0:1]]), axis=0), axis=1))
    s_arr = np.insert(s_arr, 0, 0)[:-1]   # one value per waypoint

    ax2_twin = ax2.twinx()

    ax2.plot(s_arr, kappa_sm, color="#ff6b6b", lw=1.5, label="κ (rad/m)")
    ax2.axhline(1.32, color="#ff6b6b", ls=":", lw=0.8, alpha=0.5,
                label="κ_max=1.32")
    ax2_twin.plot(s_arr, v, color="#4ecdc4", lw=1.5, label="v (m/s)")

    ax2.set_xlabel("Arc length (m)", color="white")
    ax2.set_ylabel("Curvature κ (rad/m)", color="#ff6b6b")
    ax2_twin.set_ylabel("Speed v (m/s)", color="#4ecdc4")
    ax2.set_title("Curvature & Speed Profile", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2_twin.tick_params(colors="#4ecdc4")
    ax2.yaxis.label.set_color("#ff6b6b")
    ax2.tick_params(axis="y", colors="#ff6b6b")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
               facecolor="#222", edgecolor="#555", labelcolor="white",
               loc="upper right")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved PNG  → {out_png}")
    if max_off < 0.02 and lane == "mincurv":
        print("  ⚠  WARNING: max offset from centerline is < 2 cm — "
              "mincurv may have silently returned zero offsets.")
        print("     Check the traceback above for QP errors, or try --lane centerline "
              "to confirm the centerline itself looks correct.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a F1TENTH racing line from a map.")
    parser.add_argument("--map",     required=True,
                        help="Path to map .yaml file")
    parser.add_argument("--out",     required=True,
                        help="Output path.yaml file")
    parser.add_argument("--lane",    default="mincurv",
                        choices=["centerline", "mincurv", "inner", "outer"],
                        help="Lane / raceline type (default: mincurv)")
    parser.add_argument("--vmin",    type=float, default=0.0,
                        help="Minimum speed floor m/s (default: 0 = disabled). "
                             "Speeds below this are clamped up — useful when "
                             "tight corners would otherwise produce sub-crawl "
                             "values the VESC can't hold.")
    parser.add_argument("--vmax",    type=float, default=4.0,
                        help="Maximum speed m/s (default: 4.0)")
    parser.add_argument("--alat",    type=float, default=4.0,
                        help="Max lateral acceleration m/s² (default: 4.0)")
    parser.add_argument("--alon",    type=float, default=2.0,
                        help="Max longitudinal acceleration m/s² (default: 2.0)")
    parser.add_argument("--spacing", type=float, default=0.3,
                        help="Waypoint spacing m (default: 0.3)")
    parser.add_argument("--width-opt", type=float, default=0.3,
                        help="Vehicle total width m for mincurv QP (default: 0.3)")
    parser.add_argument("--wall-margin", type=float, default=0.4,
                        help="Base wall buffer on straights (m, default: 0.4).")
    parser.add_argument("--corner-margin", type=float, default=0.7,
                        help="Wall buffer at tight corners (m, default: 0.7). "
                             "Interpolated with --wall-margin based on curvature.")
    parser.add_argument("--kappa-start", type=float, default=0.3,
                        help="Curvature (rad/m) below which path is pure mincurv — "
                             "no corner protection (default: 0.3).")
    parser.add_argument("--kappa-corner", type=float, default=0.6,
                        help="Curvature (rad/m) above which full corner protection applies "
                             "— path follows centerline (default: 0.6). "
                             "Between kappa-start and kappa-corner = blend.")
    parser.add_argument("--smooth-buffer", type=float, default=0.05,
                        help="Extra QP wall margin (m) to absorb downstream Gaussian-smooth "
                             "drift (default: 0.05). Increase if smoothing pushes the path "
                             "back toward walls.")
    parser.add_argument("--min-clearance", type=float, default=0.3,
                        help="Reporting threshold only — the corner report flags any "
                             "waypoint whose clearance is below this (default: 0.3 m).")
    parser.add_argument("--max-offset", type=float, default=0.5,
                        help="Hard cap on lateral deviation from centerline (m, default: 0.5). "
                             "0 = disabled. Prevents the QP from swinging wide on broad straights.")
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="Gaussian σ (waypoints) applied after QP. Default 0 (off) — "
                             "the QP already produces a smooth min-curvature path. Only "
                             "enable if you see jagged output; increase --smooth-buffer "
                             "alongside it to keep wall clearance.")
    parser.add_argument("--kappa-max", type=float, default=1.32,
                        help="Max path curvature = 1/R_min (default: 1.32 rad/m "
                             "= R_min 0.76m, matches F1TENTH wheelbase/steer limit)")
    parser.add_argument("--contour-ids", default=None,
                        help="Manual contour selection 'OUTER,INNER' (cv2 ids "
                             "printed in the contour summary). Use this when "
                             "the auto-pick traces a noisy outer hull instead "
                             "of the real track walls.")
    parser.add_argument("--skip-outer", type=int, default=1,
                        help="How many outer hierarchy levels to peel before "
                             "looking for the track (default: 1 — assumes the "
                             "outermost contour is the noisy scan boundary). "
                             "Ignored when --contour-ids is given.")
    parser.add_argument("--no-curvature-limit", action="store_true",
                        help="Skip the post-hoc curvature smoothing step. "
                             "Use this when the raw centerline is fine but "
                             "the smoother destroys it (e.g. tight oval "
                             "tracks with a wiggly inner island).")
    parser.add_argument("--png", default=None,
                        help="Path for diagnostic PNG (default: same stem as --out)")
    args = parser.parse_args()

    # ── Step 1: Extract centerline ────────────────────────────────────────────
    print("\n── Step 1: Centerline extraction (contour-based) ──")
    contour_ids = None
    if args.contour_ids:
        try:
            contour_ids = [int(x) for x in args.contour_ids.split(",")]
            if len(contour_ids) != 2:
                raise ValueError
        except ValueError:
            sys.exit("ERROR: --contour-ids must be 'OUTER,INNER' "
                     "with two integers (e.g. 169,171)")
    xy_center, w_right, w_left = extract_centerline(
        args.map, spacing=args.spacing,
        contour_ids=contour_ids,
        skip_outer=args.skip_outer)

    # ── Step 2: Generate racing line ──────────────────────────────────────────
    print(f"\n── Step 2: Lane = '{args.lane}' ──")
    normals = calc_normals(xy_center)

    if args.lane == "centerline":
        xy_race = lane_centerline(xy_center)
    elif args.lane == "mincurv":
        xy_race = lane_mincurv(
            xy_center, normals, w_right, w_left,
            width_opt=args.width_opt,
            wall_margin=args.wall_margin,
            corner_margin=args.corner_margin,
            kappa_start=args.kappa_start,
            kappa_corner=args.kappa_corner,
            max_offset=args.max_offset,
            smooth_buffer=args.smooth_buffer,
            kappa_bound=args.kappa_max,
        )
    elif args.lane == "inner":
        xy_race = lane_offset(xy_center, normals, w_right, w_left, fraction=-0.5)
    elif args.lane == "outer":
        xy_race = lane_offset(xy_center, normals, w_right, w_left, fraction=+0.5)

    # ── Step 2b: Smooth raceline ──────────────────────────────────────────────
    if args.smooth_sigma > 0 and args.lane == "mincurv":
        print(f"\n── Step 2b: Post-QP smoothing (σ={args.smooth_sigma} waypoints) ──")
        xy_race[:, 0] = gaussian_filter1d(xy_race[:, 0], sigma=args.smooth_sigma, mode="wrap")
        xy_race[:, 1] = gaussian_filter1d(xy_race[:, 1], sigma=args.smooth_sigma, mode="wrap")

    # ── Step 2c: Curvature limiting (respect car's steering ability) ─────────
    if args.no_curvature_limit:
        print(f"\n── Step 2c: Curvature limit SKIPPED (--no-curvature-limit) ──")
    else:
        print(f"\n── Step 2c: Curvature limit (κ_max={args.kappa_max:.2f} rad/m) ──")
        xy_race = _limit_curvature(xy_race, kappa_max=args.kappa_max)

    # No post-hoc safety fixation — the QP wall margin already includes a
    # smooth_buffer to absorb downstream smoothing/curvature-limit drift.

    # ── Step 3: Speed profile ─────────────────────────────────────────────────
    print("\n── Step 3: Speed profile (curvature-based) ──")
    v = calc_speed_profile(xy_race, v_max=args.vmax,
                            a_lat=args.alat, a_lon=args.alon,
                            v_min=args.vmin)
    print(f"  v: min={v.min():.2f}  mean={v.mean():.2f}  max={v.max():.2f} m/s")

    # ── Corner report: curvature + actual wall clearance ─────────────────────
    # Re-run the same EDT-based ray-cast on the final RACELINE (not centerline)
    # to get true clearance from each wall at every waypoint.
    kappa_final = calc_curvature(xy_race)
    from scipy.ndimage import label

    # Reuse wall_mask from centerline extraction (re-load map quickly)
    img_cl, res_cl, ox_cl, oy_cl, meta_cl = load_map(args.map)
    occ_thr_cl = float(meta_cl.get("occupied_thresh", 0.65))
    wall_mask_cl = (img_cl < int(occ_thr_cl * 255)).astype(np.uint8)
    H_r, W_r = wall_mask_cl.shape

    race_normals = calc_normals(xy_race)
    step_cl = res_cl
    max_steps_cl = int(3.0 / step_cl)
    clr_left  = np.full(len(xy_race), 3.0)
    clr_right = np.full(len(xy_race), 3.0)
    for i in range(len(xy_race)):
        x0, y0 = float(xy_race[i, 0]), float(xy_race[i, 1])
        nx, ny = float(race_normals[i, 0]), float(race_normals[i, 1])
        for k in range(1, max_steps_cl + 1):
            col = int((x0 + k*step_cl*nx - ox_cl) / res_cl)
            row = int((y0 + k*step_cl*ny - oy_cl) / res_cl)
            if not (0 <= row < H_r and 0 <= col < W_r) or wall_mask_cl[row, col]:
                clr_left[i] = k * step_cl; break
        for k in range(1, max_steps_cl + 1):
            col = int((x0 - k*step_cl*nx - ox_cl) / res_cl)
            row = int((y0 - k*step_cl*ny - oy_cl) / res_cl)
            if not (0 <= row < H_r and 0 <= col < W_r) or wall_mask_cl[row, col]:
                clr_right[i] = k * step_cl; break

    half_w = args.width_opt / 2.0
    gap_left  = clr_left  - half_w   # clearance from car edge to left wall
    gap_right = clr_right - half_w   # clearance from car edge to right wall

    hot = kappa_final > 0.3
    labeled, n_corners = label(hot)
    print(f"\n── Corner Report (κ > 0.3 rad/m, {n_corners} corners) ──")
    print(f"  {'#':>2}  {'κ (rad/m)':>10}  {'R (m)':>6}  {'v (m/s)':>8}  "
          f"{'gap_L (m)':>10}  {'gap_R (m)':>10}  pos")
    for seg_id in range(1, n_corners + 1):
        mask = labeled == seg_id
        peak_idx = int(np.argmax(kappa_final * mask))
        k   = kappa_final[peak_idx]
        R   = 1.0 / k if k > 1e-6 else float("inf")
        spd = v[peak_idx]
        gl  = gap_left[peak_idx]
        gr  = gap_right[peak_idx]
        x, y = xy_race[peak_idx]
        warn = " ← TIGHT" if min(gl, gr) < 0.3 else ""
        print(f"  {seg_id:>2}  {k:>10.3f}  {R:>6.2f}  {spd:>8.2f}  "
              f"{gl:>10.2f}  {gr:>10.2f}  ({x:.1f},{y:.1f}){warn}")

    theta = calc_heading(xy_race)

    # ── Step 4: Save path.yaml ────────────────────────────────────────────────
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

    # ── Step 5: Save PNG ─────────────────────────────────────────────────────
    out_png = args.png or os.path.splitext(args.out)[0] + ".png"
    print(f"\n── Step 5: PNG visualisation ──")
    save_png(args.map, xy_center, xy_race, v, w_right, w_left, args.lane, out_png)

    print(f"Saved {len(waypoints)} waypoints → {args.out}")
    print(f"  x: [{xy_race[:,0].min():.1f}, {xy_race[:,0].max():.1f}]  "
          f"y: [{xy_race[:,1].min():.1f}, {xy_race[:,1].max():.1f}]")


if __name__ == "__main__":
    main()