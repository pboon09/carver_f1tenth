#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(SCRIPT_DIR, "..", "maps")
SIM_YAML = os.path.join(SCRIPT_DIR, "..", "config", "sim.yaml")
OUTPUT_YAML = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "..", "f1tenth_controller", "path", "path.yaml"
))
NUM_WAYPOINTS = 1000


def load_spawn_and_map():
    with open(SIM_YAML) as f:
        config = yaml.safe_load(f)
    params = config["bridge"]["ros__parameters"]
    map_path = params["map_path"]
    map_name = os.path.basename(map_path)
    spawn_x = float(params["sx"])
    spawn_y = float(params["sy"])
    spawn_theta = float(params["stheta"])
    return map_name, spawn_x, spawn_y, spawn_theta


def main():
    map_name, spawn_x, spawn_y, spawn_theta = load_spawn_and_map()
    map_png = os.path.join(MAPS_DIR, f"{map_name}.png")
    map_yaml = os.path.join(MAPS_DIR, f"{map_name}.yaml")

    if not os.path.exists(map_png):
        sys.exit(f"Map not found: {map_png}")

    print(f"Map: {map_name}")
    print(f"Spawn: ({spawn_x}, {spawn_y}, {spawn_theta})")

    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    resolution = meta['resolution']
    origin_x, origin_y = meta['origin'][0], meta['origin'][1]

    img = cv2.imread(map_png, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    occ_thresh = int(meta['occupied_thresh'] * 255)

    drivable = (img > occ_thresh).astype(np.uint8)
    dist_map = distance_transform_edt(drivable) * resolution

    track_mask = ((dist_map > 0.3) & (dist_map < 4.0)).astype(np.uint8)
    spawn_c = int((spawn_x - origin_x) / resolution)
    spawn_r = int(h - (spawn_y - origin_y) / resolution)

    num_labels, labels = cv2.connectedComponents(track_mask)
    spawn_label = labels[spawn_r, spawn_c]
    track_region = (labels == spawn_label).astype(np.uint8)

    skel = skeletonize(track_region > 0).astype(np.uint8)
    skel_pts = np.column_stack(np.where(skel))
    print(f"Skeleton points: {len(skel_pts)}")

    tree = cKDTree(skel_pts)
    _, start_idx = tree.query([spawn_r, spawn_c])

    visited = np.zeros(len(skel_pts), dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    current = start_idx
    for _ in range(len(skel_pts) - 1):
        dists, idxs = tree.query(skel_pts[current], k=30)
        found = False
        for d, i in zip(dists, idxs):
            if not visited[i]:
                order.append(i)
                visited[i] = True
                current = i
                found = True
                break
        if not found:
            break

    ordered = skel_pts[order]
    print(f"Ordered: {len(ordered)} points")

    px_x = ordered[:, 1].astype(float)
    px_y = ordered[:, 0].astype(float)

    all_x = np.append(px_x, px_x[0])
    all_y = np.append(px_y, px_y[0])
    diffs = np.diff(np.column_stack([all_x, all_y]), axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])

    even_s = np.linspace(0, cum[-1], NUM_WAYPOINTS + 1, endpoint=False)
    even_x = np.interp(even_s, cum, all_x)
    even_y = np.interp(even_s, cum, all_y)

    window = 31
    pad = window
    ex = np.concatenate([even_x[-pad:], even_x, even_x[:pad]])
    ey = np.concatenate([even_y[-pad:], even_y, even_y[:pad]])
    sx = savgol_filter(ex, window, 3)[pad:-pad]
    sy = savgol_filter(ey, window, 3)[pad:-pad]

    world_x = sx * resolution + origin_x
    world_y = (h - sy) * resolution + origin_y

    world_x[0] = spawn_x
    world_y[0] = spawn_y

    world_x = world_x[:NUM_WAYPOINTS]
    world_y = world_y[:NUM_WAYPOINTS]

    dx = np.diff(world_x, append=world_x[0])
    dy = np.diff(world_y, append=world_y[0])
    theta = np.arctan2(dy, dx)
    theta[0] = spawn_theta

    loop_length = np.sqrt(dx**2 + dy**2).sum()
    print(f"Loop length: {loop_length:.1f} m, {NUM_WAYPOINTS} waypoints")
    print(f"First waypoint: ({world_x[0]}, {world_y[0]}, {theta[0]})")

    waypoints_list = []
    for i in range(NUM_WAYPOINTS):
        waypoints_list.append({
            'x': round(float(world_x[i]), 4),
            'y': round(float(world_y[i]), 4),
            'theta': round(float(theta[i]), 4),
        })

    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump({'waypoints': waypoints_list}, f, default_flow_style=False)
    print(f"Saved: {OUTPUT_YAML}")

    output_img = os.path.join(SCRIPT_DIR, f"{map_name}_centerline.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(img, cmap='gray')
    axes[0].plot(sx[:NUM_WAYPOINTS], sy[:NUM_WAYPOINTS], 'r-', linewidth=1.2)
    axes[0].plot(sx[0], sy[0], 'go', markersize=8)
    axes[0].set_title('Centerline on Map')
    axes[0].axis('off')

    axes[1].plot(world_x, world_y, 'r-', linewidth=1.2)
    axes[1].plot(world_x[0], world_y[0], 'go', markersize=8, label=f'Spawn ({spawn_x}, {spawn_y})')
    axes[1].set_title(f'World Coords - {loop_length:.1f}m loop, {NUM_WAYPOINTS} pts')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"Saved image: {output_img}")


if __name__ == "__main__":
    main()
