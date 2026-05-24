#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion, label as cc_label

def load_map(yaml_path):
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = yaml_path.parent / meta["image"]
    img = np.array(Image.open(img_path))
    if img.ndim == 3:
        img = img[..., 0]
    return img, meta


def to_binary_free(img, free_thresh_pixel=250):
    """Free = pixel value >= free_thresh_pixel (white). 1 = free, 0 = obstacle/unknown."""
    return (img >= free_thresh_pixel).astype(np.uint8)


def keep_largest_cc(binary):
    labeled, n = cc_label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0           # exclude background
    largest = counts.argmax()
    return (labeled == largest).astype(np.uint8)

def _shift(img, dr, dc):
    out = np.zeros_like(img)
    h, w = img.shape
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    out[r_dst, c_dst] = img[r_src, c_src]
    return out


def _zs_neighbours(img):
    P2 = _shift(img, -1, 0)
    P3 = _shift(img, -1, 1)
    P4 = _shift(img, 0, 1)
    P5 = _shift(img, 1, 1)
    P6 = _shift(img, 1, 0)
    P7 = _shift(img, 1, -1)
    P8 = _shift(img, 0, -1)
    P9 = _shift(img, -1, -1)
    return P2, P3, P4, P5, P6, P7, P8, P9


def _zs_step(img, step):
    P2, P3, P4, P5, P6, P7, P8, P9 = _zs_neighbours(img)
    B = (P2.astype(int) + P3 + P4 + P5 + P6 + P7 + P8 + P9)
    seq = np.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], axis=0).astype(int)
    A = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(axis=0)
    base = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
    if step == 1:
        return base & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
    return base & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)


def skeletonize(binary, max_iter=500):
    img = binary.astype(np.uint8).copy()
    for _ in range(max_iter):
        c1 = _zs_step(img, 1)
        if c1.any():
            img[c1] = 0
        c2 = _zs_step(img, 2)
        if c2.any():
            img[c2] = 0
        if not c1.any() and not c2.any():
            break
    return img

def prune_spurs(skel, min_branch_length=30):
    """Iteratively delete branches shorter than min_branch_length pixels.
    A 'branch' is a chain starting from an endpoint (degree-1 pixel) and
    ending at the first junction (degree>=3) or another endpoint."""
    skel = skel.astype(np.uint8).copy()
    while True:
        pts_set = set(zip(*np.where(skel == 1)))
        if not pts_set:
            break

        def nbrs(p):
            r, c = p
            return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0) and (r + dr, c + dc) in pts_set]

        endpoints = [p for p in pts_set if len(nbrs(p)) == 1]
        if not endpoints:
            break

        removed_any = False
        already_removed = set()
        for ep in endpoints:
            if ep in already_removed:
                continue
            walk = [ep]
            prev = None
            cur = ep
            while True:
                ns = [n for n in nbrs(cur) if n != prev and n in pts_set
                      and n not in already_removed]
                if len(ns) != 1:
                    break       # endpoint or junction
                prev = cur
                cur = ns[0]
                walk.append(cur)
                if len(walk) > min_branch_length:
                    break
            if len(walk) <= min_branch_length:
                for p in walk:
                    already_removed.add(p)
                removed_any = True
        if not removed_any:
            break
        for p in already_removed:
            skel[p] = 0
    return skel

def trace_skeleton(skel):
    """Walk the skeleton, return ordered list of (row, col)."""
    pts = list(zip(*np.where(skel == 1)))
    if not pts:
        return []
    pts_set = set(pts)

    def neighbours(p):
        r, c = p
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0) and (r + dr, c + dc) in pts_set]

    endpoints = [p for p in pts if len(neighbours(p)) == 1]
    start = endpoints[0] if endpoints else pts[0]

    visited = {start}
    path = [start]
    prev = None
    cur = start
    while True:
        cands = [n for n in neighbours(cur) if n != prev and n not in visited]
        if not cands:
            break
        cands.sort(key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
        prev = cur
        cur = nxt
    return path

def pixel_to_world(rc, meta, img_h):
    """ROS map_server convention: yaml origin is the bottom-left of the image.
    PIL image[0, :] is the TOP row, so y must be flipped."""
    r, c = rc
    ox, oy = meta["origin"][0], meta["origin"][1]
    res = meta["resolution"]
    x = ox + (c + 0.5) * res
    y = oy + (img_h - r - 0.5) * res
    return x, y

def write_yaml(waypoints, out_path):
    with open(out_path, "w") as f:
        f.write("waypoints:\n")
        for wp in waypoints:
            f.write(f"- theta: {wp['theta']:.6f}\n")
            f.write(f"  v: {wp['v']}\n")
            f.write(f"  x: {wp['x']:.6f}\n")
            f.write(f"  y: {wp['y']:.6f}\n")


def plot(img, meta, waypoints, out_png):
    h, w = img.shape
    res = meta["resolution"]
    ox, oy = meta["origin"][0], meta["origin"][1]
    extent = [ox, ox + w * res, oy, oy + h * res]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img, cmap="gray", extent=extent)
    xs = [wp["x"] for wp in waypoints]
    ys = [wp["y"] for wp in waypoints]
    ax.plot(xs, ys, "-", color="#d62728", linewidth=1.6, label="Centerline")
    ax.scatter([xs[0]], [ys[0]], color="#2ca02c", s=80, zorder=5, label="Start")
    ax.scatter([xs[-1]], [ys[-1]], color="#1f77b4", s=80, zorder=5, label="End")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Map + centerline ({len(waypoints)} waypoints)")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    return fig

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml", help="Path to the map .yaml (alongside .pgm)")
    ap.add_argument("--output", default="centerline_from_map.yaml",
                    help="Output YAML path (default: centerline_from_map.yaml)")
    ap.add_argument("--default-v", type=float, default=2.0,
                    help="v value written to every waypoint (default: 2.0)")
    ap.add_argument("--step", type=int, default=4,
                    help="Keep every Nth skeleton pixel (default: 4 → ~10 cm at 0.025 res)")
    ap.add_argument("--free-thresh", type=int, default=250,
                    help="PGM pixel value >= this is treated as free (default: 250)")
    ap.add_argument("--erode", type=int, default=3,
                    help="Erode the free mask by N px before skeletonizing — "
                         "removes wall-noise that creates spurs (default: 3)")
    ap.add_argument("--prune", type=int, default=30,
                    help="Delete skeleton branches shorter than N px (default: 30)")
    ap.add_argument("--no-show", action="store_true",
                    help="Save the plot PNG but don't open the matplotlib window")
    args = ap.parse_args()

    img, meta = load_map(args.map_yaml)
    print(f"Map     : {img.shape[1]}×{img.shape[0]} px, "
          f"resolution {meta['resolution']} m, origin {meta['origin']}")

    free = to_binary_free(img, args.free_thresh)
    free = keep_largest_cc(free)
    print(f"Free CC : {free.sum()} px")

    if args.erode > 0:
        free = binary_erosion(free, iterations=args.erode).astype(np.uint8)
        free = keep_largest_cc(free)
        print(f"Eroded  : {free.sum()} px (erode={args.erode})")

    skel = skeletonize(free)
    print(f"Skeleton: {skel.sum()} px")

    if args.prune > 0:
        skel = prune_spurs(skel, min_branch_length=args.prune)
        print(f"Pruned  : {skel.sum()} px (min_branch={args.prune})")

    path_pix = trace_skeleton(skel)
    print(f"Traced  : {len(path_pix)} ordered px")

    path_pix = path_pix[::args.step]
    print(f"Resample: {len(path_pix)} px (every {args.step}th)")

    img_h = img.shape[0]
    coords = [pixel_to_world(rc, meta, img_h) for rc in path_pix]

    waypoints = []
    for i, (x, y) in enumerate(coords):
        nx, ny = coords[(i + 1) % len(coords)]    # close the loop for heading
        theta = math.atan2(ny - y, nx - x)
        waypoints.append({"theta": theta, "v": args.default_v, "x": x, "y": y})

    out_yaml = Path(args.output)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(waypoints, out_yaml)
    print(f"Wrote   : {out_yaml} ({len(waypoints)} waypoints)")

    out_png = out_yaml.with_suffix(".png")
    plot(img, meta, waypoints, out_png)
    print(f"Plot    : {out_png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
