#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label


MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"
SIM_YAML = Path(__file__).resolve().parent.parent / "config" / "sim.yaml"

OBSTACLE_PIXEL_VALUE = 0
MIN_RADIUS_CELLS = 2

CAR_WIDTH = 0.31
CAR_LENGTH = 0.58
SAFETY_MARGIN = 0.10
MIN_CLEARANCE = CAR_WIDTH / 2 + SAFETY_MARGIN
SPAWN_EXCLUSION_RADIUS = CAR_LENGTH + SAFETY_MARGIN

CONNECTIVITY_STRUCTURE = generate_binary_structure(2, 2)


@dataclass
class SpawnPose:
    x: float
    y: float
    theta: float = 0.0


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


@dataclass
class MapMetadata:
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> MapMetadata:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(
            image=data["image"],
            resolution=data["resolution"],
            origin=tuple(data["origin"]),
            negate=data["negate"],
            occupied_thresh=data["occupied_thresh"],
            free_thresh=data["free_thresh"],
        )

    def to_yaml(self, yaml_path: Path) -> None:
        data = {
            "image": self.image,
            "resolution": self.resolution,
            "origin": list(self.origin),
            "negate": self.negate,
            "occupied_thresh": self.occupied_thresh,
            "free_thresh": self.free_thresh,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def free_pixel_threshold(self) -> int:
        return int(255 * (1.0 - self.free_thresh))


@dataclass
class FeasibilityResult:
    obstacle: Obstacle
    effective_radius: float
    on_track: bool
    wall_clearance: float
    remaining_gap: float
    passable: bool


@dataclass
class ObstacleMapGenerator:
    base_map_name: str
    output_suffix: str = "obs"
    obstacles: List[Obstacle] = field(default_factory=list)
    skip_feasibility: bool = False

    @property
    def base_image_path(self) -> Path:
        return MAPS_DIR / f"{self.base_map_name}.png"

    @property
    def base_yaml_path(self) -> Path:
        return MAPS_DIR / f"{self.base_map_name}.yaml"

    @property
    def output_name(self) -> str:
        return f"{self.base_map_name}_{self.output_suffix}"

    @property
    def output_image_path(self) -> Path:
        return MAPS_DIR / f"{self.output_name}.png"

    @property
    def output_yaml_path(self) -> Path:
        return MAPS_DIR / f"{self.output_name}.yaml"

    def world_to_pixel(
        self, x: float, y: float, metadata: MapMetadata, image_height: int
    ) -> Tuple[int, int]:
        origin_x, origin_y, _ = metadata.origin
        u = int((x - origin_x) / metadata.resolution)
        v = image_height - int((y - origin_y) / metadata.resolution)
        return u, v

    def radius_to_pixels(self, radius: float, resolution: float) -> int:
        return max(round(radius / resolution), MIN_RADIUS_CELLS)

    def effective_radius(self, obstacle: Obstacle, resolution: float) -> float:
        return self.radius_to_pixels(obstacle.radius, resolution) * resolution

    def validate(self) -> None:
        if not self.base_image_path.exists():
            sys.exit(f"Base image not found: {self.base_image_path}")
        if not self.base_yaml_path.exists():
            sys.exit(f"Base YAML not found: {self.base_yaml_path}")
        if not self.obstacles:
            sys.exit("No obstacles defined.")

    def _build_free_mask(
        self, array: np.ndarray, metadata: MapMetadata
    ) -> np.ndarray:
        return array > metadata.free_pixel_threshold()

    def _build_distance_field(
        self, array: np.ndarray, metadata: MapMetadata
    ) -> np.ndarray:
        free_mask = self._build_free_mask(array, metadata)
        return distance_transform_edt(free_mask) * metadata.resolution

    def _check_on_track(
        self, base_array: np.ndarray, u: int, v: int, metadata: MapMetadata
    ) -> bool:
        if not (0 <= u < base_array.shape[1] and 0 <= v < base_array.shape[0]):
            return False
        return bool(base_array[v, u] > metadata.free_pixel_threshold())

    def _paint_obstacles(
        self, img: Image.Image, metadata: MapMetadata
    ) -> Image.Image:
        draw = ImageDraw.Draw(img)
        for obstacle in self.obstacles:
            u, v = self.world_to_pixel(obstacle.x, obstacle.y, metadata, img.height)
            rp = self.radius_to_pixels(obstacle.radius, metadata.resolution)
            draw.ellipse(
                (u - rp, v - rp, u + rp, v + rp), fill=OBSTACLE_PIXEL_VALUE
            )
        return img

    def check_feasibility(
        self, metadata: MapMetadata, painted_array: np.ndarray, base_array: np.ndarray
    ) -> List[FeasibilityResult]:
        dist_field = self._build_distance_field(painted_array, metadata)
        results = []

        for obstacle in self.obstacles:
            u, v = self.world_to_pixel(
                obstacle.x, obstacle.y, metadata, base_array.shape[0]
            )
            on_track = self._check_on_track(base_array, u, v, metadata)
            eff_radius = self.effective_radius(obstacle, metadata.resolution)

            if 0 <= u < painted_array.shape[1] and 0 <= v < painted_array.shape[0]:
                rp = self.radius_to_pixels(obstacle.radius, metadata.resolution)
                search_radius = rp + int(MIN_CLEARANCE / metadata.resolution) + 2
                y_min = max(0, v - search_radius)
                y_max = min(painted_array.shape[0], v + search_radius + 1)
                x_min = max(0, u - search_radius)
                x_max = min(painted_array.shape[1], u + search_radius + 1)

                region = dist_field[y_min:y_max, x_min:x_max]
                free_region = self._build_free_mask(
                    painted_array[y_min:y_max, x_min:x_max], metadata
                )

                if np.any(free_region):
                    wall_clearance = float(np.max(region[free_region]))
                else:
                    wall_clearance = 0.0
            else:
                wall_clearance = 0.0

            remaining_gap = wall_clearance
            passable = remaining_gap >= MIN_CLEARANCE

            results.append(
                FeasibilityResult(
                    obstacle=obstacle,
                    effective_radius=eff_radius,
                    on_track=on_track,
                    wall_clearance=wall_clearance,
                    remaining_gap=remaining_gap,
                    passable=passable,
                )
            )

        return results

    def check_spawn_exclusion(
        self,
        spawn_poses: List[SpawnPose],
        metadata: MapMetadata,
    ) -> List[Tuple[Obstacle, SpawnPose, float]]:
        violations = []
        for obstacle in self.obstacles:
            eff_r = self.effective_radius(obstacle, metadata.resolution)
            for pose in spawn_poses:
                dist = ((obstacle.x - pose.x) ** 2 + (obstacle.y - pose.y) ** 2) ** 0.5
                min_dist = eff_r + SPAWN_EXCLUSION_RADIUS
                if dist < min_dist:
                    violations.append((obstacle, pose, dist))
        return violations

    def check_connectivity(
        self,
        painted_array: np.ndarray,
        metadata: MapMetadata,
        spawn_x: float,
        spawn_y: float,
    ) -> bool:
        free_mask = self._build_free_mask(painted_array, metadata)
        labeled, _ = label(free_mask, structure=CONNECTIVITY_STRUCTURE)

        u, v = self.world_to_pixel(
            spawn_x, spawn_y, metadata, painted_array.shape[0]
        )
        if not (0 <= u < painted_array.shape[1] and 0 <= v < painted_array.shape[0]):
            print(f"WARN: spawn point ({spawn_x}, {spawn_y}) is outside map bounds")
            return False

        if not free_mask[v, u]:
            print(f"WARN: spawn point ({spawn_x}, {spawn_y}) is not in free space")
            return False

        spawn_label = labeled[v, u]

        region_sizes = np.bincount(labeled.ravel())
        region_sizes[0] = 0
        largest_label = int(np.argmax(region_sizes))

        return spawn_label == largest_label

    def generate(
        self, spawn_poses: List[SpawnPose] | None = None
    ) -> Path:
        self.validate()

        if spawn_poses is None:
            spawn_poses = load_spawn_poses_from_sim_yaml()

        metadata = MapMetadata.from_yaml(self.base_yaml_path)
        img = Image.open(self.base_image_path).convert("L")
        base_array = np.array(img)

        if not self.skip_feasibility:
            violations = self.check_spawn_exclusion(spawn_poses, metadata)
            for obstacle, pose, dist in violations:
                min_dist = self.effective_radius(obstacle, metadata.resolution) + SPAWN_EXCLUSION_RADIUS
                print(
                    f"ERROR: obstacle at ({obstacle.x}, {obstacle.y}) is {dist:.2f}m "
                    f"from spawn ({pose.x}, {pose.y}), need >= {min_dist:.2f}m"
                )
            if violations:
                sys.exit("Aborted: obstacles overlap with spawn positions.")

        img = self._paint_obstacles(img, metadata)

        if not self.skip_feasibility:
            painted_array = np.array(img)
            results = self.check_feasibility(metadata, painted_array, base_array)
            has_warnings = False

            for r in results:
                pos = f"({r.obstacle.x}, {r.obstacle.y})"
                if not r.on_track:
                    print(f"WARN: obstacle at {pos} is not on drivable surface")
                    has_warnings = True
                if abs(r.effective_radius - r.obstacle.radius) > 1e-6:
                    print(
                        f"WARN: obstacle at {pos} radius clamped from "
                        f"{r.obstacle.radius:.3f}m to {r.effective_radius:.3f}m"
                    )
                    has_warnings = True
                if not r.passable:
                    print(
                        f"WARN: obstacle at {pos} leaves {r.remaining_gap:.2f}m "
                        f"max gap, need >= {MIN_CLEARANCE:.2f}m for car to pass"
                    )
                    has_warnings = True

            if has_warnings:
                print()

            spawn_main = spawn_poses[0] if spawn_poses else SpawnPose(0.0, 0.0)
            if not self.check_connectivity(painted_array, metadata, spawn_main.x, spawn_main.y):
                print("WARN: obstacles disconnect the track from spawn point\n")

        img.save(self.output_image_path)

        output_metadata = copy.deepcopy(metadata)
        output_metadata.image = f"{self.output_name}.png"
        output_metadata.to_yaml(self.output_yaml_path)

        return self.output_image_path


def load_spawn_poses_from_sim_yaml() -> List[SpawnPose]:
    if not SIM_YAML.exists():
        return [SpawnPose(0.0, 0.0)]

    with open(SIM_YAML) as f:
        config = yaml.safe_load(f)

    params = config["bridge"]["ros__parameters"]
    poses = [SpawnPose(params["sx"], params["sy"], params["stheta"])]

    if params.get("num_agent", 1) > 1:
        poses.append(SpawnPose(params["sx1"], params["sy1"], params["stheta1"]))

    return poses


def parse_obstacle(value: str) -> Obstacle:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected x,y,radius but got: '{value}'"
        )
    x, y, radius = float(parts[0]), float(parts[1]), float(parts[2])
    if radius <= 0:
        raise argparse.ArgumentTypeError(f"Radius must be positive, got: {radius}")
    return Obstacle(x=x, y=y, radius=radius)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate obstacle map variants for f1tenth_gym_ros."
    )
    parser.add_argument(
        "--base-map",
        type=str,
        default="levine",
        help="Base map name without extension (default: levine)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="obs",
        help="Output file suffix, e.g. 'obs' produces levine_obs.png (default: obs)",
    )
    parser.add_argument(
        "--obstacle",
        type=parse_obstacle,
        action="append",
        dest="obstacles",
        required=True,
        metavar="X,Y,RADIUS",
        help="Obstacle in world coordinates as x,y,radius (meters). Repeat for multiple.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip all feasibility and connectivity checks",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    generator = ObstacleMapGenerator(
        base_map_name=args.base_map,
        output_suffix=args.suffix,
        obstacles=args.obstacles,
        skip_feasibility=args.no_check,
    )

    output_path = generator.generate()
    print(f"Generated: {output_path}")
    print(f"YAML:      {generator.output_yaml_path}")
    print(f"Obstacles: {len(generator.obstacles)}")
    print(
        f"\nTo use this map, update sim.yaml map_path to:\n"
        f"  {MAPS_DIR / generator.output_name}"
    )


if __name__ == "__main__":
    main()
