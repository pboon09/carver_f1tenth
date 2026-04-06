#!/usr/bin/env python3

from __future__ import annotations

from generate_obstacle_map import Obstacle, ObstacleMapGenerator


BASE_MAP = "Spielberg_map"

VARIANTS = {
    "easy": [
        Obstacle(-31.7, 18.5, 0.10),
        Obstacle(-58.7, 53.6, 0.10),
    ],
    "medium": [
        Obstacle(18.6, 24.5, 0.10),
        Obstacle(-44.4, 22.6, 0.10),
        Obstacle(-42.0, 50.5, 0.10),
        Obstacle(-24.3, -6.5, 0.10),
    ],
    "hard": [
        Obstacle(8.1, 2.2, 0.10),
        Obstacle(23.4, 8.8, 0.10),
        Obstacle(-16.6, 25.0, 0.10),
        Obstacle(-45.4, 37.3, 0.10),
        Obstacle(-24.9, 48.3, 0.10),
        Obstacle(-75.9, 52.5, 0.10),
        Obstacle(-47.9, 10.1, 0.10),
        Obstacle(-38.0, -3.8, 0.10),
    ],
    "chicane": [
        Obstacle(8.1, 2.2, 0.10),
        Obstacle(18.6, 24.5, 0.10),
        Obstacle(-13.9, 42.6, 0.10),
        Obstacle(-55.5, 25.0, 0.10),
        Obstacle(-24.3, -6.5, 0.10),
        Obstacle(-64.1, 39.5, 0.10),
    ],
    "dense": [
        Obstacle(8.1, 2.2, 0.10),
        Obstacle(23.4, 8.8, 0.10),
        Obstacle(18.6, 24.5, 0.10),
        Obstacle(1.2, 25.3, 0.10),
        Obstacle(-16.6, 25.0, 0.10),
        Obstacle(-31.7, 18.5, 0.10),
        Obstacle(-45.4, 37.3, 0.10),
        Obstacle(-13.9, 42.6, 0.10),
        Obstacle(-42.0, 50.5, 0.10),
        Obstacle(-75.9, 52.5, 0.10),
        Obstacle(-64.1, 39.5, 0.10),
        Obstacle(-55.5, 25.0, 0.10),
        Obstacle(-47.9, 10.1, 0.10),
        Obstacle(-38.0, -3.8, 0.10),
        Obstacle(-24.3, -6.5, 0.10),
    ],
}


def main() -> None:
    for suffix, obstacles in VARIANTS.items():
        print(f"{'=' * 50}")
        print(f"Generating: {BASE_MAP}_{suffix} ({len(obstacles)} obstacles)")
        print(f"{'=' * 50}")

        generator = ObstacleMapGenerator(
            base_map_name=BASE_MAP,
            output_suffix=suffix,
            obstacles=obstacles,
        )
        output_path = generator.generate()
        print(f"  -> {output_path}\n")

    print("All variants generated.")


if __name__ == "__main__":
    main()
