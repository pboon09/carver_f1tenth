#!/usr/bin/env python3

from __future__ import annotations

from generate_obstacle_map import Obstacle, ObstacleMapGenerator


VARIANTS = {
    "easy": [
        Obstacle(4.0, 0.0, 0.2),
    ],
    "medium": [
        Obstacle(-5.0, 0.0, 0.2),
        Obstacle(5.0, 8.7, 0.25),
        Obstacle(-13.7, 5.0, 0.2),
    ],
    "hard": [
        Obstacle(-3.0, 0.0, 0.25),
        Obstacle(5.0, 0.0, 0.25),
        Obstacle(-8.0, 8.7, 0.25),
        Obstacle(3.0, 8.7, 0.25),
        Obstacle(9.7, 4.0, 0.2),
    ],
    "chicane": [
        Obstacle(-4.0, -0.3, 0.2),
        Obstacle(-2.0, 0.3, 0.2),
        Obstacle(4.0, -0.3, 0.2),
        Obstacle(6.0, 0.3, 0.2),
    ],
    "dense": [
        Obstacle(-10.0, 0.0, 0.3),
        Obstacle(-5.0, 0.0, 0.2),
        Obstacle(-3.0, 0.0, 0.25),
        Obstacle(5.0, 0.0, 0.2),
        Obstacle(-10.0, 8.7, 0.25),
        Obstacle(-3.0, 8.7, 0.3),
        Obstacle(4.0, 8.7, 0.2),
        Obstacle(-13.7, 3.0, 0.2),
        Obstacle(-13.7, 7.0, 0.2),
        Obstacle(9.7, 4.0, 0.2),
    ],
}


def main() -> None:
    for suffix, obstacles in VARIANTS.items():
        print(f"{'=' * 50}")
        print(f"Generating: levine_{suffix} ({len(obstacles)} obstacles)")
        print(f"{'=' * 50}")

        generator = ObstacleMapGenerator(
            base_map_name="levine",
            output_suffix=suffix,
            obstacles=obstacles,
        )
        output_path = generator.generate()
        print(f"  -> {output_path}\n")

    print("All variants generated.")


if __name__ == "__main__":
    main()
