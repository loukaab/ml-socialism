"""Late-game performance benchmark for the ABM simulation."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import WorldModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile late-game simulation steps.")
    parser.add_argument("--target-inhabitants", type=int, default=100_000)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--max-warmup-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=35)
    parser.add_argument("--populations", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = WorldModel(
        width=args.width,
        height=args.height,
        initial_populations=args.populations,
        seed=args.seed,
    )

    warmup_start = time.perf_counter()
    warmup_steps = 0
    while (
        model.total_inhabitants() < args.target_inhabitants
        and warmup_steps < args.max_warmup_steps
    ):
        model.step()
        warmup_steps += 1
    warmup_seconds = time.perf_counter() - warmup_start

    profiler = cProfile.Profile()
    profile_start = time.perf_counter()
    profiler.enable()
    for _ in range(args.profile_steps):
        model.step()
    profiler.disable()
    profile_seconds = time.perf_counter() - profile_start

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)

    print(f"Warmup steps: {warmup_steps}")
    print(f"Warmup seconds: {warmup_seconds:.3f}")
    print(f"Inhabitants: {model.total_inhabitants()}")
    print(f"Population agents: {len(model.population_agents)}")
    print(f"Profiled steps: {args.profile_steps}")
    print(f"Profile seconds: {profile_seconds:.3f}")
    print(f"Seconds per step: {profile_seconds / max(1, args.profile_steps):.4f}")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
