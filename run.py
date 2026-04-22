"""Command-line entry point for the phase-1 economic simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from model import WorldModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the phase-1 ABM world.")
    parser.add_argument("--width", type=int, default=50, help="Grid width.")
    parser.add_argument("--height", type=int, default=35, help="Grid height.")
    parser.add_argument(
        "--populations",
        type=int,
        default=8,
        help="Number of founding population agents.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Simulation ticks to run before opening or rendering.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a rendered map image.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render an image and exit instead of opening the interactive viewer.",
    )
    parser.add_argument(
        "--resource-overlay",
        action="store_true",
        help="Start with or render the static resource overlay.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Interactive window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=820,
        help="Interactive window height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Interactive viewer frame limit.",
    )
    return parser


def print_summary(model: WorldModel, output: Optional[Path] = None) -> None:
    model_data = model.datacollector.get_model_vars_dataframe()
    latest = model_data.iloc[-1].to_dict()

    if output:
        print(f"Rendered world to {output}")
    print(f"Population agents: {latest['PopulationAgents']}")
    print(f"Occupied tiles: {latest['OccupiedTiles']}")
    print(f"Total inhabitants: {latest['TotalInhabitants']}")
    print(f"Expansion events: {latest['ExpansionEvents']}")
    print(f"Surviving lineages: {latest['SurvivingLineages']}")
    print(f"Max tech level: {latest['MaxTech']}")
    print(f"Dominant trait: {latest['DominantTrait']}")


def main() -> None:
    args = build_parser().parse_args()
    model = WorldModel(
        width=args.width,
        height=args.height,
        initial_populations=args.populations,
        seed=args.seed,
    )

    for _ in range(args.steps):
        model.step()

    if args.headless:
        output = args.output or Path("initialized_world.png")
        model.render_map(
            output_path=str(output),
            show=False,
            resource_overlay=args.resource_overlay,
        )
        print_summary(model, output)
        return

    if args.output:
        model.render_map(
            output_path=str(args.output),
            show=False,
            resource_overlay=args.resource_overlay,
        )
        print_summary(model, args.output)

    from viewer import InteractiveViewer

    viewer = InteractiveViewer(
        model,
        width=args.window_width,
        height=args.window_height,
        fps=args.fps,
    )
    viewer.show_resource_overlay = args.resource_overlay
    viewer.run()
    print_summary(model)


if __name__ == "__main__":
    main()
