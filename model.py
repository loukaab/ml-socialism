"""World model, terrain generation, scheduling, and rendering."""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import Population, ResourceCell, TraitMap, normalize_traits, random_traits

try:
    import mesa
except ModuleNotFoundError:
    class _Model:
        def __init__(self, *args, **kwargs):
            self.running = True

    mesa = SimpleNamespace(Model=_Model)

try:
    from mesa.space import MultiGrid
except (ModuleNotFoundError, ImportError):
    MultiGrid = None

try:
    from mesa.time import RandomActivation
except (ModuleNotFoundError, ImportError):
    RandomActivation = None

try:
    from mesa.datacollection import DataCollector
except (ModuleNotFoundError, ImportError):
    DataCollector = None


Position = Tuple[int, int]


class FallbackMultiGrid:
    """Small subset of Mesa's MultiGrid API used by this project."""

    def __init__(self, width: int, height: int, torus: bool = False) -> None:
        self.width = width
        self.height = height
        self.torus = torus
        self._cells: Dict[Position, List[object]] = {
            (x, y): [] for x in range(width) for y in range(height)
        }

    def out_of_bounds(self, pos: Position) -> bool:
        x, y = pos
        return not (0 <= x < self.width and 0 <= y < self.height)

    def place_agent(self, agent, pos: Position) -> None:
        if self.out_of_bounds(pos):
            raise ValueError(f"Position outside grid: {pos}")
        self._cells[pos].append(agent)
        agent.pos = pos

    def remove_agent(self, agent) -> None:
        if agent.pos is not None and agent in self._cells[agent.pos]:
            self._cells[agent.pos].remove(agent)
        agent.pos = None

    def move_agent(self, agent, pos: Position) -> None:
        self.remove_agent(agent)
        self.place_agent(agent, pos)

    def get_cell_list_contents(self, positions: Iterable[Position]) -> List[object]:
        contents: List[object] = []
        for pos in positions:
            if not self.out_of_bounds(pos):
                contents.extend(self._cells[pos])
        return contents

    def get_neighborhood(
        self,
        pos: Position,
        moore: bool = True,
        include_center: bool = False,
        radius: int = 1,
    ) -> List[Position]:
        x, y = pos
        neighbors: List[Position] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and not include_center:
                    continue
                if not moore and abs(dx) + abs(dy) > radius:
                    continue
                next_pos = (x + dx, y + dy)
                if not self.out_of_bounds(next_pos):
                    neighbors.append(next_pos)
        return neighbors


class FallbackRandomActivation:
    """Random scheduler compatible with Mesa's old RandomActivation shape."""

    def __init__(self, model) -> None:
        self.model = model
        self.agents: List[object] = []
        self.steps = 0
        self.time = 0

    def add(self, agent) -> None:
        self.agents.append(agent)

    def remove(self, agent) -> None:
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self) -> None:
        agents = list(self.agents)
        self.model.python_random.shuffle(agents)
        for agent in agents:
            if agent in self.agents:
                agent.step()
        self.steps += 1
        self.time += 1


class FallbackDataCollector:
    """Simple model-level data collector with a Mesa-like interface."""

    def __init__(
        self,
        model_reporters: Optional[Dict[str, Callable]] = None,
        agent_reporters: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self._model_records: List[Dict[str, object]] = []
        self._agent_records: List[Dict[str, object]] = []

    def collect(self, model) -> None:
        step = getattr(model.schedule, "steps", 0)
        model_row = {"Step": step}
        for name, reporter in self.model_reporters.items():
            model_row[name] = reporter(model)
        self._model_records.append(model_row)

        for agent in model.schedule.agents:
            row = {
                "Step": step,
                "AgentID": getattr(agent, "unique_id", None),
                "AgentType": type(agent).__name__,
            }
            for name, reporter in self.agent_reporters.items():
                row[name] = reporter(agent)
            self._agent_records.append(row)

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._model_records)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._agent_records)


GridClass = MultiGrid or FallbackMultiGrid
SchedulerClass = RandomActivation or FallbackRandomActivation
CollectorClass = DataCollector or FallbackDataCollector


@dataclass(frozen=True)
class TerrainConfig:
    land_fraction: float = 0.58
    smoothing_passes: int = 5
    land_capacity_min: float = 70.0
    land_capacity_max: float = 130.0
    land_regen_min: float = 1.2
    land_regen_max: float = 3.0


class WorldModel(mesa.Model):
    """Localized capitalist baseline world for the phase-1 ABM."""

    def __init__(
        self,
        width: int = 50,
        height: int = 35,
        initial_populations: int = 8,
        seed: Optional[int] = None,
        terrain: TerrainConfig = TerrainConfig(),
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.initial_populations = initial_populations
        self.terrain_config = terrain
        self.rng = np.random.default_rng(seed)
        self.python_random = random.Random(seed)

        self.grid = GridClass(width, height, torus=False)
        self.schedule = SchedulerClass(self)
        self._next_id = 1
        self.resource_cells: Dict[Position, ResourceCell] = {}

        self.tech_threshold = 35.0
        self.tech_diffusion_rate = 0.35
        self.trait_drift_rate = 0.35
        self.expansion_threshold = 145
        self.migration_fraction = 0.25
        self.resource_upkeep_per_person = 0.035
        self.population_growth_rate = 0.025

        self.terrain_map = self._generate_terrain()
        self._populate_resource_layer()
        self._seed_populations()

        self.datacollector = CollectorClass(
            model_reporters={
                "MaxTech": lambda model: model.max_tech_level(),
                "SurvivingLineages": lambda model: model.surviving_lineage_count(),
                "DominantTrait": lambda model: model.dominant_trait(),
                "PopulationAgents": lambda model: len(model.populations),
            },
            agent_reporters={
                "Inhabitants": lambda agent: getattr(agent, "inhabitant_count", None),
                "TechLevel": lambda agent: getattr(agent, "tech_level", None),
                "LineageColor": lambda agent: getattr(agent, "lineage_color", None),
            },
        )
        self.datacollector.collect(self)

    def next_id(self) -> int:
        unique_id = self._next_id
        self._next_id += 1
        return unique_id

    @property
    def populations(self) -> List[Population]:
        return [
            agent for agent in self.schedule.agents if isinstance(agent, Population)
        ]

    def _generate_terrain(self) -> np.ndarray:
        noise = self.rng.random((self.height, self.width))
        for _ in range(self.terrain_config.smoothing_passes):
            padded = np.pad(noise, 1, mode="edge")
            noise = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0

        threshold = np.quantile(noise, 1.0 - self.terrain_config.land_fraction)
        return noise >= threshold

    def _populate_resource_layer(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                is_land = bool(self.terrain_map[y, x])
                if is_land:
                    capacity = self.rng.uniform(
                        self.terrain_config.land_capacity_min,
                        self.terrain_config.land_capacity_max,
                    )
                    starting_resources = capacity * self.rng.uniform(0.35, 0.85)
                    regen_rate = self.rng.uniform(
                        self.terrain_config.land_regen_min,
                        self.terrain_config.land_regen_max,
                    )
                else:
                    capacity = 0.0
                    starting_resources = 0.0
                    regen_rate = 0.0

                cell = ResourceCell(
                    unique_id=self.next_id(),
                    model=self,
                    terrain_type="Land" if is_land else "Water",
                    resource_amount=starting_resources,
                    max_capacity=capacity,
                    regen_rate=regen_rate,
                )
                pos = (x, y)
                self.resource_cells[pos] = cell
                self.grid.place_agent(cell, pos)
                self.schedule.add(cell)

    def _seed_populations(self) -> None:
        land_positions = list(self.resource_cells_by_terrain("Land"))
        self.python_random.shuffle(land_positions)
        if self.initial_populations > len(land_positions):
            raise ValueError("Cannot place more starting populations than land cells.")

        for index, pos in enumerate(land_positions[: self.initial_populations]):
            population = Population(
                unique_id=self.next_id(),
                model=self,
                inhabitant_count=int(self.rng.integers(70, 120)),
                stockpile=float(self.rng.uniform(35.0, 80.0)),
                tech_level=0,
                lineage_color=self._lineage_color(index),
                traits=random_traits(self.rng),
            )
            self.grid.place_agent(population, pos)
            self.schedule.add(population)

    def _lineage_color(self, index: int) -> str:
        hue = (index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.92)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def resource_cells_by_terrain(self, terrain_type: str) -> Iterable[Position]:
        for pos, cell in self.resource_cells.items():
            if cell.terrain_type == terrain_type:
                yield pos

    def neighbor_positions(
        self,
        pos: Position,
        moore: bool = True,
        include_center: bool = False,
        radius: int = 1,
    ) -> List[Position]:
        if hasattr(self.grid, "get_neighborhood"):
            return list(
                self.grid.get_neighborhood(
                    pos,
                    moore=moore,
                    include_center=include_center,
                    radius=radius,
                )
            )
        raise RuntimeError("The configured grid does not support neighborhoods.")

    def resource_cells_near(
        self,
        pos: Position,
        include_center: bool = True,
    ) -> List[ResourceCell]:
        positions = self.neighbor_positions(pos, moore=True, include_center=include_center)
        return [self.resource_cells[position] for position in positions]

    def populations_near(self, pos: Position) -> List[Population]:
        agents = self.grid.get_cell_list_contents(
            self.neighbor_positions(pos, moore=True, include_center=False)
        )
        return [agent for agent in agents if isinstance(agent, Population)]

    def population_at(self, pos: Position) -> Optional[Population]:
        agents = self.grid.get_cell_list_contents([pos])
        for agent in agents:
            if isinstance(agent, Population):
                return agent
        return None

    def attempt_expansion(
        self,
        parent: Population,
        target_pos: Position,
        migrants: int,
    ) -> bool:
        target_cell = self.resource_cells.get(target_pos)
        if target_cell is None or not target_cell.is_land:
            return False

        occupant = self.population_at(target_pos)
        if occupant is None:
            child = Population(
                unique_id=self.next_id(),
                model=self,
                inhabitant_count=migrants,
                stockpile=parent.stockpile * 0.20,
                tech_level=parent.tech_level,
                lineage_color=parent.lineage_color,
                traits=parent.traits,
            )
            parent.stockpile *= 0.80
            self.grid.place_agent(child, target_pos)
            self.schedule.add(child)
            return True

        if occupant.lineage_color == parent.lineage_color:
            occupant.inhabitant_count += migrants
            return True

        land_desire_score = max(0.0, parent.inhabitant_count - self.expansion_threshold)
        deterrence_score = occupant.diplomatic_output() * 1.4 + occupant.military_output() * 0.6
        if deterrence_score > land_desire_score:
            return False

        attacker_strength = parent.military_output() * migrants
        defender_strength = occupant.military_output() * occupant.inhabitant_count
        if attacker_strength <= defender_strength:
            parent.inhabitant_count = max(1, parent.inhabitant_count - migrants)
            occupant.inhabitant_count = max(1, int(occupant.inhabitant_count * 0.85))
            return False

        self.grid.remove_agent(occupant)
        self.schedule.remove(occupant)
        child = Population(
            unique_id=self.next_id(),
            model=self,
            inhabitant_count=migrants,
            stockpile=parent.stockpile * 0.20,
            tech_level=parent.tech_level,
            lineage_color=parent.lineage_color,
            traits=parent.traits,
        )
        parent.stockpile *= 0.80
        self.grid.place_agent(child, target_pos)
        self.schedule.add(child)
        return True

    def max_tech_level(self) -> int:
        return max((population.tech_level for population in self.populations), default=0)

    def surviving_lineage_count(self) -> int:
        return len({population.lineage_color for population in self.populations})

    def dominant_trait(self) -> str:
        totals = {key: 0.0 for key in ("military_pct", "economic_pct", "diplomatic_pct", "tech_pct")}
        for population in self.populations:
            for key, value in population.traits.items():
                totals[key] += value * population.inhabitant_count
        if not totals or sum(totals.values()) <= 0:
            return "none"
        return max(totals, key=totals.get).replace("_pct", "")

    def step(self) -> None:
        self.schedule.step()
        self.datacollector.collect(self)

    def render_map(self, output_path: Optional[str] = None, show: bool = False):
        terrain_rgb = np.zeros((self.height, self.width, 3), dtype=float)
        terrain_rgb[self.terrain_map] = (0.18, 0.55, 0.24)
        terrain_rgb[~self.terrain_map] = (0.15, 0.35, 0.78)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(terrain_rgb, origin="lower")

        for population in self.populations:
            x, y = population.pos
            ax.scatter(
                [x],
                [y],
                s=max(55, min(210, population.inhabitant_count * 1.5)),
                color=population.lineage_color,
                edgecolors="white",
                linewidths=1.2,
                marker="o",
                zorder=3,
            )

        ax.set_title("Initialized ABM World")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
