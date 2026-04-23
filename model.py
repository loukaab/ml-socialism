"""World model, terrain generation, scheduling, and rendering."""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import Population, ResourceCell, random_traits

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

LINEAGE_COLORS = (
    "#e83f6f",
    "#a855f7",
    "#f43f5e",
    "#ec4899",
    "#8b5cf6",
    "#d946ef",
    "#fb7185",
    "#c026d3",
    "#7c3aed",
    "#be123c",
    "#f0abfc",
    "#881337",
)

MIN_POPULATION_BRIGHTNESS = 0.35
POPULATION_BRIGHTNESS_GAMMA = 0.65


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
    resource_smoothing_passes: int = 7
    carrying_capacity_min: float = 90.0
    carrying_capacity_max: float = 520.0


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
        self.expansion_events = 0

        self.tech_threshold = 35.0
        self.tech_diffusion_rate = 0.35
        self.trait_drift_rate = 0.35
        self.expansion_pressure_threshold = 0.68
        self.migration_fraction = 0.22
        self.minimum_migrants = 14
        self.population_growth_rate = 0.085

        self.terrain_map = self._generate_terrain()
        self.resource_map = self._generate_resource_map()
        self.carrying_capacity_map = self._generate_carrying_capacity_map()
        self._populate_resource_layer()
        self._seed_populations()

        self.datacollector = CollectorClass(
            model_reporters={
                "MaxTech": lambda model: model.max_tech_level(),
                "SurvivingLineages": lambda model: model.surviving_lineage_count(),
                "DominantTrait": lambda model: model.dominant_trait(),
                "PopulationAgents": lambda model: len(model.populations),
                "TotalInhabitants": lambda model: model.total_inhabitants(),
                "OccupiedTiles": lambda model: len(model.populations),
                "ExpansionEvents": lambda model: model.expansion_events,
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

    def _smooth_noise(self, smoothing_passes: int) -> np.ndarray:
        noise = self.rng.random((self.height, self.width))
        for _ in range(smoothing_passes):
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
        return noise

    def _generate_terrain(self) -> np.ndarray:
        noise = self._smooth_noise(self.terrain_config.smoothing_passes)
        threshold = np.quantile(noise, 1.0 - self.terrain_config.land_fraction)
        return noise >= threshold

    def _generate_resource_map(self) -> np.ndarray:
        resource_noise = self._smooth_noise(self.terrain_config.resource_smoothing_passes)
        land_values = resource_noise[self.terrain_map]
        resource_map = np.zeros((self.height, self.width), dtype=float)
        if land_values.size == 0:
            return resource_map

        low = float(land_values.min())
        high = float(land_values.max())
        if high == low:
            normalized = np.ones_like(resource_noise)
        else:
            normalized = (resource_noise - low) / (high - low)
        resource_map[self.terrain_map] = np.clip(normalized[self.terrain_map], 0.0, 1.0)
        return resource_map

    def _generate_carrying_capacity_map(self) -> np.ndarray:
        minimum = self.terrain_config.carrying_capacity_min
        maximum = self.terrain_config.carrying_capacity_max
        capacity = minimum + self.resource_map * (maximum - minimum)
        capacity[~self.terrain_map] = 0.0
        return capacity

    def _populate_resource_layer(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                is_land = bool(self.terrain_map[y, x])
                resource_value = float(self.resource_map[y, x])
                capacity = float(self.carrying_capacity_map[y, x])

                cell = ResourceCell(
                    unique_id=self.next_id(),
                    model=self,
                    terrain_type="Land" if is_land else "Water",
                    resource_value=resource_value,
                    carrying_capacity=capacity,
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
                inhabitant_count=int(self.rng.integers(35, 80)),
                stockpile=float(self.rng.uniform(35.0, 80.0)),
                tech_level=0,
                lineage_color=self._lineage_color(index),
                traits=random_traits(self.rng),
            )
            self.grid.place_agent(population, pos)
            self.schedule.add(population)

    def _lineage_color(self, index: int) -> str:
        if index < len(LINEAGE_COLORS):
            return LINEAGE_COLORS[index]

        hue = (0.83 + index * 0.071) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
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

    def resource_cell_at(self, pos: Position) -> Optional[ResourceCell]:
        return self.resource_cells.get(pos)

    def resource_value_at(self, pos: Position) -> float:
        x, y = pos
        return float(self.resource_map[y, x])

    def carrying_capacity_at(self, pos: Position) -> float:
        x, y = pos
        return float(self.carrying_capacity_map[y, x])

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
        if occupant is not None:
            return False

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
        self.expansion_events += 1
        return True

    def best_expansion_targets(self, pos: Position) -> List[Position]:
        candidates = []
        for target_pos in self.neighbor_positions(pos, moore=True):
            target_cell = self.resource_cells.get(target_pos)
            if target_cell is None or not target_cell.is_land:
                continue
            if self.population_at(target_pos) is not None:
                continue
            candidates.append(target_pos)

        self.rng.shuffle(candidates)
        candidates.sort(
            key=lambda candidate: self.carrying_capacity_at(candidate),
            reverse=True,
        )
        return candidates

    def max_tech_level(self) -> int:
        return max((population.tech_level for population in self.populations), default=0)

    def surviving_lineage_count(self) -> int:
        return len({population.lineage_color for population in self.populations})

    def total_inhabitants(self) -> int:
        return sum(population.inhabitant_count for population in self.populations)

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

    def render_map(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
        resource_overlay: bool = False,
    ):
        terrain_rgb = self.render_rgb_array(resource_overlay=resource_overlay)
        for population in self.populations:
            x, y = population.pos
            terrain_rgb[y, x] = self.population_rgb(population)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(terrain_rgb, origin="lower")

        title = "ABM World - Resource Overlay" if resource_overlay else "ABM World"
        ax.set_title(title)
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

    def render_rgb_array(self, resource_overlay: bool = False) -> np.ndarray:
        terrain_rgb = np.zeros((self.height, self.width, 3), dtype=float)
        terrain_rgb[self.terrain_map] = (0.18, 0.55, 0.24)
        terrain_rgb[~self.terrain_map] = (0.15, 0.35, 0.78)
        if not resource_overlay:
            return terrain_rgb

        low_resource = np.array((0.38, 0.27, 0.12))
        high_resource = np.array((0.98, 0.84, 0.22))
        resource_rgb = low_resource + self.resource_map[..., None] * (
            high_resource - low_resource
        )
        terrain_rgb[self.terrain_map] = (
            terrain_rgb[self.terrain_map] * 0.35
            + resource_rgb[self.terrain_map] * 0.65
        )
        return terrain_rgb

    def population_rgb(self, population) -> np.ndarray:
        lineage = population.lineage_color.lstrip("#")
        rgb = np.array(
            [
                int(lineage[0:2], 16) / 255.0,
                int(lineage[2:4], 16) / 255.0,
                int(lineage[4:6], 16) / 255.0,
            ]
        )
        capacity = max(1.0, self.carrying_capacity_at(population.pos))
        fullness = min(1.0, max(0.0, population.inhabitant_count / capacity))
        curved = fullness ** POPULATION_BRIGHTNESS_GAMMA
        brightness = 1.0 - curved * (1.0 - MIN_POPULATION_BRIGHTNESS)
        return rgb * brightness
