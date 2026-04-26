"""World orchestration for the phase-2 macroeconomic ABM."""

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

from agents import Population, ResourceCell
from core.economy import EconomyConfig, NationManager
from core.geography import TerrainConfig, generate_geography_maps

try:
    import mesa
except (ModuleNotFoundError, ImportError):
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
DISPLAY_MAP_MODES = ("terrain", "arable", "raw", "manufactories", "tech", "diplo", "physical")
MAP_MODES = DISPLAY_MAP_MODES + ("resources",)
ENVIRONMENTAL_MAP_MODES = {"arable", "raw", "resources", "manufactories"}


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
    """Container compatible with Mesa's old RandomActivation shape."""

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


@dataclass
class AttackArrow:
    source: Position
    target: Position
    remaining_steps: int = 3


def normalize_map_mode(map_mode: str) -> str:
    return "raw" if map_mode == "resources" else map_mode


class WorldModel(mesa.Model):
    """Macroeconomic world with nations, supply chains, and deterministic phases."""

    def __init__(
        self,
        width: int = 50,
        height: int = 35,
        initial_populations: int = 8,
        seed: Optional[int] = None,
        terrain: TerrainConfig = TerrainConfig(),
        economy: EconomyConfig = EconomyConfig(),
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.initial_populations = initial_populations
        self.terrain_config = terrain
        self.economy_config = economy
        self.rng = np.random.default_rng(seed)
        self.python_random = random.Random(seed)

        self.grid = GridClass(width, height, torus=False)
        self.schedule = SchedulerClass(self)
        self._next_id = 1
        self.resource_cells: Dict[Position, ResourceCell] = {}
        self.nations: List[NationManager] = []
        self.expansion_events = 0
        self.attack_events = 0
        self.conquest_events = 0
        self.attack_arrows: List[AttackArrow] = []
        self._pending_attack_arrows: List[Tuple[Position, Position]] = []

        self.tech_threshold = 35.0
        self.tech_diffusion_rate = 0.35
        self.trait_drift_rate = 0.01
        self.expansion_pressure_threshold = 0.68
        self.migration_fraction = 0.22
        self.minimum_migrants = 14
        self.population_growth_rate = 0.085
        self.attack_scale_constant = 0.05

        geography = generate_geography_maps(self.rng, width, height, terrain)
        self.terrain_map = geography.terrain_map
        self.arable_map = geography.arable_map
        self.raw_goods_map = geography.raw_goods_map
        self.resource_map = self.raw_goods_map
        self.carrying_capacity_map = geography.carrying_capacity_map

        self._populate_resource_layer()
        self._seed_populations()
        self.datacollector = self._build_datacollector()
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

    def surviving_nations(self) -> List[NationManager]:
        return [
            nation
            for nation in self.nations
            if not nation.defeated and nation.controlled_populations(self)
        ]

    def create_nation(
        self,
        lineage_color: str,
        capital_pos: Optional[Position],
        food_stockpile: float = 0.0,
        refined_stockpile: float = 0.0,
    ) -> NationManager:
        nation = NationManager(
            unique_id=self.next_id(),
            lineage_color=lineage_color,
            capital_pos=capital_pos,
            food_stockpile=float(food_stockpile),
            refined_stockpile=float(refined_stockpile),
        )
        self.nations.append(nation)
        return nation

    def nation_for_lineage(
        self,
        lineage_color: str,
        capital_pos: Optional[Position] = None,
    ) -> NationManager:
        for nation in self.nations:
            if not nation.defeated and nation.lineage_color == lineage_color:
                if nation.capital_pos is None and capital_pos is not None:
                    nation.capital_pos = capital_pos
                return nation
        return self.create_nation(lineage_color, capital_pos)

    def _populate_resource_layer(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                is_land = bool(self.terrain_map[y, x])
                pos = (x, y)
                cell = ResourceCell(
                    unique_id=self.next_id(),
                    model=self,
                    terrain_type="Land" if is_land else "Water",
                    resource_value=float(self.raw_goods_map[y, x]),
                    raw_goods_value=float(self.raw_goods_map[y, x]),
                    arable_value=float(self.arable_map[y, x]),
                    carrying_capacity=float(self.carrying_capacity_map[y, x]),
                )
                self.resource_cells[pos] = cell
                self.grid.place_agent(cell, pos)
                self.schedule.add(cell)

    def _seed_populations(self) -> None:
        land_positions = list(self.resource_cells_by_terrain("Land"))
        self.python_random.shuffle(land_positions)
        if self.initial_populations > len(land_positions):
            raise ValueError("Cannot place more starting populations than land cells.")

        for index, pos in enumerate(land_positions[: self.initial_populations]):
            initial_food = float(self.rng.uniform(35.0, 80.0))
            nation = self.create_nation(
                lineage_color=self._lineage_color(index),
                capital_pos=pos,
                food_stockpile=initial_food,
            )
            population = Population(
                unique_id=self.next_id(),
                model=self,
                inhabitant_count=int(self.rng.integers(35, 80)),
                stockpile=0.0,
                tech_level=0,
                nation=nation,
            )
            self.grid.place_agent(population, pos)
            self.schedule.add(population)

    def _lineage_color(self, index: int) -> str:
        if index < len(LINEAGE_COLORS):
            return LINEAGE_COLORS[index]

        hue = (0.83 + index * 0.071) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _build_datacollector(self):
        return CollectorClass(
            model_reporters={
                "MaxTech": lambda model: model.max_tech_level(),
                "SurvivingLineages": lambda model: model.surviving_lineage_count(),
                "DominantTrait": lambda model: model.dominant_trait(),
                "PopulationAgents": lambda model: len(model.populations),
                "TotalInhabitants": lambda model: model.total_inhabitants(),
                "OccupiedTiles": lambda model: len(model.populations),
                "ExpansionEvents": lambda model: model.expansion_events,
                "AttackEvents": lambda model: model.attack_events,
                "ConquestEvents": lambda model: model.conquest_events,
                "GDP": lambda model: model.total_gdp(),
                "FoodStockpile": lambda model: model.total_food_stockpile(),
                "RefinedStockpile": lambda model: model.total_refined_stockpile(),
                "Manufactories": lambda model: model.total_manufactories(),
            },
            agent_reporters={
                "Inhabitants": lambda agent: getattr(agent, "inhabitant_count", None),
                "TechLevel": lambda agent: getattr(agent, "tech_level", None),
                "LineageColor": lambda agent: getattr(agent, "lineage_color", None),
                "NationID": lambda agent: getattr(getattr(agent, "nation", None), "unique_id", None),
                "Farmers": lambda agent: getattr(agent, "last_farmers", None),
                "Extractors": lambda agent: getattr(agent, "last_extractors", None),
                "Manufacturers": lambda agent: getattr(agent, "last_manufacturers", None),
                "Artisans": lambda agent: getattr(agent, "last_artisans", None),
            },
        )

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
        return self.raw_goods_value_at(pos)

    def raw_goods_value_at(self, pos: Position) -> float:
        x, y = pos
        return float(self.raw_goods_map[y, x])

    def arable_value_at(self, pos: Position) -> float:
        x, y = pos
        return float(self.arable_map[y, x])

    def carrying_capacity_at(self, pos: Position) -> float:
        return self.food_growth_capacity_at(pos)

    def max_farmers_at(self, pos: Position) -> int:
        cell = self.resource_cell_at(pos)
        if cell is None or not cell.is_land:
            return 0
        return max(0, int(cell.arable_value * self.economy_config.farmer_slot_scale))

    def max_food_output_at(self, pos: Position, tech_multiplier: float = 1.0) -> float:
        farmers = self.max_farmers_at(pos)
        return farmers * self.economy_config.food_per_farmer * max(0.0, tech_multiplier)

    def food_growth_capacity_at(self, pos: Position, tech_multiplier: float = 1.0) -> float:
        denominator = (
            self.economy_config.food_need_per_person
            * self.economy_config.food_claim_multiplier
        )
        if denominator <= 0:
            return 0.0
        return self.max_food_output_at(pos, tech_multiplier=tech_multiplier) / denominator

    def food_growth_capacity_for_population(self, population: Population) -> float:
        if population.pos is None:
            return 0.0
        return self.food_growth_capacity_at(
            population.pos,
            tech_multiplier=population.tech_multiplier,
        )

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
            stockpile=0.0,
            tech_level=parent.tech_level,
            nation=parent.nation,
            beliefs=parent.beliefs,
        )
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

        self.python_random.shuffle(candidates)
        candidates.sort(
            key=lambda candidate: self.food_growth_capacity_at(candidate),
            reverse=True,
        )
        return candidates

    def handle_conquest(
        self,
        attacker: Population,
        target: Population,
        new_inhabitants: int,
        new_beliefs: Dict[str, float],
        new_tech_level: int,
    ) -> None:
        old_nation = target.nation
        new_nation = attacker.nation
        target_pos = target.pos

        target.nation = new_nation
        target._lineage_color = new_nation.lineage_color if new_nation is not None else target._lineage_color
        target._set_beliefs(new_beliefs)
        target.tech_level = new_tech_level
        target.inhabitant_count = max(1, int(new_inhabitants))
        target.growth_remainder = 0.0
        target.food_deficit_ticks = 0
        target.refined_deficit_ticks = 0
        target.refined_growth_multiplier = 1.0
        self.conquest_events += 1

        if old_nation is None or new_nation is None or old_nation is new_nation:
            return

        if not old_nation.controlled_populations(self):
            old_nation.mark_defeated_by(new_nation)
            return

        if old_nation.capital_pos == target_pos:
            old_nation.reassign_capital(self)

    def max_tech_level(self) -> int:
        return max((population.tech_level for population in self.populations), default=0)

    def surviving_lineage_count(self) -> int:
        return len(self.surviving_nations())

    def total_inhabitants(self) -> int:
        return sum(population.inhabitant_count for population in self.populations)

    def dominant_trait(self) -> str:
        totals = {key: 0.0 for key in ("military", "economic", "diplomatic", "tech")}
        for population in self.populations:
            for key, value in population.traits.items():
                totals[key] += value * population.inhabitant_count
        if not totals or sum(totals.values()) <= 0:
            return "none"
        return max(totals, key=totals.get)

    def total_gdp(self) -> float:
        return sum(nation.gdp for nation in self.surviving_nations())

    def total_food_stockpile(self) -> float:
        return sum(nation.food_stockpile for nation in self.surviving_nations())

    def total_refined_stockpile(self) -> float:
        return sum(nation.refined_stockpile for nation in self.surviving_nations())

    def total_manufactories(self) -> int:
        return sum(1 for cell in self.resource_cells.values() if cell.manufactory_level > 0)

    def max_manufactory_level(self) -> int:
        return 1 if self.total_manufactories() > 0 else 0

    def step(self) -> None:
        self._age_attack_arrows()
        self._reset_tick_state()
        self._run_population_production()
        if self._next_step_number() % self.economy_config.local_logistics_period == 0:
            self.redistribute_local_raw_goods()
        self._run_population_consumption_and_growth()
        self._run_conflict_and_expansion()
        self._run_nation_investment()
        self._flush_pending_attack_arrows()
        self._advance_schedule_clock()
        self.datacollector.collect(self)

    def _next_step_number(self) -> int:
        return int(getattr(self.schedule, "steps", 0)) + 1

    def _advance_schedule_clock(self) -> None:
        if hasattr(self.schedule, "steps"):
            self.schedule.steps += 1
        if hasattr(self.schedule, "time"):
            self.schedule.time += 1

    def _reset_tick_state(self) -> None:
        for nation in self.nations:
            nation.reset_tick()
        for cell in self.resource_cells.values():
            cell.reset_tick_production()
        for population in self.populations:
            population.reset_tick_production()

    def _run_population_production(self) -> None:
        for population in sorted(self.populations, key=lambda item: item.unique_id):
            population.produce_goods()

    def _run_population_consumption_and_growth(self) -> None:
        for population in sorted(self.populations, key=lambda item: item.unique_id):
            population.consume_goods()
            population.diffuse_tech()
            population.advance_tech()
            population.drift_traits()

    def _run_conflict_and_expansion(self) -> None:
        for population in sorted(list(self.populations), key=lambda item: item.unique_id):
            if population in self.populations:
                population.maybe_attack_neighbor()
        for population in sorted(list(self.populations), key=lambda item: item.unique_id):
            if population in self.populations:
                population.expand_or_migrate()

    def _run_nation_investment(self) -> None:
        for nation in self.surviving_nations():
            nation.invest_in_manufactory(self)

    def redistribute_local_raw_goods(self) -> None:
        deltas = {pos: 0.0 for pos in self.resource_cells}
        for source_population in self.populations:
            source_pos = source_population.pos
            source_cell = self.resource_cell_at(source_pos)
            if source_cell is None or source_cell.raw_goods_stockpile <= 0:
                continue

            recipients = []
            for pos in self.neighbor_positions(
                source_pos,
                moore=True,
                include_center=True,
                radius=1,
            ):
                recipient_population = self.population_at(pos)
                if recipient_population is None:
                    continue
                if recipient_population.nation is not source_population.nation:
                    continue
                recipient_cell = self.resource_cell_at(pos)
                if recipient_cell is None:
                    continue
                weight = recipient_cell.last_refined_produced
                if pos == source_pos:
                    weight *= self.economy_config.center_lps_weight
                recipients.append((pos, max(0.0, weight)))

            total_weight = sum(weight for _, weight in recipients)
            if total_weight <= 0:
                continue

            amount = source_cell.raw_goods_stockpile
            deltas[source_pos] -= amount
            for recipient_pos, weight in recipients:
                deltas[recipient_pos] += amount * (weight / total_weight)

        for pos, delta in deltas.items():
            if delta:
                cell = self.resource_cells[pos]
                cell.raw_goods_stockpile = max(0.0, cell.raw_goods_stockpile + delta)

    def register_attack_arrow(self, source: Position, target: Position) -> None:
        self._pending_attack_arrows.append((source, target))

    def _flush_pending_attack_arrows(self) -> None:
        for source, target in self._pending_attack_arrows:
            self.attack_arrows.append(
                AttackArrow(source=source, target=target, remaining_steps=3)
            )
        self._pending_attack_arrows.clear()

    def _age_attack_arrows(self) -> None:
        survivors: List[AttackArrow] = []
        for arrow in self.attack_arrows:
            arrow.remaining_steps -= 1
            if arrow.remaining_steps > 0:
                survivors.append(arrow)
        self.attack_arrows = survivors

    def render_map(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
        resource_overlay: bool = False,
        map_mode: str = "terrain",
    ):
        if resource_overlay:
            map_mode = "resources"
        if map_mode not in MAP_MODES:
            raise ValueError(f"Unknown map mode: {map_mode}")

        normalized_mode = normalize_map_mode(map_mode)
        terrain_rgb = self.render_rgb_array(map_mode=normalized_mode)
        global_max_population = self.global_max_population()
        if normalized_mode not in ENVIRONMENTAL_MAP_MODES:
            for population in self.populations:
                x, y = population.pos
                terrain_rgb[y, x] = self.population_rgb(
                    population,
                    map_mode=normalized_mode,
                    global_max_population=global_max_population,
                )

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(terrain_rgb, origin="lower")
        for nation in self.surviving_nations():
            if nation.capital_pos is None:
                continue
            x, y = nation.capital_pos
            ax.scatter(
                [x],
                [y],
                marker="*",
                s=95,
                c="#ffd84d",
                edgecolors="black",
                linewidths=0.8,
            )

        title = f"ABM World - {normalized_mode.title()} Map"
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

    def render_rgb_array(
        self,
        resource_overlay: bool = False,
        map_mode: str = "terrain",
    ) -> np.ndarray:
        if resource_overlay:
            map_mode = "raw"
        map_mode = normalize_map_mode(map_mode)

        terrain_rgb = np.zeros((self.height, self.width, 3), dtype=float)
        terrain_rgb[self.terrain_map] = (0.18, 0.55, 0.24)
        terrain_rgb[~self.terrain_map] = (0.15, 0.35, 0.78)

        if map_mode == "arable":
            return self.scalar_rgb_array(
                self.arable_map,
                low=np.array((0.35, 0.28, 0.16)),
                high=np.array((0.56, 0.82, 0.34)),
            )
        if map_mode == "raw":
            return self.scalar_rgb_array(
                self.raw_goods_map,
                low=np.array((0.38, 0.27, 0.12)),
                high=np.array((0.98, 0.84, 0.22)),
            )
        if map_mode == "manufactories":
            values = np.zeros((self.height, self.width), dtype=float)
            for (x, y), cell in self.resource_cells.items():
                values[y, x] = 1.0 if cell.manufactory_level > 0 else 0.0
            return self.scalar_rgb_array(
                values,
                low=np.array((0.22, 0.29, 0.32)),
                high=np.array((0.38, 0.84, 0.88)),
            )

        return terrain_rgb

    def scalar_rgb_array(self, values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        rgb = np.zeros((self.height, self.width, 3), dtype=float)
        rgb[~self.terrain_map] = (0.15, 0.35, 0.78)
        scalar_rgb = low + values[..., None] * (high - low)
        rgb[self.terrain_map] = scalar_rgb[self.terrain_map]
        return rgb

    def global_max_population(self) -> int:
        return max((population.inhabitant_count for population in self.populations), default=1)

    def population_rgb(
        self,
        population,
        map_mode: str = "terrain",
        global_max_population: Optional[int] = None,
    ) -> np.ndarray:
        if map_mode == "tech":
            return self.dark_investment_rgb(
                population.x_tech,
                max_value=0.3,
                light=np.array((222, 210, 255)) / 255.0,
                dark=np.array((44, 22, 92)) / 255.0,
            )
        if map_mode == "diplo":
            return self.dark_investment_rgb(
                population.y_dip,
                max_value=0.3,
                light=np.array((255, 212, 232)) / 255.0,
                dark=np.array((95, 18, 58)) / 255.0,
            )
        if map_mode == "physical":
            return self.physical_split_rgb(population.e_econ_ratio)

        lineage = population.lineage_color.lstrip("#")
        rgb = np.array(
            [
                int(lineage[0:2], 16) / 255.0,
                int(lineage[2:4], 16) / 255.0,
                int(lineage[4:6], 16) / 255.0,
            ]
        )
        max_population = max(1, global_max_population or self.global_max_population())
        if max_population <= 1:
            normalized = 0.0
        else:
            normalized = (population.inhabitant_count - 1) / (max_population - 1)
        normalized = min(1.0, max(0.0, normalized))
        curved = normalized ** POPULATION_BRIGHTNESS_GAMMA
        brightness = 1.0 - curved * (1.0 - MIN_POPULATION_BRIGHTNESS)
        return rgb * brightness

    def dark_investment_rgb(
        self,
        value: float,
        max_value: float,
        light: np.ndarray,
        dark: np.ndarray,
    ) -> np.ndarray:
        normalized = min(1.0, max(0.0, value / max_value))
        return light + normalized * (dark - light)

    def physical_split_rgb(self, e_econ_ratio: float) -> np.ndarray:
        ratio = min(1.0, max(0.0, e_econ_ratio))
        red = np.array((225, 42, 42)) / 255.0
        yellow = np.array((250, 220, 42)) / 255.0
        return red + ratio * (yellow - red)
