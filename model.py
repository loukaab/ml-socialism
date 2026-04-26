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
DISPLAY_MAP_MODES = (
    "terrain",
    "arable",
    "raw",
    "manufactories",
    "tech",
    "diplo",
    "physical",
    "devastation",
)
MAP_MODES = DISPLAY_MAP_MODES + ("resources",)
ENVIRONMENTAL_MAP_MODES = {"arable", "raw", "resources", "manufactories", "devastation"}
STAT_METRICS = (
    {"key": "step", "label": "Step", "format": "int", "graph": True},
    {"key": "inhabitants", "label": "Inhabitants", "format": "int", "graph": True},
    {"key": "occupied_tiles", "label": "Tiles", "format": "int", "graph": True},
    {"key": "gdp", "label": "GDP", "format": "float", "graph": True},
    {"key": "gdp_per_capita", "label": "GDP / Capita", "format": "float", "graph": True},
    {"key": "food_stockpile", "label": "Food", "format": "float", "graph": True},
    {"key": "refined_stockpile", "label": "Refined", "format": "float", "graph": True},
    {"key": "raw_stockpile", "label": "Raw Stock", "format": "float", "graph": True},
    {"key": "food_produced", "label": "Food Prod.", "format": "float", "graph": True},
    {"key": "raw_extracted", "label": "Raw Extracted", "format": "float", "graph": True},
    {"key": "refined_produced", "label": "Refined Prod.", "format": "float", "graph": True},
    {"key": "food_produced_per_capita", "label": "Food / Capita", "format": "float", "graph": True},
    {"key": "raw_extracted_per_capita", "label": "Raw / Capita", "format": "float", "graph": True},
    {"key": "refined_produced_per_capita", "label": "Refined / Capita", "format": "float", "graph": True},
    {"key": "farmers", "label": "Farmers", "format": "int", "graph": True},
    {"key": "extractors", "label": "Extractors", "format": "int", "graph": True},
    {"key": "manufacturers", "label": "Manufacturers", "format": "int", "graph": True},
    {"key": "artisans", "label": "Artisans", "format": "int", "graph": True},
    {"key": "births", "label": "Births", "format": "int", "graph": True},
    {"key": "manufactories", "label": "Manufactories", "format": "int", "graph": True},
    {"key": "avg_devastation", "label": "Avg Devastation", "format": "float", "graph": True},
    {"key": "max_devastation", "label": "Max Devastation", "format": "float", "graph": True},
    {"key": "max_tech", "label": "Max Tech", "format": "int", "graph": True},
    {"key": "avg_tech", "label": "Avg Tech", "format": "float", "graph": True},
    {"key": "military_investment", "label": "Military Share", "format": "percent", "graph": True},
    {"key": "economic_investment", "label": "Economic Share", "format": "percent", "graph": True},
    {"key": "diplomatic_investment", "label": "Diplomatic Share", "format": "percent", "graph": True},
    {"key": "tech_investment", "label": "Tech Share", "format": "percent", "graph": True},
    {"key": "expansions", "label": "Expansions", "format": "int", "graph": True},
    {"key": "attacks", "label": "Attacks", "format": "int", "graph": True},
    {"key": "conquests", "label": "Conquests", "format": "int", "graph": True},
)


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
        collect_agent_records: bool = False,
    ) -> None:
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self.collect_agent_records = collect_agent_records
        self._model_records: List[Dict[str, object]] = []
        self._agent_records: List[Dict[str, object]] = []
        self.latest_model_record: Dict[str, object] = {}

    def collect(self, model) -> None:
        step = getattr(model.schedule, "steps", 0)
        model_row = {"Step": step}
        for name, reporter in self.model_reporters.items():
            model_row[name] = reporter(model)
        self._model_records.append(model_row)
        self.latest_model_record = model_row

        if not self.collect_agent_records:
            return
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
        collect_agent_records: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.initial_populations = initial_populations
        self.terrain_config = terrain
        self.economy_config = economy
        self.collect_agent_records = collect_agent_records
        self.rng = np.random.default_rng(seed)
        self.python_random = random.Random(seed)

        self.grid = GridClass(width, height, torus=False)
        self.schedule = SchedulerClass(self)
        self._next_id = 1
        self.resource_cells: Dict[Position, ResourceCell] = {}
        self.population_agents: List[Population] = []
        self.population_by_pos: Dict[Position, Population] = {}
        self._neighborhood_cache: Dict[Tuple[Position, bool, bool, int], Tuple[Position, ...]] = {}
        self.stats_history: List[Dict[str, object]] = []
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
        self.collect_stats_snapshot()

    def next_id(self) -> int:
        unique_id = self._next_id
        self._next_id += 1
        return unique_id

    @property
    def populations(self) -> List[Population]:
        self._sync_population_index_if_needed()
        return list(self.population_agents)

    def _sync_population_index_if_needed(self) -> None:
        expected_agents = len(self.resource_cells) + len(self.population_agents)
        if len(self.schedule.agents) != expected_agents:
            self._rebuild_population_indexes()

    def _rebuild_population_indexes(self) -> None:
        populations = [
            agent for agent in self.schedule.agents if isinstance(agent, Population)
        ]
        self.population_agents = populations
        self.population_by_pos = {
            population.pos: population
            for population in populations
            if population.pos is not None
        }
        for nation in self.nations:
            nation.population_agents.clear()
        for population in populations:
            if population.nation is not None:
                population.nation.add_population(population)

    def population_snapshot(self) -> List[Population]:
        return sorted(self.population_agents, key=lambda item: item.unique_id)

    def register_population(self, population: Population, pos: Position) -> None:
        if pos in self.population_by_pos:
            raise ValueError(f"Population already occupies {pos}")
        self.grid.place_agent(population, pos)
        self.schedule.add(population)
        self.population_agents.append(population)
        self.population_by_pos[pos] = population
        if population.nation is not None:
            population.nation.add_population(population)

    def surviving_nations(self) -> List[NationManager]:
        return [
            nation
            for nation in self.nations
            if not nation.defeated and nation.population_agents
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
            self.register_population(population, pos)

    def _lineage_color(self, index: int) -> str:
        if index < len(LINEAGE_COLORS):
            return LINEAGE_COLORS[index]

        hue = (0.83 + index * 0.071) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _build_datacollector(self):
        model_reporters = {
            "MaxTech": lambda model: model.max_tech_level(),
            "SurvivingLineages": lambda model: model.surviving_lineage_count(),
            "DominantTrait": lambda model: model.dominant_trait(),
            "PopulationAgents": lambda model: len(model.population_agents),
            "TotalInhabitants": lambda model: model.total_inhabitants(),
            "OccupiedTiles": lambda model: len(model.population_agents),
            "ExpansionEvents": lambda model: model.expansion_events,
            "AttackEvents": lambda model: model.attack_events,
            "ConquestEvents": lambda model: model.conquest_events,
            "GDP": lambda model: model.total_gdp(),
            "FoodStockpile": lambda model: model.total_food_stockpile(),
            "RefinedStockpile": lambda model: model.total_refined_stockpile(),
            "Manufactories": lambda model: model.total_manufactories(),
        }
        agent_reporters = {
            "Inhabitants": lambda agent: getattr(agent, "inhabitant_count", None),
            "TechLevel": lambda agent: getattr(agent, "tech_level", None),
            "LineageColor": lambda agent: getattr(agent, "lineage_color", None),
            "NationID": lambda agent: getattr(getattr(agent, "nation", None), "unique_id", None),
            "Farmers": lambda agent: getattr(agent, "last_farmers", None),
            "Extractors": lambda agent: getattr(agent, "last_extractors", None),
            "Manufacturers": lambda agent: getattr(agent, "last_manufacturers", None),
            "Artisans": lambda agent: getattr(agent, "last_artisans", None),
        }
        try:
            return CollectorClass(
                model_reporters=model_reporters,
                agent_reporters=agent_reporters,
                collect_agent_records=self.collect_agent_records,
            )
        except TypeError:
            return CollectorClass(
                model_reporters=model_reporters,
                agent_reporters=agent_reporters if self.collect_agent_records else {},
            )

    def available_stat_metrics(self) -> List[Dict[str, object]]:
        return [dict(metric) for metric in STAT_METRICS]

    def current_stats_snapshot(self) -> Dict[str, object]:
        if not self.stats_history:
            return self.collect_stats_snapshot()
        return self.stats_history[-1]

    def collect_stats_snapshot(self) -> Dict[str, object]:
        step = int(getattr(self.schedule, "steps", 0))
        snapshot: Dict[str, object] = {
            "step": step,
            "global": self._stats_row(
                step=step,
                scope="global",
                label="All",
                color="#e5e7eb",
                populations=list(self.population_agents),
                nation=None,
                defeated=False,
            ),
            "lineages": {},
        }

        lineages = snapshot["lineages"]
        for index, nation in enumerate(self.nations, start=1):
            lineages[nation.unique_id] = self._stats_row(
                step=step,
                scope="lineage",
                label=f"Lineage {index}",
                color=nation.lineage_color,
                populations=list(nation.population_agents),
                nation=nation,
                defeated=nation.defeated,
            )

        self.stats_history.append(snapshot)
        return snapshot

    def _stats_row(
        self,
        step: int,
        scope: str,
        label: str,
        color: str,
        populations: List[Population],
        nation: Optional[NationManager],
        defeated: bool,
    ) -> Dict[str, object]:
        active_populations = [] if defeated else populations
        inhabitants = sum(population.inhabitant_count for population in active_populations)
        occupied_tiles = len(active_populations)
        raw_stockpile = self._raw_stockpile_for(active_populations, global_scope=nation is None)
        manufactories = self._manufactories_for(active_populations, global_scope=nation is None)
        avg_devastation, max_devastation = self._devastation_stats_for(
            active_populations,
            global_scope=nation is None,
        )
        food_produced = self._food_produced_for(active_populations, nation)
        refined_produced = self._refined_produced_for(active_populations, nation)
        raw_extracted = sum(population.last_raw_extracted for population in active_populations)
        farmers = sum(population.last_farmers for population in active_populations)
        extractors = sum(population.last_extractors for population in active_populations)
        manufacturers = sum(population.last_manufacturers for population in active_populations)
        artisans = sum(population.last_artisans for population in active_populations)
        births = sum(population.last_new_inhabitants for population in active_populations)
        max_tech = max((population.tech_level for population in active_populations), default=0)
        avg_tech = self._weighted_average(
            ((population.tech_level, population.inhabitant_count) for population in active_populations)
        )
        investments = self._weighted_investment_shares(active_populations)
        gdp = self.total_gdp() if nation is None else (0.0 if defeated else nation.gdp)
        food_stockpile = (
            self.total_food_stockpile()
            if nation is None
            else (0.0 if defeated else nation.food_stockpile)
        )
        refined_stockpile = (
            self.total_refined_stockpile()
            if nation is None
            else (0.0 if defeated else nation.refined_stockpile)
        )
        per_capita_denominator = max(1, inhabitants)

        return {
            "step": step,
            "scope": scope,
            "nation_id": None if nation is None else nation.unique_id,
            "label": label,
            "color": color,
            "lineage_color": color,
            "status": "Defeated" if defeated else "Active",
            "defeated": defeated,
            "capital_pos": None if nation is None else nation.capital_pos,
            "inhabitants": inhabitants,
            "occupied_tiles": occupied_tiles,
            "gdp": gdp,
            "gdp_per_capita": gdp / per_capita_denominator,
            "manufactories": manufactories,
            "food_stockpile": food_stockpile,
            "refined_stockpile": refined_stockpile,
            "raw_stockpile": raw_stockpile,
            "food_produced": food_produced,
            "raw_extracted": raw_extracted,
            "refined_produced": refined_produced,
            "food_produced_per_capita": food_produced / per_capita_denominator,
            "raw_extracted_per_capita": raw_extracted / per_capita_denominator,
            "refined_produced_per_capita": refined_produced / per_capita_denominator,
            "farmers": farmers,
            "extractors": extractors,
            "manufacturers": manufacturers,
            "artisans": artisans,
            "births": births,
            "max_tech": max_tech,
            "avg_tech": avg_tech,
            "avg_devastation": avg_devastation,
            "max_devastation": max_devastation,
            "military_investment": investments["military"],
            "economic_investment": investments["economic"],
            "diplomatic_investment": investments["diplomatic"],
            "tech_investment": investments["tech"],
            "expansions": self.expansion_events if nation is None else None,
            "attacks": self.attack_events if nation is None else None,
            "conquests": self.conquest_events if nation is None else None,
        }

    def _raw_stockpile_for(
        self,
        populations: List[Population],
        global_scope: bool = False,
    ) -> float:
        if global_scope:
            return sum(cell.raw_goods_stockpile for cell in self.resource_cells.values())
        total = 0.0
        for population in populations:
            cell = self.resource_cell_at(population.pos)
            if cell is not None:
                total += cell.raw_goods_stockpile
        return total

    def _manufactories_for(
        self,
        populations: List[Population],
        global_scope: bool = False,
    ) -> int:
        if global_scope:
            return self.total_manufactories()
        total = 0
        for population in populations:
            cell = self.resource_cell_at(population.pos)
            if cell is not None and cell.manufactory_level > 0:
                total += 1
        return total

    def _devastation_stats_for(
        self,
        populations: List[Population],
        global_scope: bool = False,
    ) -> Tuple[float, float]:
        if global_scope:
            values = [
                cell.devastation
                for cell in self.resource_cells.values()
                if cell.is_land
            ]
        else:
            values = []
            for population in populations:
                cell = self.resource_cell_at(population.pos)
                if cell is not None and cell.is_land:
                    values.append(cell.devastation)
        if not values:
            return 0.0, 0.0
        return sum(values) / len(values), max(values)

    def _food_produced_for(
        self,
        populations: List[Population],
        nation: Optional[NationManager],
    ) -> float:
        if nation is None:
            return sum(population.last_food_produced for population in populations)
        return nation.total_food_produced

    def _refined_produced_for(
        self,
        populations: List[Population],
        nation: Optional[NationManager],
    ) -> float:
        if nation is None:
            return sum(population.last_refined_produced for population in populations)
        return nation.total_refined_produced

    def _weighted_investment_shares(self, populations: List[Population]) -> Dict[str, float]:
        totals = {"military": 0.0, "economic": 0.0, "diplomatic": 0.0, "tech": 0.0}
        weight_total = sum(population.inhabitant_count for population in populations)
        if weight_total <= 0:
            return totals

        for population in populations:
            weight = population.inhabitant_count
            proportions = population.investment_proportions
            for key in totals:
                totals[key] += proportions[key] * weight
        return {key: value / weight_total for key, value in totals.items()}

    def _weighted_average(self, values: Iterable[Tuple[float, float]]) -> float:
        total = 0.0
        weight_total = 0.0
        for value, weight in values:
            total += value * weight
            weight_total += weight
        if weight_total <= 0:
            return 0.0
        return total / weight_total

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
        key = (pos, moore, include_center, radius)
        cached = self._neighborhood_cache.get(key)
        if cached is None:
            cached = tuple(
                self._compute_neighbor_positions(
                    pos,
                    moore=moore,
                    include_center=include_center,
                    radius=radius,
                )
            )
            self._neighborhood_cache[key] = cached
        return list(cached)

    def _compute_neighbor_positions(
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
                if 0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height:
                    neighbors.append(next_pos)
        return neighbors

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

    def devastation_at(self, pos: Position) -> float:
        cell = self.resource_cell_at(pos)
        if cell is None:
            return 0.0
        return float(cell.devastation)

    def devastation_multiplier_at(self, pos: Position) -> float:
        cell = self.resource_cell_at(pos)
        if cell is None:
            return 0.0
        return cell.production_multiplier

    def add_tile_devastation(self, pos: Position, amount: float) -> None:
        cell = self.resource_cell_at(pos)
        if cell is None or not cell.is_land:
            return
        cell.add_devastation(amount)

    def manufactory_cost_for_cell(self, cell: ResourceCell) -> float:
        return self.economy_config.manufactory_cost * (1.0 + max(0.0, cell.devastation))

    def carrying_capacity_at(self, pos: Position) -> float:
        return self.food_growth_capacity_at(pos)

    def max_farmers_at(self, pos: Position) -> int:
        cell = self.resource_cell_at(pos)
        if cell is None or not cell.is_land:
            return 0
        return max(0, int(cell.arable_value * self.economy_config.farmer_slot_scale))

    def max_food_output_at(self, pos: Position, tech_multiplier: float = 1.0) -> float:
        farmers = self.max_farmers_at(pos)
        return (
            farmers
            * self.economy_config.food_per_farmer
            * max(0.0, tech_multiplier)
            * self.devastation_multiplier_at(pos)
        )

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
        return [
            population
            for neighbor_pos in self.neighbor_positions(pos, moore=True, include_center=False)
            if (population := self.population_by_pos.get(neighbor_pos)) is not None
        ]

    def population_at(self, pos: Position) -> Optional[Population]:
        self._sync_population_index_if_needed()
        population = self.population_by_pos.get(pos)
        if population is not None:
            return population
        agents = self.grid.get_cell_list_contents([pos])
        for agent in agents:
            if isinstance(agent, Population):
                self.population_by_pos[pos] = agent
                if agent not in self.population_agents:
                    self.population_agents.append(agent)
                if agent.nation is not None:
                    agent.nation.add_population(agent)
                return agent
        return None

    def attempt_expansion(
        self,
        parent: Population,
        target_pos: Position,
        migrants: int,
    ) -> bool:
        self._sync_population_index_if_needed()
        target_cell = self.resource_cells.get(target_pos)
        if target_cell is None or not target_cell.is_land:
            return False

        if target_pos in self.population_by_pos:
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
        self.register_population(child, target_pos)
        self.expansion_events += 1
        return True

    def best_expansion_targets(self, pos: Position) -> List[Position]:
        candidates = []
        for target_pos in self.neighbor_positions(pos, moore=True):
            target_cell = self.resource_cells.get(target_pos)
            if target_cell is None or not target_cell.is_land:
                continue
            if target_pos in self.population_by_pos:
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

        if old_nation is not None and old_nation is not new_nation:
            old_nation.remove_population(target)
        target.nation = new_nation
        if new_nation is not None and old_nation is not new_nation:
            new_nation.add_population(target)
        target._lineage_color = new_nation.lineage_color if new_nation is not None else target._lineage_color
        target._set_beliefs(new_beliefs)
        target.tech_level = new_tech_level
        target.inhabitant_count = max(1, int(new_inhabitants))
        target.growth_remainder = 0.0
        target.food_deficit_ticks = 0
        target.refined_deficit_ticks = 0
        target.refined_growth_multiplier = 1.0
        self.conquest_events += 1
        self.add_tile_devastation(
            target_pos,
            self.economy_config.devastation_capture_increase,
        )
        target_cell = self.resource_cell_at(target_pos)
        if (
            target_cell is not None
            and target_cell.manufactory_level > 0
            and self.rng.random() < self.economy_config.manufactory_destruction_chance
        ):
            target_cell.manufactory_level = 0

        if old_nation is None or new_nation is None or old_nation is new_nation:
            return

        if not old_nation.controlled_populations(self):
            old_nation.mark_defeated_by(new_nation)
            return

        if old_nation.capital_pos == target_pos:
            old_nation.reassign_capital(self)

    def max_tech_level(self) -> int:
        return max((population.tech_level for population in self.population_agents), default=0)

    def surviving_lineage_count(self) -> int:
        return len(self.surviving_nations())

    def total_inhabitants(self) -> int:
        return sum(population.inhabitant_count for population in self.population_agents)

    def dominant_trait(self) -> str:
        totals = {key: 0.0 for key in ("military", "economic", "diplomatic", "tech")}
        for population in self.population_agents:
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
        self._sync_population_index_if_needed()
        self._age_attack_arrows()
        self._recover_devastation()
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
        self.collect_stats_snapshot()

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
        for population in self.population_agents:
            population.reset_tick_production()

    def _run_population_production(self) -> None:
        for population in self.population_snapshot():
            population.produce_goods()

    def _run_population_consumption_and_growth(self) -> None:
        for population in self.population_snapshot():
            population.consume_goods()
            population.diffuse_tech()
            population.advance_tech()
            population.drift_traits()

    def _run_conflict_and_expansion(self) -> None:
        active = set(self.population_agents)
        for population in self.population_snapshot():
            if population in active:
                population.maybe_attack_neighbor()
        active = set(self.population_agents)
        for population in self.population_snapshot():
            if population in active:
                population.expand_or_migrate()

    def _run_nation_investment(self) -> None:
        for nation in self.surviving_nations():
            nation.invest_in_manufactory(self)

    def _recover_devastation(self) -> None:
        for cell in self.resource_cells.values():
            cell.recover_devastation()

    def redistribute_local_raw_goods(self) -> None:
        deltas = {pos: 0.0 for pos in self.resource_cells}
        for source_population in self.population_agents:
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
                recipient_population = self.population_by_pos.get(pos)
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
            for population in self.population_agents:
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
                low=np.array((0.83, 0.91, 0.52)),
                high=np.array((0.07, 0.37, 0.15)),
            )
        if map_mode == "raw":
            return self.scalar_rgb_array(
                self.raw_goods_map,
                low=np.array((0.88, 0.79, 0.60)),
                high=np.array((0.27, 0.18, 0.09)),
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
        if map_mode == "devastation":
            values = np.zeros((self.height, self.width), dtype=float)
            max_devastation = max(
                (
                    cell.devastation
                    for cell in self.resource_cells.values()
                    if cell.is_land
                ),
                default=0.0,
            )
            for (x, y), cell in self.resource_cells.items():
                if max_devastation > 0:
                    values[y, x] = min(1.0, max(0.0, cell.devastation / max_devastation))
            return self.scalar_rgb_array(
                values,
                low=np.array((0.00, 0.36, 0.13)),
                high=np.array((1.00, 0.09, 0.09)),
            )

        return terrain_rgb

    def scalar_rgb_array(self, values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        rgb = np.zeros((self.height, self.width, 3), dtype=float)
        rgb[~self.terrain_map] = (0.15, 0.35, 0.78)
        scalar_rgb = low + values[..., None] * (high - low)
        rgb[self.terrain_map] = scalar_rgb[self.terrain_map]
        return rgb

    def global_max_population(self) -> int:
        return max((population.inhabitant_count for population in self.population_agents), default=1)

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
