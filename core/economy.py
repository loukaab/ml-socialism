"""National stockpiles, GDP accounting, and infrastructure investment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agents.population import Population
    from model import Position, WorldModel


@dataclass(frozen=True)
class EconomyConfig:
    farmer_slot_scale: int = 120
    extractor_slot_scale: int = 80
    food_per_farmer: float = 4.0
    raw_per_extractor: float = 1.0
    manufacturer_jobs_per_level: int = 25
    artisan_raw_throughput: float = 1.0
    manufacturer_raw_throughput: float = 5.0
    food_need_per_person: float = 1.0
    refined_need_per_person: float = 0.04
    food_deficit_loss_rate: float = 0.02
    manufactory_cost: float = 250.0
    local_logistics_period: int = 10
    center_lps_weight: float = 1.5


@dataclass
class NationManager:
    unique_id: int
    lineage_color: str
    capital_pos: Optional["Position"]
    food_stockpile: float = 0.0
    refined_stockpile: float = 0.0
    total_food_produced: float = 0.0
    total_refined_produced: float = 0.0
    defeated: bool = False

    @property
    def gdp(self) -> float:
        return self.total_food_produced + 3.0 * self.total_refined_produced

    def reset_tick(self) -> None:
        self.total_food_produced = 0.0
        self.total_refined_produced = 0.0

    def add_production(self, food: float = 0.0, refined: float = 0.0) -> None:
        self.total_food_produced += max(0.0, float(food))
        self.total_refined_produced += max(0.0, float(refined))

    def controlled_populations(self, model: "WorldModel") -> List["Population"]:
        return [population for population in model.populations if population.nation is self]

    def controlled_positions(self, model: "WorldModel") -> list["Position"]:
        return [
            population.pos
            for population in self.controlled_populations(model)
            if population.pos is not None
        ]

    def reassign_capital(self, model: "WorldModel") -> None:
        populations = self.controlled_populations(model)
        if not populations:
            self.capital_pos = None
            return
        populations.sort(
            key=lambda population: (
                population.inhabitant_count,
                -(population.pos[1] if population.pos else 0),
                -(population.pos[0] if population.pos else 0),
            ),
            reverse=True,
        )
        self.capital_pos = populations[0].pos

    def mark_defeated_by(self, conqueror: "NationManager") -> None:
        if self is conqueror or self.defeated:
            return
        conqueror.food_stockpile += self.food_stockpile
        conqueror.refined_stockpile += self.refined_stockpile
        self.food_stockpile = 0.0
        self.refined_stockpile = 0.0
        self.capital_pos = None
        self.defeated = True

    def invest_in_manufactory(self, model: "WorldModel") -> bool:
        cost = model.economy_config.manufactory_cost
        if self.defeated or self.refined_stockpile < cost:
            return False

        candidates = []
        for population in self.controlled_populations(model):
            if population.last_artisans <= 0:
                continue
            cell = model.resource_cell_at(population.pos)
            if cell is None:
                continue
            candidates.append((population.last_artisans, population.inhabitant_count, population.pos, cell))

        if not candidates:
            return False

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, _, cell = candidates[0]
        self.refined_stockpile -= cost
        cell.manufactory_level += 1
        return True
