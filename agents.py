"""Agent definitions for the phase-1 economic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

try:
    import mesa
except ModuleNotFoundError:
    class _Agent:
        def __init__(self, unique_id=None, model=None):
            self.unique_id = unique_id
            self.model = model
            self.pos = None

    mesa = SimpleNamespace(Agent=_Agent)


Position = Tuple[int, int]
TraitMap = Dict[str, float]

TRAIT_KEYS = ("military_pct", "economic_pct", "diplomatic_pct", "tech_pct")


def normalize_traits(traits: TraitMap) -> TraitMap:
    """Return trait percentages normalized to exactly 100 total points."""
    clipped = {key: max(0.0, float(traits.get(key, 0.0))) for key in TRAIT_KEYS}
    total = sum(clipped.values())
    if total <= 0:
        equal_share = 100.0 / len(TRAIT_KEYS)
        return {key: equal_share for key in TRAIT_KEYS}
    return {key: (value / total) * 100.0 for key, value in clipped.items()}


def random_traits(rng: np.random.Generator) -> TraitMap:
    raw = rng.dirichlet(np.ones(len(TRAIT_KEYS)))
    return {key: float(value * 100.0) for key, value in zip(TRAIT_KEYS, raw)}


@dataclass(frozen=True)
class Allocation:
    military: float
    economic: float
    diplomatic: float
    tech: float


class ResourceCell(mesa.Agent):
    """Static environmental agent representing land or water on a map tile."""

    def __init__(
        self,
        unique_id: int,
        model,
        terrain_type: str,
        resource_value: float = 0.0,
        carrying_capacity: float = 0.0,
    ) -> None:
        super().__init__(unique_id, model)
        self.terrain_type = terrain_type
        self.resource_value = float(resource_value)
        self.carrying_capacity = float(carrying_capacity)

    @property
    def is_land(self) -> bool:
        return self.terrain_type == "Land"

    @property
    def color(self) -> str:
        return "green" if self.is_land else "blue"

    def harvest(self, amount: float) -> float:
        if not self.is_land or amount <= 0:
            return 0.0
        return min(amount, self.resource_value * 10.0)

    def step(self) -> None:
        return None


class Population(mesa.Agent):
    """Competitive population occupying one land tile."""

    def __init__(
        self,
        unique_id: int,
        model,
        inhabitant_count: int,
        lineage_color: str,
        traits: TraitMap,
        stockpile: float = 25.0,
        tech_level: int = 0,
    ) -> None:
        super().__init__(unique_id, model)
        self.inhabitant_count = int(inhabitant_count)
        self.stockpile = float(stockpile)
        self.tech_level = int(tech_level)
        self.lineage_color = lineage_color

        normalized = normalize_traits(traits)
        self.military_pct = normalized["military_pct"]
        self.economic_pct = normalized["economic_pct"]
        self.diplomatic_pct = normalized["diplomatic_pct"]
        self.tech_pct = normalized["tech_pct"]

        self.military_bank = 0.0
        self.economic_bank = 0.0
        self.diplomatic_bank = 0.0
        self.tech_bank = 0.0
        self.growth_remainder = 0.0

    @property
    def traits(self) -> TraitMap:
        return {key: getattr(self, key) for key in TRAIT_KEYS}

    def _set_traits(self, traits: TraitMap) -> None:
        normalized = normalize_traits(traits)
        for key, value in normalized.items():
            setattr(self, key, value)

    @property
    def tech_multiplier(self) -> float:
        return 1.0 + (self.tech_level * 0.05)

    def military_output(self) -> float:
        return self.military_pct * self.tech_multiplier

    def economic_output(self) -> float:
        return self.economic_pct * self.tech_multiplier

    def diplomatic_output(self) -> float:
        return self.diplomatic_pct * self.tech_multiplier

    def allocate(self, harvested: float) -> Allocation:
        normalized = self.traits
        allocation = Allocation(
            military=harvested * normalized["military_pct"] / 100.0,
            economic=harvested * normalized["economic_pct"] / 100.0,
            diplomatic=harvested * normalized["diplomatic_pct"] / 100.0,
            tech=harvested * normalized["tech_pct"] / 100.0,
        )
        self.military_bank += allocation.military
        self.economic_bank += allocation.economic
        self.diplomatic_bank += allocation.diplomatic
        self.tech_bank += allocation.tech
        self.stockpile += allocation.economic
        return allocation

    def harvest(self) -> float:
        cell = self.model.resource_cell_at(self.pos)
        if cell is None:
            return 0.0
        return cell.resource_value * (1.0 + self.economic_output() / 200.0)

    def advance_tech(self) -> None:
        threshold = self.model.tech_threshold * (1.0 + self.tech_level * 0.35)
        while self.tech_bank >= threshold:
            self.tech_bank -= threshold
            self.tech_level += 1
            threshold = self.model.tech_threshold * (1.0 + self.tech_level * 0.35)

    def diffuse_tech(self) -> None:
        richer_neighbors = [
            neighbor.tech_level
            for neighbor in self.model.populations_near(self.pos)
            if neighbor.tech_level > self.tech_level
        ]
        if richer_neighbors:
            gap = max(richer_neighbors) - self.tech_level
            self.tech_bank += gap * self.model.tech_diffusion_rate

    def drift_traits(self) -> None:
        drift = self.model.rng.normal(0.0, self.model.trait_drift_rate, len(TRAIT_KEYS))
        self._set_traits(
            {key: value + delta for (key, value), delta in zip(self.traits.items(), drift)}
        )

    def trade_with_neighbors(self) -> None:
        for neighbor in self.model.populations_near(self.pos):
            if self.stockpile <= 0 or neighbor.stockpile <= 0:
                continue

            diplomatic_delta = self.diplomatic_output() - neighbor.diplomatic_output()
            perceived_bonus = min(0.20, max(0.0, diplomatic_delta / 500.0))
            trade_amount = min(self.stockpile, neighbor.stockpile, 2.0)

            if trade_amount * (1.0 + perceived_bonus) >= trade_amount * 0.95:
                self.stockpile += trade_amount * perceived_bonus
                neighbor.stockpile += trade_amount * 0.02

    def expand_or_migrate(self) -> None:
        capacity = self.model.carrying_capacity_at(self.pos)
        if capacity <= 0:
            return
        if self.inhabitant_count < capacity * self.model.expansion_pressure_threshold:
            return

        candidate_positions = self.model.best_expansion_targets(self.pos)
        if not candidate_positions:
            return

        overflow = max(0, int(self.inhabitant_count - capacity * 0.72))
        migrants = max(
            self.model.minimum_migrants,
            overflow,
            int(self.inhabitant_count * self.model.migration_fraction),
        )
        migrants = min(migrants, max(1, self.inhabitant_count - 1))
        for target_pos in candidate_positions:
            if self.model.attempt_expansion(self, target_pos, migrants):
                self.inhabitant_count -= migrants
                break

    def grow_logistically(self) -> None:
        capacity = self.model.carrying_capacity_at(self.pos)
        if capacity <= 0:
            return

        economic_bonus = 1.0 + (self.economic_output() / 250.0)
        growth_rate = self.model.population_growth_rate * economic_bonus
        population = float(self.inhabitant_count)
        delta = growth_rate * population * (1.0 - population / capacity)
        next_population = max(1.0, population + delta + self.growth_remainder)
        self.inhabitant_count = max(1, int(next_population))
        self.growth_remainder = next_population - self.inhabitant_count

    def step(self) -> None:
        harvested = self.harvest()
        self.allocate(harvested)
        self.diffuse_tech()
        self.advance_tech()
        self.trade_with_neighbors()
        self.grow_logistically()
        self.drift_traits()
        self.expand_or_migrate()
