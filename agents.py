"""Agent definitions for the phase-1 economic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

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
BeliefMap = Dict[str, float]


def random_beliefs(rng: np.random.Generator) -> BeliefMap:
    x_tech = float(rng.uniform(0.0, 0.3))
    y_dip = float(rng.uniform(0.0, 0.3))
    e_econ_ratio = float(rng.uniform(0.0, 1.0))
    return normalize_beliefs(
        {
            "x_tech": x_tech,
            "y_dip": y_dip,
            "e_econ_ratio": e_econ_ratio,
        }
    )


def normalize_beliefs(beliefs: BeliefMap) -> BeliefMap:
    x_tech = float(np.clip(beliefs.get("x_tech", 0.0), 0.0, 0.3))
    y_dip = float(np.clip(beliefs.get("y_dip", 0.0), 0.0, 0.3))
    if x_tech + y_dip > 1.0:
        scale = 1.0 / (x_tech + y_dip)
        x_tech *= scale
        y_dip *= scale
    return {
        "x_tech": x_tech,
        "y_dip": y_dip,
        "z_physical": 1.0 - (x_tech + y_dip),
        "e_econ_ratio": float(np.clip(beliefs.get("e_econ_ratio", 0.5), 0.0, 1.0)),
    }


@dataclass(frozen=True)
class Allocation:
    military: float
    economic: float
    diplomatic: float
    tech: float


@dataclass(frozen=True)
class AttackPlan:
    target: "Population"
    defender_power: float
    optimal_force: float
    actual_force: int
    actual_win_prob: float
    attack_chance: float


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
        beliefs: Optional[BeliefMap] = None,
        stockpile: float = 25.0,
        tech_level: int = 0,
    ) -> None:
        super().__init__(unique_id, model)
        self.inhabitant_count = int(inhabitant_count)
        self.stockpile = float(stockpile)
        self.tech_level = int(tech_level)
        self.lineage_color = lineage_color

        normalized = normalize_beliefs(beliefs or random_beliefs(model.rng))
        self.x_tech = normalized["x_tech"]
        self.y_dip = normalized["y_dip"]
        self.z_physical = normalized["z_physical"]
        self.e_econ_ratio = normalized["e_econ_ratio"]

        self.military_bank = 0.0
        self.economic_bank = 0.0
        self.diplomatic_bank = 0.0
        self.tech_bank = 0.0
        self.growth_remainder = 0.0

    @property
    def beliefs(self) -> BeliefMap:
        return {
            "x_tech": self.x_tech,
            "y_dip": self.y_dip,
            "z_physical": self.z_physical,
            "e_econ_ratio": self.e_econ_ratio,
        }

    def _set_beliefs(self, beliefs: BeliefMap) -> None:
        normalized = normalize_beliefs(beliefs)
        self.x_tech = normalized["x_tech"]
        self.y_dip = normalized["y_dip"]
        self.z_physical = normalized["z_physical"]
        self.e_econ_ratio = normalized["e_econ_ratio"]

    @property
    def traits(self) -> Dict[str, float]:
        return self.investment_proportions

    @property
    def investment_proportions(self) -> Dict[str, float]:
        proportions = {
            "military": self.z_physical * (1.0 - self.e_econ_ratio),
            "economic": self.z_physical * self.e_econ_ratio,
            "diplomatic": self.y_dip,
            "tech": self.x_tech,
        }
        total = sum(proportions.values())
        if total <= 0:
            return {key: 0.25 for key in proportions}
        return {key: value / total for key, value in proportions.items()}

    @property
    def tech_multiplier(self) -> float:
        return 1.0 + (self.tech_level * 0.05)

    def military_output(self) -> float:
        return self.investment_proportions["military"] * self.tech_multiplier

    def economic_output(self) -> float:
        return self.investment_proportions["economic"] * self.tech_multiplier

    def diplomatic_output(self) -> float:
        return self.investment_proportions["diplomatic"] * self.tech_multiplier

    def allocate(self, harvested: float) -> Allocation:
        investments = self.investment_proportions
        allocation = Allocation(
            military=harvested * investments["military"],
            economic=harvested * investments["economic"],
            diplomatic=harvested * investments["diplomatic"],
            tech=harvested * investments["tech"],
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
        return cell.resource_value * (1.0 + self.economic_output())

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
        drift = self.model.rng.normal(0.0, self.model.trait_drift_rate, 3)
        self._set_beliefs(
            {
                "x_tech": self.x_tech + drift[0],
                "y_dip": self.y_dip + drift[1],
                "e_econ_ratio": self.e_econ_ratio + drift[2],
            }
        )

    def trade_with_neighbors(self) -> None:
        for neighbor in self.model.populations_near(self.pos):
            if self.stockpile <= 0 or neighbor.stockpile <= 0:
                continue

            diplomatic_delta = self.diplomatic_output() - neighbor.diplomatic_output()
            perceived_bonus = min(0.20, max(0.0, diplomatic_delta * 0.2))
            trade_amount = min(self.stockpile, neighbor.stockpile, 2.0)

            if trade_amount * (1.0 + perceived_bonus) >= trade_amount * 0.95:
                self.stockpile += trade_amount * perceived_bonus
                neighbor.stockpile += trade_amount * 0.02

    def bordering_enemy_populations(self) -> list["Population"]:
        return [
            neighbor
            for neighbor in self.model.populations_near(self.pos)
            if neighbor.lineage_color != self.lineage_color
        ]

    def plan_attack(self, target: "Population") -> Optional[AttackPlan]:
        defender_power = target.inhabitant_count * target.military_output()
        if self.military_output() > 0:
            optimal_force = 4.0 * (defender_power / self.military_output())
        else:
            optimal_force = float("inf")

        max_force = min(float(self.inhabitant_count) * 0.8, float(self.inhabitant_count - 1))
        if max_force < 1.0:
            return None

        actual_force_float = min(optimal_force, max_force)
        actual_force = max(1, int(actual_force_float))
        actual_attacker_power = actual_force * self.military_output()
        total_power = actual_attacker_power + defender_power
        if total_power > 0:
            actual_win_prob = actual_attacker_power / total_power
        else:
            actual_win_prob = 0.0

        dip_factor = max(
            0.1,
            target.diplomatic_output() - self.diplomatic_output() + 1.0,
        )
        carrying_capacity = max(1.0, self.model.carrying_capacity_at(self.pos))
        pop_pressure = self.inhabitant_count / carrying_capacity
        attack_chance = min(
            1.0,
            max(
                0.0,
                (actual_win_prob / dip_factor)
                * pop_pressure
                * self.model.attack_scale_constant,
            ),
        )
        return AttackPlan(
            target=target,
            defender_power=defender_power,
            optimal_force=optimal_force,
            actual_force=actual_force,
            actual_win_prob=actual_win_prob,
            attack_chance=attack_chance,
        )

    def maybe_attack_neighbor(self) -> bool:
        plans = [
            plan
            for enemy in self.bordering_enemy_populations()
            if (plan := self.plan_attack(enemy)) is not None
        ]
        if not plans:
            return False

        plans.sort(
            key=lambda plan: (plan.attack_chance, plan.actual_win_prob),
            reverse=True,
        )
        plan = plans[0]
        if self.model.rng.random() >= plan.attack_chance:
            return False

        self.inhabitant_count -= plan.actual_force
        if self.inhabitant_count < 1:
            self.inhabitant_count = 1
        self.model.attack_events += 1

        if self.model.rng.random() < plan.actual_win_prob:
            target = plan.target
            assimilation_rate = min(1.25 * self.diplomatic_output(), 1.0)
            assimilated = int(target.inhabitant_count * assimilation_rate)
            target.lineage_color = self.lineage_color
            target._set_beliefs(self.beliefs)
            target.tech_level = self.tech_level
            target.inhabitant_count = max(1, plan.actual_force + assimilated)
            target.growth_remainder = 0.0
            self.model.conquest_events += 1
            return True
        return False

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

        economic_bonus = 1.0 + (self.economic_output() * 0.05)
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
        self.maybe_attack_neighbor()
        self.expand_or_migrate()
