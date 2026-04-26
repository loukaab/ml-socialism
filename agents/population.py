"""Population and tile agents for the phase-2 macroeconomic simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import mesa
except (ModuleNotFoundError, ImportError):
    class _Agent:
        def __init__(self, unique_id=None, model=None):
            self.unique_id = unique_id
            self.model = model
            self.pos = None

    mesa = SimpleNamespace(Agent=_Agent)


Position = Tuple[int, int]
BeliefMap = Dict[str, float]


def init_mesa_agent(agent, unique_id: int, model) -> None:
    try:
        super(type(agent), agent).__init__(unique_id, model)
    except TypeError:
        super(type(agent), agent).__init__(model)
        agent.unique_id = unique_id
    agent.model = model
    if not hasattr(agent, "pos"):
        agent.pos = None


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
class JobAllocation:
    farmers: int
    extractors: int
    manufacturers: int
    artisans: int


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
        arable_value: float = 0.0,
        raw_goods_value: Optional[float] = None,
    ) -> None:
        init_mesa_agent(self, unique_id, model)
        self.terrain_type = terrain_type
        self.arable_value = float(arable_value)
        self.raw_goods_value = float(resource_value if raw_goods_value is None else raw_goods_value)
        self.resource_value = self.raw_goods_value
        self.carrying_capacity = float(carrying_capacity)
        self.raw_goods_stockpile = 0.0
        self.manufactory_level = 0
        self.devastation = 0.0
        self.steps_since_conflict = 0
        self.reset_tick_production()

    @property
    def is_land(self) -> bool:
        return self.terrain_type == "Land"

    @property
    def color(self) -> str:
        return "green" if self.is_land else "blue"

    def reset_tick_production(self) -> None:
        self.last_farmers = 0
        self.last_extractors = 0
        self.last_manufacturers = 0
        self.last_artisans = 0
        self.last_food_produced = 0.0
        self.last_food_claimed = 0.0
        self.last_birth_food = 0.0
        self.last_new_inhabitants = 0
        self.last_raw_extracted = 0.0
        self.last_refined_produced = 0.0

    @property
    def production_multiplier(self) -> float:
        max_devastation = self.model.economy_config.devastation_max
        if max_devastation <= 0:
            return 1.0
        return max(0.0, 1.0 - self.devastation / max_devastation)

    def clamp_devastation(self) -> None:
        max_devastation = self.model.economy_config.devastation_max
        self.devastation = min(max_devastation, max(0.0, float(self.devastation)))

    def add_devastation(self, amount: float) -> None:
        if amount <= 0:
            return
        self.devastation += amount
        self.clamp_devastation()
        self.steps_since_conflict = 0

    def recover_devastation(self) -> None:
        config = self.model.economy_config
        if not self.is_land or self.devastation <= 0:
            self.steps_since_conflict = 0
            return
        period = max(1, int(config.devastation_recovery_period))
        self.steps_since_conflict += 1
        if self.steps_since_conflict < period:
            return
        self.devastation -= config.devastation_recovery_amount
        self.clamp_devastation()
        self.steps_since_conflict = 0

    def harvest(self, amount: float) -> float:
        if not self.is_land or amount <= 0:
            return 0.0
        return min(amount, self.raw_goods_value * 10.0)

    def step(self) -> None:
        return None


class Population(mesa.Agent):
    """Competitive population occupying one controlled land tile."""

    def __init__(
        self,
        unique_id: int,
        model,
        inhabitant_count: int,
        lineage_color: Optional[str] = None,
        beliefs: Optional[BeliefMap] = None,
        stockpile: float = 25.0,
        tech_level: int = 0,
        nation=None,
    ) -> None:
        init_mesa_agent(self, unique_id, model)
        self.inhabitant_count = int(inhabitant_count)
        self._lineage_color = lineage_color or "#cccccc"
        self.nation = nation
        self.stockpile = float(stockpile)
        self.tech_level = int(tech_level)

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

        self.food_deficit_ticks = 0
        self.refined_deficit_ticks = 0
        self.refined_growth_multiplier = 1.0
        self.reset_tick_production()

    @property
    def lineage_color(self) -> str:
        if self.nation is not None:
            return self.nation.lineage_color
        return self._lineage_color

    @lineage_color.setter
    def lineage_color(self, value: str) -> None:
        self._lineage_color = value
        if self.nation is not None:
            self.nation.lineage_color = value

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

    def reset_tick_production(self) -> None:
        self.last_farmers = 0
        self.last_extractors = 0
        self.last_manufacturers = 0
        self.last_artisans = 0
        self.last_food_produced = 0.0
        self.last_food_claimed = 0.0
        self.last_birth_food = 0.0
        self.last_new_inhabitants = 0
        self.last_raw_extracted = 0.0
        self.last_refined_produced = 0.0

    def allocate_jobs(self) -> JobAllocation:
        cell = self.model.resource_cell_at(self.pos)
        if cell is None or not cell.is_land:
            return JobAllocation(0, 0, 0, max(0, self.inhabitant_count))

        config = self.model.economy_config
        total_people = max(0, self.inhabitant_count)

        farmer_slots = max(0, int(cell.arable_value * config.farmer_slot_scale))
        extractor_slots = max(0, int(cell.raw_goods_value * config.extractor_slot_scale))
        manufactory_level = min(max(0, cell.manufactory_level), 1)
        manufacturer_slots = max(0, manufactory_level * config.manufacturer_jobs_per_level)

        starter_jobs = self.starter_job_targets(min(total_people, config.starter_population_band))
        farmers = min(starter_jobs.farmers, farmer_slots)
        extractors = min(starter_jobs.extractors, extractor_slots)
        starter_shortfall = (
            starter_jobs.farmers
            + starter_jobs.extractors
            + starter_jobs.artisans
            - farmers
            - extractors
            - starter_jobs.artisans
        )
        artisans = starter_jobs.artisans + max(0, starter_shortfall)

        remaining = total_people - farmers - extractors - artisans
        remaining_farmer_slots = max(0, farmer_slots - farmers)
        extra_farmers = min(remaining, remaining_farmer_slots)
        farmers += extra_farmers
        remaining -= extra_farmers

        remaining_extractor_slots = max(0, extractor_slots - extractors)
        extra_extractors = min(remaining, remaining_extractor_slots)
        extractors += extra_extractors
        remaining -= extra_extractors

        manufacturers = min(remaining, manufacturer_slots)
        remaining -= manufacturers

        artisans += remaining
        return JobAllocation(farmers, extractors, manufacturers, artisans)

    def starter_job_targets(self, people: int) -> JobAllocation:
        config = self.model.economy_config
        if people <= 0 or config.starter_population_band <= 0:
            return JobAllocation(0, 0, 0, 0)

        starter_total = (
            config.starter_farmers
            + config.starter_extractors
            + config.starter_artisans
        )
        if starter_total <= 0:
            return JobAllocation(0, 0, 0, people)

        quotas = [
            ("farmers", people * config.starter_farmers / starter_total),
            ("extractors", people * config.starter_extractors / starter_total),
            ("artisans", people * config.starter_artisans / starter_total),
        ]
        allocations = {name: int(value) for name, value in quotas}
        remaining = people - sum(allocations.values())
        fractions = sorted(
            ((value - int(value), name) for name, value in quotas),
            key=lambda item: (-item[0], ["farmers", "extractors", "artisans"].index(item[1])),
        )
        for _, name in fractions[:remaining]:
            allocations[name] += 1

        return JobAllocation(
            farmers=allocations["farmers"],
            extractors=allocations["extractors"],
            manufacturers=0,
            artisans=allocations["artisans"],
        )

    def produce_goods(self) -> None:
        cell = self.model.resource_cell_at(self.pos)
        if cell is None or not cell.is_land:
            return

        config = self.model.economy_config
        jobs = self.allocate_jobs()
        multiplier = self.tech_multiplier * cell.production_multiplier

        food_produced = jobs.farmers * config.food_per_farmer * multiplier
        raw_extracted = jobs.extractors * config.raw_per_extractor * multiplier
        cell.raw_goods_stockpile += raw_extracted

        manufacturer_capacity = (
            jobs.manufacturers * config.manufacturer_raw_throughput * multiplier
        )
        manufacturer_output = min(cell.raw_goods_stockpile, manufacturer_capacity)
        cell.raw_goods_stockpile -= manufacturer_output

        artisan_capacity = jobs.artisans * config.artisan_raw_throughput * multiplier
        artisan_output = min(cell.raw_goods_stockpile, artisan_capacity)
        cell.raw_goods_stockpile -= artisan_output

        refined_produced = manufacturer_output + artisan_output

        self.last_farmers = cell.last_farmers = jobs.farmers
        self.last_extractors = cell.last_extractors = jobs.extractors
        self.last_manufacturers = cell.last_manufacturers = jobs.manufacturers
        self.last_artisans = cell.last_artisans = jobs.artisans
        self.last_food_produced = cell.last_food_produced = food_produced
        self.last_raw_extracted = cell.last_raw_extracted = raw_extracted
        self.last_refined_produced = cell.last_refined_produced = refined_produced

        if self.nation is not None:
            self.nation.add_production(food=food_produced, refined=refined_produced)

        self.allocate_development(food_produced + raw_extracted + refined_produced)

    def allocate_development(self, output_value: float) -> Allocation:
        investment_value = max(0.0, output_value) * 0.02
        investments = self.investment_proportions
        allocation = Allocation(
            military=investment_value * investments["military"],
            economic=investment_value * investments["economic"],
            diplomatic=investment_value * investments["diplomatic"],
            tech=investment_value * investments["tech"],
        )
        self.military_bank += allocation.military
        self.economic_bank += allocation.economic
        self.diplomatic_bank += allocation.diplomatic
        self.tech_bank += allocation.tech
        return allocation

    def allocate(self, harvested: float) -> Allocation:
        return self.allocate_development(harvested)

    def harvest(self) -> float:
        cell = self.model.resource_cell_at(self.pos)
        if cell is None:
            return 0.0
        return cell.raw_goods_value * (1.0 + self.economic_output())

    def consume_goods(self) -> None:
        if self.nation is None:
            return

        config = self.model.economy_config
        food_need = self.inhabitant_count * config.food_need_per_person
        food_claim_target = food_need * config.food_claim_multiplier
        claimed_food = self.claim_food(food_claim_target)
        self.last_food_claimed = claimed_food

        unmet_food = max(0.0, food_need - claimed_food)
        birth_food = 0.0
        if unmet_food > 0:
            self.food_deficit_ticks += 1
            loss = int(math.ceil(unmet_food * self.food_deficit_ticks * config.food_deficit_loss_rate))
            if loss > 0:
                self.inhabitant_count = max(1, self.inhabitant_count - loss)
        else:
            self.food_deficit_ticks = 0
            birth_food = max(0.0, claimed_food - food_need)
        self.last_birth_food = birth_food

        refined_need = self.inhabitant_count * config.refined_need_per_person
        unmet_refined = self.consume_one_good(
            need=refined_need,
            local_amount=self.last_refined_produced,
            stockpile_name="refined_stockpile",
        )
        if unmet_refined > 0:
            self.refined_deficit_ticks += 1
            stall = unmet_refined * self.refined_deficit_ticks / max(1e-9, refined_need)
            self.refined_growth_multiplier = max(0.0, 1.0 - min(1.0, stall))
        else:
            self.refined_deficit_ticks = 0
            self.refined_growth_multiplier = 1.0

        self.grow_from_food(birth_food)
        self.stockpile = self.nation.food_stockpile

    def claim_food(self, claim_target: float) -> float:
        local_food = max(0.0, self.last_food_produced)
        if claim_target <= 0:
            self.nation.food_stockpile += local_food
            return 0.0

        local_claim = min(local_food, claim_target)
        local_surplus = max(0.0, local_food - claim_target)
        if local_surplus > 0:
            self.nation.food_stockpile += local_surplus

        remaining_claim = claim_target - local_claim
        if remaining_claim <= 0:
            return local_claim

        pulled = min(self.nation.food_stockpile, remaining_claim)
        self.nation.food_stockpile -= pulled
        return local_claim + pulled

    def grow_from_food(self, birth_food: float) -> None:
        config = self.model.economy_config
        self.last_new_inhabitants = 0
        if birth_food <= 0 or config.food_per_new_person <= 0:
            return

        potential_births = (birth_food / config.food_per_new_person) * self.refined_growth_multiplier
        total_births = potential_births + self.growth_remainder
        new_inhabitants = int(total_births)
        self.growth_remainder = total_births - new_inhabitants
        if new_inhabitants <= 0:
            return
        self.inhabitant_count += new_inhabitants
        self.last_new_inhabitants = new_inhabitants

    def consume_one_good(
        self,
        need: float,
        local_amount: float,
        stockpile_name: str,
    ) -> float:
        if need <= 0:
            setattr(
                self.nation,
                stockpile_name,
                getattr(self.nation, stockpile_name) + max(0.0, local_amount),
            )
            return 0.0

        local_amount = max(0.0, local_amount)
        if local_amount >= need:
            surplus = local_amount - need
            setattr(self.nation, stockpile_name, getattr(self.nation, stockpile_name) + surplus)
            return 0.0

        deficit = need - local_amount
        available = getattr(self.nation, stockpile_name)
        pulled = min(available, deficit)
        setattr(self.nation, stockpile_name, available - pulled)
        return deficit - pulled

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
        return None

    def bordering_enemy_populations(self) -> list["Population"]:
        return [
            neighbor
            for neighbor in self.model.populations_near(self.pos)
            if neighbor.nation is not self.nation
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
        carrying_capacity = max(1.0, self.model.food_growth_capacity_for_population(self))
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
        if self.nation is None or self.nation.defeated:
            return False

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
        if not self.pay_combat_refined_cost():
            return False

        self.model.register_attack_arrow(self.pos, plan.target.pos)
        self.inhabitant_count -= plan.actual_force
        if self.inhabitant_count < 1:
            self.inhabitant_count = 1
        self.model.attack_events += 1

        if self.model.rng.random() < plan.actual_win_prob:
            target = plan.target
            assimilation_rate = min(1.25 * self.diplomatic_output(), 1.0)
            assimilated = int(target.inhabitant_count * assimilation_rate)
            new_inhabitants = max(1, plan.actual_force + assimilated)
            self.model.handle_conquest(
                attacker=self,
                target=target,
                new_inhabitants=new_inhabitants,
                new_beliefs=self.beliefs,
                new_tech_level=self.tech_level,
            )
            return True
        self.model.add_tile_devastation(
            plan.target.pos,
            self.model.economy_config.devastation_failed_attack_increase,
        )
        return False

    def combat_refined_cost(self) -> float:
        config = self.model.economy_config
        return (
            config.combat_refined_base_cost
            + config.combat_refined_cost_per_tech_level * self.tech_level
        )

    def pay_combat_refined_cost(self) -> bool:
        if self.nation is None:
            return False

        cost = self.combat_refined_cost()
        if self.nation.refined_stockpile < cost:
            return False

        self.nation.refined_stockpile -= cost
        return True

    def expand_or_migrate(self) -> None:
        capacity = self.model.food_growth_capacity_for_population(self)
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
        return None

    def step(self) -> None:
        self.produce_goods()
        self.consume_goods()
        self.diffuse_tech()
        self.advance_tech()
        self.drift_traits()
        self.maybe_attack_neighbor()
        self.expand_or_migrate()
