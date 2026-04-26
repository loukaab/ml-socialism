from __future__ import annotations

import unittest

import numpy as np

from agents import Population
from model import WorldModel
from run import build_parser


FIXED_BELIEFS = {"x_tech": 0.1, "y_dip": 0.1, "e_econ_ratio": 0.5}


def force_land(model: WorldModel, pos, arable: float = 1.0, raw: float = 1.0):
    x, y = pos
    model.terrain_map[y, x] = True
    model.arable_map[y, x] = arable
    model.raw_goods_map[y, x] = raw
    model.resource_map[y, x] = raw
    model.carrying_capacity_map[y, x] = (
        model.terrain_config.carrying_capacity_min
        + arable
        * (
            model.terrain_config.carrying_capacity_max
            - model.terrain_config.carrying_capacity_min
        )
    )
    cell = model.resource_cell_at(pos)
    cell.terrain_type = "Land"
    cell.arable_value = arable
    cell.raw_goods_value = raw
    cell.resource_value = raw
    cell.carrying_capacity = float(model.carrying_capacity_map[y, x])
    return cell


def force_water(model: WorldModel, pos):
    x, y = pos
    model.terrain_map[y, x] = False
    model.arable_map[y, x] = 0.0
    model.raw_goods_map[y, x] = 0.0
    model.resource_map[y, x] = 0.0
    model.carrying_capacity_map[y, x] = 0.0
    cell = model.resource_cell_at(pos)
    cell.terrain_type = "Water"
    cell.arable_value = 0.0
    cell.raw_goods_value = 0.0
    cell.resource_value = 0.0
    cell.carrying_capacity = 0.0
    return cell


def add_population(
    model: WorldModel,
    pos,
    count: int,
    nation=None,
    color: str = "#e83f6f",
):
    if nation is None:
        nation = model.create_nation(color, pos)
    population = Population(
        unique_id=model.next_id(),
        model=model,
        inhabitant_count=count,
        nation=nation,
        beliefs=FIXED_BELIEFS,
    )
    model.grid.place_agent(population, pos)
    model.schedule.add(population)
    return population


class Phase2MacroeconomicsTests(unittest.TestCase):
    def test_arable_raw_independent_and_legacy_capacity_map_from_arable(self):
        model = WorldModel(width=20, height=15, initial_populations=0, seed=3)

        self.assertFalse(np.allclose(model.arable_map, model.raw_goods_map))
        expected_capacity = (
            model.terrain_config.carrying_capacity_min
            + model.arable_map
            * (
                model.terrain_config.carrying_capacity_max
                - model.terrain_config.carrying_capacity_min
            )
        )
        expected_capacity[~model.terrain_map] = 0.0
        np.testing.assert_allclose(model.carrying_capacity_map, expected_capacity)

    def test_carrying_capacity_wrapper_is_food_growth_capacity(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos, arable=0.5, raw=1.0)

        expected_capacity = 60 * 4.0 / 1.2

        self.assertAlmostEqual(model.carrying_capacity_at(pos), expected_capacity)

    def test_job_waterfall_and_manufacturer_throughput(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        cell = force_land(model, pos, arable=0.5, raw=0.5)
        cell.raw_goods_stockpile = 1000.0
        cell.manufactory_level = 2
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=200, nation=nation)

        population.produce_goods()

        self.assertEqual(population.last_farmers, 60)
        self.assertEqual(population.last_extractors, 40)
        self.assertEqual(population.last_manufacturers, 25)
        self.assertEqual(population.last_artisans, 75)
        self.assertEqual(population.last_food_produced, 240.0)
        self.assertEqual(population.last_refined_produced, 200.0)
        self.assertEqual(cell.raw_goods_stockpile, 840.0)
        self.assertEqual(nation.gdp, 840.0)

    def test_first_sixty_use_starter_job_distribution(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos, arable=1.0, raw=1.0)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=60, nation=nation)

        jobs = population.allocate_jobs()

        self.assertEqual(jobs.farmers, 30)
        self.assertEqual(jobs.extractors, 20)
        self.assertEqual(jobs.manufacturers, 0)
        self.assertEqual(jobs.artisans, 10)

    def test_small_starter_population_scales_distribution(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos, arable=1.0, raw=1.0)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=36, nation=nation)

        jobs = population.allocate_jobs()

        self.assertEqual(jobs.farmers, 18)
        self.assertEqual(jobs.extractors, 12)
        self.assertEqual(jobs.manufacturers, 0)
        self.assertEqual(jobs.artisans, 6)

    def test_starter_job_limits_fall_back_to_artisans(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos, arable=0.1, raw=0.1)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=60, nation=nation)

        jobs = population.allocate_jobs()

        self.assertEqual(jobs.farmers, 12)
        self.assertEqual(jobs.extractors, 8)
        self.assertEqual(jobs.manufacturers, 0)
        self.assertEqual(jobs.artisans, 40)

    def test_consumption_uses_local_then_global_and_tracks_deficits(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 15.0
        population.last_refined_produced = 1.0
        population.consume_goods()

        self.assertEqual(population.food_deficit_ticks, 0)
        self.assertEqual(population.refined_deficit_ticks, 0)
        self.assertAlmostEqual(nation.food_stockpile, 3.0)
        self.assertAlmostEqual(population.last_birth_food, 2.0)
        self.assertAlmostEqual(population.growth_remainder, 0.4)
        self.assertAlmostEqual(nation.refined_stockpile, 0.6)

        nation.food_stockpile = 0.0
        nation.refined_stockpile = 0.0
        population.last_food_produced = 0.0
        population.last_refined_produced = 0.0
        population.consume_goods()

        self.assertEqual(population.food_deficit_ticks, 1)
        self.assertEqual(population.refined_deficit_ticks, 1)
        self.assertEqual(population.inhabitant_count, 9)
        self.assertEqual(population.refined_growth_multiplier, 0.0)

    def test_food_below_consumption_causes_deficit_and_no_growth(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 5.0
        population.last_refined_produced = 1.0
        population.consume_goods()

        self.assertEqual(population.food_deficit_ticks, 1)
        self.assertEqual(population.inhabitant_count, 9)
        self.assertEqual(population.last_birth_food, 0.0)
        self.assertEqual(population.growth_remainder, 0.0)

    def test_food_at_110_percent_generates_partial_growth_without_stockpile_draw(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 11.0
        population.last_refined_produced = 1.0
        population.consume_goods()

        self.assertEqual(population.food_deficit_ticks, 0)
        self.assertAlmostEqual(nation.food_stockpile, 0.0)
        self.assertAlmostEqual(population.last_birth_food, 1.0)
        self.assertAlmostEqual(population.growth_remainder, 0.2)

    def test_food_stockpile_can_fill_to_120_percent_for_growth(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos, food_stockpile=2.0)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 10.0
        population.last_refined_produced = 1.0
        population.consume_goods()

        self.assertEqual(population.food_deficit_ticks, 0)
        self.assertAlmostEqual(nation.food_stockpile, 0.0)
        self.assertAlmostEqual(population.last_food_claimed, 12.0)
        self.assertAlmostEqual(population.last_birth_food, 2.0)
        self.assertAlmostEqual(population.growth_remainder, 0.4)

    def test_food_above_120_percent_sends_surplus_to_stockpile(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 20.0
        population.last_refined_produced = 1.0
        population.consume_goods()

        self.assertAlmostEqual(population.last_food_claimed, 12.0)
        self.assertAlmostEqual(population.last_birth_food, 2.0)
        self.assertAlmostEqual(nation.food_stockpile, 8.0)

    def test_refined_deficit_still_stalls_food_generated_growth(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        force_land(model, pos)
        nation = model.create_nation("#e83f6f", pos)
        population = add_population(model, pos, count=10, nation=nation)

        population.last_food_produced = 12.0
        population.last_refined_produced = 0.0
        population.consume_goods()

        self.assertEqual(population.refined_deficit_ticks, 1)
        self.assertEqual(population.refined_growth_multiplier, 0.0)
        self.assertAlmostEqual(population.last_birth_food, 2.0)
        self.assertEqual(population.growth_remainder, 0.0)
        self.assertEqual(population.inhabitant_count, 10)

    def test_lps_raw_goods_distribution_uses_center_bias_and_same_nation(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        center = (1, 1)
        east = (2, 1)
        north = (1, 2)
        for pos in (center, east, north):
            force_land(model, pos)

        nation = model.create_nation("#e83f6f", center)
        add_population(model, center, count=20, nation=nation)
        add_population(model, east, count=20, nation=nation)
        add_population(model, north, count=20, nation=nation)

        center_cell = model.resource_cell_at(center)
        east_cell = model.resource_cell_at(east)
        north_cell = model.resource_cell_at(north)
        center_cell.raw_goods_stockpile = 90.0
        center_cell.last_refined_produced = 10.0
        east_cell.last_refined_produced = 10.0
        north_cell.last_refined_produced = 0.0

        model.redistribute_local_raw_goods()

        self.assertAlmostEqual(center_cell.raw_goods_stockpile, 54.0)
        self.assertAlmostEqual(east_cell.raw_goods_stockpile, 36.0)
        self.assertAlmostEqual(north_cell.raw_goods_stockpile, 0.0)

    def test_conquest_reassigns_capital_and_transfers_stockpile_on_defeat(self):
        model = WorldModel(width=4, height=3, initial_populations=0, seed=1)
        a_capital = (0, 1)
        a_second = (1, 1)
        b_pos = (2, 1)
        for pos in (a_capital, a_second, b_pos):
            force_land(model, pos)

        nation_a = model.create_nation("#e83f6f", a_capital)
        nation_b = model.create_nation("#a855f7", b_pos)
        pop_a_capital = add_population(model, a_capital, count=10, nation=nation_a)
        pop_a_second = add_population(model, a_second, count=80, nation=nation_a)
        pop_b = add_population(model, b_pos, count=60, nation=nation_b)

        model.handle_conquest(pop_b, pop_a_capital, 20, pop_b.beliefs, pop_b.tech_level)

        self.assertIs(pop_a_capital.nation, nation_b)
        self.assertEqual(nation_a.capital_pos, a_second)
        self.assertFalse(nation_a.defeated)

        nation_a.food_stockpile = 12.0
        nation_a.refined_stockpile = 8.0
        model.handle_conquest(pop_b, pop_a_second, 30, pop_b.beliefs, pop_b.tech_level)

        self.assertTrue(nation_a.defeated)
        self.assertEqual(nation_a.capital_pos, None)
        self.assertAlmostEqual(nation_b.food_stockpile, 12.0)
        self.assertAlmostEqual(nation_b.refined_stockpile, 8.0)

    def test_investment_chooses_highest_artisan_tile(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        low = (0, 1)
        high = (1, 1)
        for pos in (low, high):
            force_land(model, pos)

        nation = model.create_nation("#e83f6f", low, refined_stockpile=250.0)
        low_pop = add_population(model, low, count=20, nation=nation)
        high_pop = add_population(model, high, count=20, nation=nation)
        low_pop.last_artisans = 5
        high_pop.last_artisans = 20

        invested = nation.invest_in_manufactory(model)

        self.assertTrue(invested)
        self.assertEqual(model.resource_cell_at(low).manufactory_level, 0)
        self.assertEqual(model.resource_cell_at(high).manufactory_level, 1)
        self.assertEqual(nation.refined_stockpile, 0.0)

    def test_investment_cannot_build_second_manufactory_on_tile(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        pos = (1, 1)
        cell = force_land(model, pos)
        cell.manufactory_level = 1
        nation = model.create_nation("#e83f6f", pos, refined_stockpile=250.0)
        population = add_population(model, pos, count=20, nation=nation)
        population.last_artisans = 20

        invested = nation.invest_in_manufactory(model)

        self.assertFalse(invested)
        self.assertEqual(cell.manufactory_level, 1)
        self.assertEqual(nation.refined_stockpile, 250.0)

    def test_expansion_uses_food_growth_capacity_threshold(self):
        model = WorldModel(width=3, height=3, initial_populations=0, seed=1)
        for y in range(model.height):
            for x in range(model.width):
                force_water(model, (x, y))
        center = (1, 1)
        target = (2, 1)
        force_land(model, center, arable=0.5, raw=1.0)
        force_land(model, target, arable=1.0, raw=1.0)
        nation = model.create_nation("#e83f6f", center)
        population = add_population(model, center, count=140, nation=nation)

        population.expand_or_migrate()

        self.assertEqual(model.expansion_events, 1)
        self.assertIsNotNone(model.population_at(target))
        self.assertEqual(model.population_at(target).inhabitant_count, 30)
        self.assertEqual(population.inhabitant_count, 110)

    def test_cli_accepts_new_modes_and_legacy_resources_alias(self):
        parser = build_parser()
        for mode in ("arable", "raw", "manufactories", "resources"):
            args = parser.parse_args(["--headless", "--map-mode", mode])
            self.assertEqual(args.map_mode, mode)


if __name__ == "__main__":
    unittest.main()
