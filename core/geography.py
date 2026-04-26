"""Terrain, arable land, raw goods, and carrying-capacity generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainConfig:
    land_fraction: float = 0.58
    smoothing_passes: int = 5
    arable_smoothing_passes: int = 7
    raw_goods_smoothing_passes: int = 7
    carrying_capacity_min: float = 90.0
    carrying_capacity_max: float = 520.0


@dataclass(frozen=True)
class GeographyMaps:
    terrain_map: np.ndarray
    arable_map: np.ndarray
    raw_goods_map: np.ndarray
    carrying_capacity_map: np.ndarray


def smooth_noise(
    rng: np.random.Generator,
    width: int,
    height: int,
    smoothing_passes: int,
) -> np.ndarray:
    noise = rng.random((height, width))
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


def generate_terrain_map(
    rng: np.random.Generator,
    width: int,
    height: int,
    config: TerrainConfig,
) -> np.ndarray:
    noise = smooth_noise(rng, width, height, config.smoothing_passes)
    threshold = np.quantile(noise, 1.0 - config.land_fraction)
    return noise >= threshold


def normalize_land_values(noise: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
    values = np.zeros_like(noise, dtype=float)
    land_values = noise[terrain_map]
    if land_values.size == 0:
        return values

    low = float(land_values.min())
    high = float(land_values.max())
    if high == low:
        values[terrain_map] = 1.0
        return values

    normalized = (noise - low) / (high - low)
    values[terrain_map] = np.clip(normalized[terrain_map], 0.0, 1.0)
    return values


def generate_geography_maps(
    rng: np.random.Generator,
    width: int,
    height: int,
    config: TerrainConfig,
) -> GeographyMaps:
    terrain_map = generate_terrain_map(rng, width, height, config)
    arable_noise = smooth_noise(rng, width, height, config.arable_smoothing_passes)
    raw_noise = smooth_noise(rng, width, height, config.raw_goods_smoothing_passes)

    arable_map = normalize_land_values(arable_noise, terrain_map)
    raw_goods_map = normalize_land_values(raw_noise, terrain_map)

    capacity = (
        config.carrying_capacity_min
        + arable_map * (config.carrying_capacity_max - config.carrying_capacity_min)
    )
    capacity[~terrain_map] = 0.0

    return GeographyMaps(
        terrain_map=terrain_map,
        arable_map=arable_map,
        raw_goods_map=raw_goods_map,
        carrying_capacity_map=capacity,
    )
