"""Interactive Pygame viewer for the economic simulation."""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from model import WorldModel


Color = Tuple[int, int, int]

LAND_COLOR: Color = (46, 140, 61)
WATER_COLOR: Color = (38, 89, 199)
GRID_COLOR: Color = (24, 31, 42)
LOW_RESOURCE_COLOR: Color = (97, 69, 31)
HIGH_RESOURCE_COLOR: Color = (250, 214, 56)
TEXT_COLOR: Color = (235, 238, 242)
PANEL_COLOR: Color = (20, 24, 31)
PANEL_BORDER: Color = (75, 84, 99)
SLIDER_TRACK_COLOR: Color = (72, 79, 92)
SLIDER_FILL_COLOR: Color = (211, 217, 225)
SLIDER_KNOB_COLOR: Color = (255, 255, 255)
MIN_POPULATION_BRIGHTNESS = 0.35
POPULATION_BRIGHTNESS_GAMMA = 0.65
MAP_MODES = ("terrain", "resources", "tech", "diplo", "physical")
MIN_STEPS_PER_SECOND = 1.0
MAX_STEPS_PER_SECOND = 30.0


def hex_to_rgb(value: str) -> Color:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))


class InteractiveViewer:
    """Resizable window with pan, zoom, and optional live simulation stepping."""

    def __init__(
        self,
        model: WorldModel,
        width: int = 1280,
        height: int = 820,
        initial_tile_size: float = 18.0,
        fps: int = 60,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("ML Socialism - Economic Simulator")

        self.model = model
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.tile_fonts = {}
        self.fps = fps

        self.tile_size = initial_tile_size
        self.min_tile_size = 5.0
        self.max_tile_size = 95.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.dragging = False
        self.slider_dragging = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.running = True
        self.playing = False
        self.steps_per_second = 4.0
        self.step_accumulator = 0.0
        self.map_mode = "terrain"

        self.center_map()

    @property
    def show_resource_overlay(self) -> bool:
        return self.map_mode == "resources"

    @show_resource_overlay.setter
    def show_resource_overlay(self, enabled: bool) -> None:
        self.map_mode = "resources" if enabled else "terrain"

    def center_map(self) -> None:
        screen_width, screen_height = self.screen.get_size()
        map_width = self.model.width * self.tile_size
        map_height = self.model.height * self.tile_size
        self.camera_x = (screen_width - map_width) / 2.0
        self.camera_y = (screen_height - map_height) / 2.0

    def run(self) -> None:
        while self.running:
            seconds = self.clock.tick(self.fps) / 1000.0
            self.handle_events()
            self.advance_if_playing(seconds)
            self.draw()
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2, 3):
                    self.dragging = False
                    self.slider_dragging = False
                    self.last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if self.slider_dragging:
                    self.update_slider_from_mouse(event.pos)
                elif self.dragging:
                    self.pan_to(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom_at(pygame.mouse.get_pos(), 1.12 if event.y > 0 else 1 / 1.12)

    def handle_key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key in (pygame.K_RETURN, pygame.K_s):
            self.model.step()
        elif key == pygame.K_r:
            self.center_map()
        elif key in (pygame.K_m, pygame.K_TAB):
            self.cycle_map_mode()
        elif key in (pygame.K_1, pygame.K_KP1):
            self.map_mode = "terrain"
        elif key in (pygame.K_2, pygame.K_KP2):
            self.map_mode = "resources"
        elif key in (pygame.K_3, pygame.K_KP3):
            self.map_mode = "tech"
        elif key in (pygame.K_4, pygame.K_KP4):
            self.map_mode = "diplo"
        elif key in (pygame.K_5, pygame.K_KP5):
            self.map_mode = "physical"
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom_at(self.screen_center(), 1.12)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom_at(self.screen_center(), 1 / 1.12)
        elif key in (pygame.K_UP, pygame.K_w):
            self.camera_y += 32
        elif key in (pygame.K_DOWN, pygame.K_x):
            self.camera_y -= 32
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.camera_x += 32
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.camera_x -= 32

    def cycle_map_mode(self) -> None:
        index = MAP_MODES.index(self.map_mode)
        self.map_mode = MAP_MODES[(index + 1) % len(MAP_MODES)]

    def handle_mouse_down(self, event) -> None:
        if event.button in (1, 2, 3):
            if self.slider_rect().collidepoint(event.pos):
                self.slider_dragging = True
                self.dragging = False
                self.last_mouse_pos = None
                self.update_slider_from_mouse(event.pos)
                return
            self.dragging = True
            self.last_mouse_pos = event.pos
        elif event.button == 4:
            self.zoom_at(event.pos, 1.12)
        elif event.button == 5:
            self.zoom_at(event.pos, 1 / 1.12)

    def pan_to(self, mouse_pos: Tuple[int, int]) -> None:
        if self.last_mouse_pos is None:
            self.last_mouse_pos = mouse_pos
            return
        dx = mouse_pos[0] - self.last_mouse_pos[0]
        dy = mouse_pos[1] - self.last_mouse_pos[1]
        self.camera_x += dx
        self.camera_y += dy
        self.last_mouse_pos = mouse_pos

    def zoom_at(self, screen_pos: Tuple[int, int], factor: float) -> None:
        old_tile_size = self.tile_size
        new_tile_size = max(
            self.min_tile_size,
            min(self.max_tile_size, self.tile_size * factor),
        )
        if new_tile_size == old_tile_size:
            return

        mouse_x, mouse_y = screen_pos
        world_x = (mouse_x - self.camera_x) / old_tile_size
        world_y = (mouse_y - self.camera_y) / old_tile_size
        self.tile_size = new_tile_size
        self.camera_x = mouse_x - world_x * new_tile_size
        self.camera_y = mouse_y - world_y * new_tile_size

    def screen_center(self) -> Tuple[int, int]:
        width, height = self.screen.get_size()
        return width // 2, height // 2

    def advance_if_playing(self, seconds: float) -> None:
        if not self.playing:
            return
        self.step_accumulator += seconds * self.steps_per_second
        while self.step_accumulator >= 1.0:
            self.model.step()
            self.step_accumulator -= 1.0

    def draw(self) -> None:
        self.screen.fill((12, 15, 20))
        self.draw_world()
        self.draw_status_panel()
        pygame.display.flip()

    def draw_world(self) -> None:
        screen_width, screen_height = self.screen.get_size()
        start_x = max(0, int((-self.camera_x // self.tile_size) - 1))
        start_y = max(0, int((-self.camera_y // self.tile_size) - 1))
        end_x = min(
            self.model.width,
            int(((screen_width - self.camera_x) // self.tile_size) + 2),
        )
        end_y = min(
            self.model.height,
            int(((screen_height - self.camera_y) // self.tile_size) + 2),
        )

        tile = max(1, int(self.tile_size + 1))
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                color = self.tile_color(x, y)
                rect = pygame.Rect(
                    int(self.camera_x + x * self.tile_size),
                    int(self.camera_y + y * self.tile_size),
                    tile,
                    tile,
                )
                pygame.draw.rect(self.screen, color, rect)

        self.draw_owned_tiles()
        self.draw_lineage_borders(start_x, start_y, end_x, end_y)
        self.draw_hover_population_label()
        if self.tile_size >= 14:
            self.draw_grid(start_x, start_y, end_x, end_y)

    def draw_grid(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        for x in range(start_x, end_x + 1):
            screen_x = int(self.camera_x + x * self.tile_size)
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (screen_x, int(self.camera_y + start_y * self.tile_size)),
                (screen_x, int(self.camera_y + end_y * self.tile_size)),
                1,
            )
        for y in range(start_y, end_y + 1):
            screen_y = int(self.camera_y + y * self.tile_size)
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (int(self.camera_x + start_x * self.tile_size), screen_y),
                (int(self.camera_x + end_x * self.tile_size), screen_y),
                1,
            )

    def draw_owned_tiles(self) -> None:
        tile = max(1, int(self.tile_size + 1))
        global_max_population = self.model.global_max_population()
        for population in self.model.populations:
            x, y = population.pos
            color = self.population_tile_color(population, global_max_population)
            rect = pygame.Rect(
                int(self.camera_x + x * self.tile_size),
                int(self.camera_y + y * self.tile_size),
                tile,
                tile,
            )
            pygame.draw.rect(self.screen, color, rect)

    def draw_lineage_borders(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> None:
        thickness = max(2, min(6, int(self.tile_size * 0.16)))
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                population = self.model.population_at((x, y))
                if population is None:
                    continue

                left = int(self.camera_x + x * self.tile_size)
                top = int(self.camera_y + y * self.tile_size)
                right = int(self.camera_x + (x + 1) * self.tile_size)
                bottom = int(self.camera_y + (y + 1) * self.tile_size)

                if self.has_border(population, x - 1, y):
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (left, top),
                        (left, bottom),
                        thickness,
                    )
                if self.has_border(population, x + 1, y):
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (right, top),
                        (right, bottom),
                        thickness,
                    )
                if self.has_border(population, x, y - 1):
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (left, top),
                        (right, top),
                        thickness,
                    )
                if self.has_border(population, x, y + 1):
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (left, bottom),
                        (right, bottom),
                        thickness,
                    )

    def has_border(self, population, neighbor_x: int, neighbor_y: int) -> bool:
        if not (0 <= neighbor_x < self.model.width and 0 <= neighbor_y < self.model.height):
            return True
        neighbor = self.model.population_at((neighbor_x, neighbor_y))
        if neighbor is None:
            return True
        return neighbor.lineage_color != population.lineage_color

    def draw_hover_population_label(self) -> None:
        if self.tile_size < 12:
            return

        tile_pos = self.screen_to_tile(pygame.mouse.get_pos())
        if tile_pos is None:
            return

        population = self.model.population_at(tile_pos)
        if population is None:
            return

        font = self.tile_label_font()
        x, y = population.pos
        label_text = str(population.inhabitant_count)
        shadow = self.fit_label(label_text, font, (0, 0, 0))
        label = self.fit_label(label_text, font, TEXT_COLOR)
        label_rect = label.get_rect(
            center=(
                int(self.camera_x + (x + 0.5) * self.tile_size),
                int(self.camera_y + (y + 0.5) * self.tile_size),
            )
        )
        shadow_rect = shadow.get_rect(center=(label_rect.centerx + 1, label_rect.centery + 1))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(label, label_rect)

    def tile_label_font(self):
        size = max(10, min(24, int(self.tile_size * 0.42)))
        if size not in self.tile_fonts:
            self.tile_fonts[size] = pygame.font.Font(None, size)
        return self.tile_fonts[size]

    def fit_label(self, text: str, font, color: Color):
        label = font.render(text, True, color)
        max_width = max(4, int(self.tile_size * 0.88))
        max_height = max(4, int(self.tile_size * 0.72))
        if label.get_width() <= max_width and label.get_height() <= max_height:
            return label

        scale = min(max_width / label.get_width(), max_height / label.get_height())
        width = max(1, int(label.get_width() * scale))
        height = max(1, int(label.get_height() * scale))
        return pygame.transform.smoothscale(label, (width, height))

    def population_tile_color(self, population, global_max_population: Optional[int] = None) -> Color:
        if self.map_mode == "tech":
            return self.dark_investment_color(
                population.x_tech,
                max_value=0.3,
                light=(222, 210, 255),
                dark=(44, 22, 92),
            )
        if self.map_mode == "diplo":
            return self.dark_investment_color(
                population.y_dip,
                max_value=0.3,
                light=(255, 212, 232),
                dark=(95, 18, 58),
            )
        if self.map_mode == "physical":
            return self.physical_split_color(population.e_econ_ratio)

        lineage_color = hex_to_rgb(population.lineage_color)
        max_population = max(1, global_max_population or self.model.global_max_population())
        if max_population <= 1:
            normalized = 0.0
        else:
            normalized = (population.inhabitant_count - 1) / (max_population - 1)
        normalized = min(1.0, max(0.0, normalized))
        curved = normalized ** POPULATION_BRIGHTNESS_GAMMA
        brightness = 1.0 - curved * (1.0 - MIN_POPULATION_BRIGHTNESS)
        return tuple(int(channel * brightness) for channel in lineage_color)

    def dark_investment_color(
        self,
        value: float,
        max_value: float,
        light: Color,
        dark: Color,
    ) -> Color:
        normalized = min(1.0, max(0.0, value / max_value))
        return tuple(
            int(light[index] + normalized * (dark[index] - light[index]))
            for index in range(3)
        )

    def physical_split_color(self, e_econ_ratio: float) -> Color:
        ratio = min(1.0, max(0.0, e_econ_ratio))
        red = (225, 42, 42)
        yellow = (250, 220, 42)
        return tuple(
            int(red[index] + ratio * (yellow[index] - red[index]))
            for index in range(3)
        )

    def tile_color(self, x: int, y: int) -> Color:
        if not self.model.terrain_map[y, x]:
            return WATER_COLOR
        if self.map_mode != "resources":
            return LAND_COLOR

        resource = float(self.model.resource_map[y, x])
        return tuple(
            int(
                LOW_RESOURCE_COLOR[index]
                + resource * (HIGH_RESOURCE_COLOR[index] - LOW_RESOURCE_COLOR[index])
            )
            for index in range(3)
        )

    def draw_status_panel(self) -> None:
        rows = self.status_rows()

        width = 220
        height = 20 + len(rows) * 24 + 36
        panel = pygame.Rect(14, 14, width, height)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, width=1, border_radius=6)

        for index, row in enumerate(rows):
            surface = self.font.render(row, True, TEXT_COLOR)
            self.screen.blit(surface, (26, 24 + index * 24))

        self.draw_slider()

    def slider_rect(self) -> pygame.Rect:
        row_count = len(self.status_rows())
        return pygame.Rect(26, 24 + row_count * 24 + 8, 170, 8)

    def status_rows(self) -> list[str]:
        latest = self.model.datacollector.get_model_vars_dataframe().iloc[-1]
        hover_text = self.hover_text()
        return [
            f"Step {int(latest['Step'])}",
            f"{'Playing' if self.playing else 'Paused'}",
            f"Populations {int(latest['PopulationAgents'])}",
            f"Occupied {int(latest['OccupiedTiles'])}",
            f"Inhabitants {int(latest['TotalInhabitants'])}",
            f"Expansions {int(latest['ExpansionEvents'])}",
            f"Attacks {int(latest['AttackEvents'])}",
            f"Conquests {int(latest['ConquestEvents'])}",
            f"Lineages {int(latest['SurvivingLineages'])}",
            f"Max tech {int(latest['MaxTech'])}",
            f"Dominant {latest['DominantTrait']}",
            f"Map {self.map_mode.title()}",
            f"Timestep {self.steps_per_second:.1f}/s",
            f"Zoom {self.tile_size:.1f} px",
            hover_text,
        ]

    def draw_slider(self) -> None:
        rect = self.slider_rect()
        pygame.draw.rect(self.screen, SLIDER_TRACK_COLOR, rect, border_radius=4)

        normalized = (self.steps_per_second - MIN_STEPS_PER_SECOND) / (
            MAX_STEPS_PER_SECOND - MIN_STEPS_PER_SECOND
        )
        normalized = min(1.0, max(0.0, normalized))
        fill_width = max(6, int(rect.width * normalized))
        fill_rect = pygame.Rect(rect.x, rect.y, fill_width, rect.height)
        pygame.draw.rect(self.screen, SLIDER_FILL_COLOR, fill_rect, border_radius=4)

        knob_x = rect.x + int(rect.width * normalized)
        knob_center = (knob_x, rect.y + rect.height // 2)
        pygame.draw.circle(self.screen, SLIDER_KNOB_COLOR, knob_center, 7)
        pygame.draw.circle(self.screen, (0, 0, 0), knob_center, 7, 1)

    def update_slider_from_mouse(self, mouse_pos: Tuple[int, int]) -> None:
        rect = self.slider_rect()
        normalized = (mouse_pos[0] - rect.x) / rect.width
        normalized = min(1.0, max(0.0, normalized))
        self.steps_per_second = (
            MIN_STEPS_PER_SECOND
            + normalized * (MAX_STEPS_PER_SECOND - MIN_STEPS_PER_SECOND)
        )

    def hover_text(self) -> str:
        tile_pos = self.screen_to_tile(pygame.mouse.get_pos())
        if tile_pos is None:
            return "Hover off map"

        population = self.model.population_at(tile_pos)
        if population is None:
            return f"Hover {tile_pos}: 0"
        return f"Hover {tile_pos}: {population.inhabitant_count}"

    def screen_to_tile(self, screen_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x = int((screen_pos[0] - self.camera_x) // self.tile_size)
        y = int((screen_pos[1] - self.camera_y) // self.tile_size)
        if 0 <= x < self.model.width and 0 <= y < self.model.height:
            return x, y
        return None
