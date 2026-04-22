"""Interactive Pygame viewer for the economic simulation."""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from model import WorldModel


Color = Tuple[int, int, int]

LAND_COLOR: Color = (46, 140, 61)
WATER_COLOR: Color = (38, 89, 199)
GRID_COLOR: Color = (24, 31, 42)
TEXT_COLOR: Color = (235, 238, 242)
PANEL_COLOR: Color = (20, 24, 31)
PANEL_BORDER: Color = (75, 84, 99)


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
        self.fps = fps

        self.tile_size = initial_tile_size
        self.min_tile_size = 5.0
        self.max_tile_size = 95.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.dragging = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.running = True
        self.playing = False
        self.steps_per_second = 4.0
        self.step_accumulator = 0.0

        self.center_map()

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
                    self.last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and self.dragging:
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

    def handle_mouse_down(self, event) -> None:
        if event.button in (1, 2, 3):
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
                color = LAND_COLOR if self.model.terrain_map[y, x] else WATER_COLOR
                rect = pygame.Rect(
                    int(self.camera_x + x * self.tile_size),
                    int(self.camera_y + y * self.tile_size),
                    tile,
                    tile,
                )
                pygame.draw.rect(self.screen, color, rect)

        if self.tile_size >= 14:
            self.draw_grid(start_x, start_y, end_x, end_y)
        self.draw_populations()

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

    def draw_populations(self) -> None:
        for population in self.model.populations:
            x, y = population.pos
            center = (
                int(self.camera_x + (x + 0.5) * self.tile_size),
                int(self.camera_y + (y + 0.5) * self.tile_size),
            )
            radius = int(max(4, min(self.tile_size * 0.42, population.inhabitant_count / 8)))
            pygame.draw.circle(
                self.screen,
                hex_to_rgb(population.lineage_color),
                center,
                radius,
            )
            pygame.draw.circle(self.screen, (245, 246, 248), center, radius, 2)

            if self.tile_size >= 22:
                label = self.small_font.render(str(population.tech_level), True, TEXT_COLOR)
                label_rect = label.get_rect(center=center)
                self.screen.blit(label, label_rect)

    def draw_status_panel(self) -> None:
        latest = self.model.datacollector.get_model_vars_dataframe().iloc[-1]
        rows = [
            f"Step {int(latest['Step'])}",
            f"{'Playing' if self.playing else 'Paused'}",
            f"Populations {int(latest['PopulationAgents'])}",
            f"Lineages {int(latest['SurvivingLineages'])}",
            f"Max tech {int(latest['MaxTech'])}",
            f"Dominant {latest['DominantTrait']}",
            f"Zoom {self.tile_size:.1f} px",
        ]

        width = 190
        height = 12 + len(rows) * 24
        panel = pygame.Rect(14, 14, width, height)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, width=1, border_radius=6)

        for index, row in enumerate(rows):
            surface = self.font.render(row, True, TEXT_COLOR)
            self.screen.blit(surface, (26, 24 + index * 24))
