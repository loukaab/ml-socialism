"""Interactive Pygame viewer for the economic simulation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pygame

from model import DISPLAY_MAP_MODES, ENVIRONMENTAL_MAP_MODES, WorldModel, normalize_map_mode


Color = Tuple[int, int, int]

LAND_COLOR: Color = (46, 140, 61)
WATER_COLOR: Color = (38, 89, 199)
GRID_COLOR: Color = (24, 31, 42)
LOW_ARABLE_COLOR: Color = (89, 71, 41)
HIGH_ARABLE_COLOR: Color = (143, 209, 87)
LOW_RAW_COLOR: Color = (97, 69, 31)
HIGH_RAW_COLOR: Color = (250, 214, 56)
LOW_FACTORY_COLOR: Color = (56, 74, 82)
HIGH_FACTORY_COLOR: Color = (97, 214, 224)
LOW_DEVASTATION_COLOR: Color = (61, 115, 66)
HIGH_DEVASTATION_COLOR: Color = (204, 31, 26)
TEXT_COLOR: Color = (235, 238, 242)
PANEL_COLOR: Color = (20, 24, 31)
PANEL_BORDER: Color = (75, 84, 99)
PANEL_MUTED: Color = (35, 42, 53)
PANEL_HOVER: Color = (52, 61, 76)
PANEL_ACTIVE: Color = (86, 106, 137)
SLIDER_TRACK_COLOR: Color = (72, 79, 92)
SLIDER_FILL_COLOR: Color = (211, 217, 225)
SLIDER_KNOB_COLOR: Color = (255, 255, 255)
STAR_COLOR: Color = (255, 216, 77)
ACCENT_COLOR: Color = (111, 191, 232)
MUTED_TEXT_COLOR: Color = (160, 169, 181)
MIN_POPULATION_BRIGHTNESS = 0.35
POPULATION_BRIGHTNESS_GAMMA = 0.65
MIN_STEPS_PER_SECOND = 1.0
MAX_STEPS_PER_SECOND = 30.0
MAP_BUTTON_SIZE = 38
MAP_BUTTON_GAP = 8
QUICK_MENU_WIDTH = 164
WINDOW_TITLE_HEIGHT = 32

MAP_MODE_LABELS = {
    "terrain": "Terrain",
    "arable": "Arable Land",
    "raw": "Raw Goods",
    "manufactories": "Manufactories",
    "devastation": "Devastation",
    "tech": "Technology",
    "diplo": "Diplomacy",
    "physical": "Physical",
}
MAP_MODE_HOTKEYS = {
    "terrain": "1",
    "arable": "2",
    "raw": "3",
    "manufactories": "4",
    "tech": "5",
    "diplo": "6",
    "physical": "7",
    "devastation": "8",
}


def hex_to_rgb(value: str) -> Color:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))


def lerp_color(low: Color, high: Color, value: float) -> Color:
    value = min(1.0, max(0.0, value))
    return tuple(int(low[index] + value * (high[index] - low[index])) for index in range(3))


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
        self.map_mode_tooltip: Optional[str] = None
        self.quick_menu_pos: Optional[Tuple[int, int]] = None
        self.quick_menu_rects: Dict[str, pygame.Rect] = {}
        self.windows: List[Dict[str, object]] = []
        self.window_dragging_id: Optional[str] = None
        self.window_drag_offset = (0, 0)
        self.stats_view = "ledger"
        self.stats_selected_scopes = {"global"}
        self.stats_x_metric = "step"
        self.stats_y_metric = "inhabitants"
        self.stats_open_dropdown: Optional[str] = None
        self.stats_scope_rects: Dict[str, pygame.Rect] = {}
        self.stats_tab_rects: Dict[str, pygame.Rect] = {}
        self.stats_dropdown_rects: Dict[str, pygame.Rect] = {}
        self.stats_dropdown_option_rects: Dict[str, Tuple[str, pygame.Rect]] = {}
        self.graph_hover_text: Optional[str] = None

        self.center_map()

    @property
    def show_resource_overlay(self) -> bool:
        return normalize_map_mode(self.map_mode) == "raw"

    @show_resource_overlay.setter
    def show_resource_overlay(self, enabled: bool) -> None:
        self.map_mode = "raw" if enabled else "terrain"

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
                self.clamp_windows_to_screen()
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2, 3):
                    self.dragging = False
                    self.slider_dragging = False
                    self.window_dragging_id = None
                    self.last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if self.window_dragging_id is not None:
                    self.drag_window_to(event.pos)
                elif self.slider_dragging:
                    self.update_slider_from_mouse(event.pos)
                elif self.dragging:
                    self.pan_to(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if self.window_at(mouse_pos) is not None:
                    continue
                if self.quick_menu_pos is not None and self.quick_menu_bounds().collidepoint(mouse_pos):
                    continue
                self.zoom_at(mouse_pos, 1.12 if event.y > 0 else 1 / 1.12)

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
            self.map_mode = "arable"
        elif key in (pygame.K_3, pygame.K_KP3):
            self.map_mode = "raw"
        elif key in (pygame.K_4, pygame.K_KP4):
            self.map_mode = "manufactories"
        elif key in (pygame.K_5, pygame.K_KP5):
            self.map_mode = "tech"
        elif key in (pygame.K_6, pygame.K_KP6):
            self.map_mode = "diplo"
        elif key in (pygame.K_7, pygame.K_KP7):
            self.map_mode = "physical"
        elif key in (pygame.K_8, pygame.K_KP8):
            self.map_mode = "devastation"
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
        index = DISPLAY_MAP_MODES.index(normalize_map_mode(self.map_mode))
        self.map_mode = DISPLAY_MAP_MODES[(index + 1) % len(DISPLAY_MAP_MODES)]

    def handle_mouse_down(self, event) -> None:
        if event.button == 3:
            self.open_quick_menu(event.pos)
            return

        if event.button == 1:
            if self.handle_quick_menu_click(event.pos):
                return
            self.quick_menu_pos = None
            if self.handle_window_mouse_down(event):
                return
            if self.handle_map_mode_selector_click(event.pos):
                return
            if self.slider_rect().collidepoint(event.pos):
                self.slider_dragging = True
                self.dragging = False
                self.last_mouse_pos = None
                self.update_slider_from_mouse(event.pos)
                return
            self.dragging = True
            self.last_mouse_pos = event.pos
        elif event.button == 2:
            self.quick_menu_pos = None
            if self.handle_window_mouse_down(event):
                return
            self.dragging = True
            self.last_mouse_pos = event.pos
        elif event.button == 4:
            self.zoom_at(event.pos, 1.12)
        elif event.button == 5:
            self.zoom_at(event.pos, 1 / 1.12)

    def open_quick_menu(self, pos: Tuple[int, int]) -> None:
        screen_width, screen_height = self.screen.get_size()
        menu_height = 40
        x = min(pos[0], screen_width - QUICK_MENU_WIDTH - 8)
        y = min(pos[1], screen_height - menu_height - 8)
        self.quick_menu_pos = (max(8, x), max(8, y))
        self.quick_menu_rects = {
            "statistics": pygame.Rect(self.quick_menu_pos[0], self.quick_menu_pos[1], QUICK_MENU_WIDTH, 34)
        }
        self.dragging = False
        self.slider_dragging = False
        self.last_mouse_pos = None

    def handle_quick_menu_click(self, pos: Tuple[int, int]) -> bool:
        if self.quick_menu_pos is None:
            return False
        if self.quick_menu_rects.get("statistics", pygame.Rect(0, 0, 0, 0)).collidepoint(pos):
            self.open_stats_window(pos)
            self.quick_menu_pos = None
            return True
        menu_rect = self.quick_menu_bounds()
        self.quick_menu_pos = None
        return menu_rect.collidepoint(pos)

    def quick_menu_bounds(self) -> pygame.Rect:
        if self.quick_menu_pos is None:
            return pygame.Rect(0, 0, 0, 0)
        return pygame.Rect(self.quick_menu_pos[0], self.quick_menu_pos[1], QUICK_MENU_WIDTH, 40)

    def open_stats_window(self, pos: Optional[Tuple[int, int]] = None) -> None:
        for window in self.windows:
            if window["id"] == "statistics":
                self.bring_window_to_front(window)
                return

        screen_width, screen_height = self.screen.get_size()
        width = min(760, max(420, screen_width - 40))
        height = min(540, max(320, screen_height - 40))
        if pos is None:
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
        else:
            x = min(pos[0], screen_width - width - 12)
            y = min(pos[1], screen_height - height - 12)
        rect = pygame.Rect(max(12, x), max(12, y), width, height)
        self.windows.append(
            {
                "id": "statistics",
                "kind": "statistics",
                "title": "Statistics",
                "rect": rect,
            }
        )

    def bring_window_to_front(self, window: Dict[str, object]) -> None:
        if window in self.windows:
            self.windows.remove(window)
            self.windows.append(window)

    def window_at(self, pos: Tuple[int, int]) -> Optional[Dict[str, object]]:
        for window in reversed(self.windows):
            rect = window["rect"]
            if isinstance(rect, pygame.Rect) and rect.collidepoint(pos):
                return window
        return None

    def handle_window_mouse_down(self, event) -> bool:
        window = self.window_at(event.pos)
        if window is None:
            return False

        self.bring_window_to_front(window)
        rect = window["rect"]
        if not isinstance(rect, pygame.Rect):
            return True
        if self.window_close_rect(rect).collidepoint(event.pos):
            self.windows.remove(window)
            return True
        if self.window_title_rect(rect).collidepoint(event.pos):
            self.window_dragging_id = str(window["id"])
            self.window_drag_offset = (event.pos[0] - rect.x, event.pos[1] - rect.y)
            return True
        if window["kind"] == "statistics":
            self.handle_stats_window_click(rect, event.pos)
        return True

    def drag_window_to(self, pos: Tuple[int, int]) -> None:
        for window in self.windows:
            if window["id"] != self.window_dragging_id:
                continue
            rect = window["rect"]
            if not isinstance(rect, pygame.Rect):
                return
            rect.x = pos[0] - self.window_drag_offset[0]
            rect.y = pos[1] - self.window_drag_offset[1]
            self.clamp_window(rect)
            return

    def clamp_windows_to_screen(self) -> None:
        for window in self.windows:
            rect = window["rect"]
            if isinstance(rect, pygame.Rect):
                self.clamp_window(rect)

    def clamp_window(self, rect: pygame.Rect) -> None:
        screen_width, screen_height = self.screen.get_size()
        rect.width = min(rect.width, max(220, screen_width - 16))
        rect.height = min(rect.height, max(180, screen_height - 16))
        rect.x = min(max(8, rect.x), max(8, screen_width - rect.width - 8))
        rect.y = min(max(8, rect.y), max(8, screen_height - rect.height - 8))

    def window_title_rect(self, rect: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(rect.x, rect.y, rect.width, WINDOW_TITLE_HEIGHT)

    def window_close_rect(self, rect: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(rect.right - 28, rect.y + 6, 20, 20)

    def map_mode_button_rects(self) -> Dict[str, pygame.Rect]:
        screen_width, screen_height = self.screen.get_size()
        count = len(DISPLAY_MAP_MODES)
        total_width = count * MAP_BUTTON_SIZE + (count - 1) * MAP_BUTTON_GAP
        start_x = max(8, screen_width - total_width - 18)
        y = max(8, screen_height - MAP_BUTTON_SIZE - 18)
        return {
            mode: pygame.Rect(
                start_x + index * (MAP_BUTTON_SIZE + MAP_BUTTON_GAP),
                y,
                MAP_BUTTON_SIZE,
                MAP_BUTTON_SIZE,
            )
            for index, mode in enumerate(DISPLAY_MAP_MODES)
        }

    def map_mode_at(self, pos: Tuple[int, int]) -> Optional[str]:
        for mode, rect in self.map_mode_button_rects().items():
            if rect.collidepoint(pos):
                return mode
        return None

    def handle_map_mode_selector_click(self, pos: Tuple[int, int]) -> bool:
        mode = self.map_mode_at(pos)
        if mode is None:
            return False
        self.map_mode = mode
        return True

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
        self.map_mode_tooltip = None
        self.graph_hover_text = None
        self.draw_world()
        self.draw_status_panel()
        self.draw_map_mode_selector()
        self.draw_windows()
        self.draw_quick_menu()
        self.draw_tooltip()
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

        if normalize_map_mode(self.map_mode) not in ENVIRONMENTAL_MAP_MODES:
            self.draw_owned_tiles(start_x, start_y, end_x, end_y)
        self.draw_lineage_borders(start_x, start_y, end_x, end_y)
        self.draw_attack_arrows()
        self.draw_capital_stars()
        self.draw_hover_population_label()
        if self.tile_size >= 14:
            self.draw_grid(start_x, start_y, end_x, end_y)

    def draw_map_mode_selector(self) -> None:
        button_rects = self.map_mode_button_rects()
        if not button_rects:
            return
        bounds = list(button_rects.values())[0].copy()
        for rect in list(button_rects.values())[1:]:
            bounds.union_ip(rect)
        bounds.inflate_ip(14, 14)
        pygame.draw.rect(self.screen, PANEL_COLOR, bounds, border_radius=7)
        pygame.draw.rect(self.screen, PANEL_BORDER, bounds, width=1, border_radius=7)

        mouse_pos = pygame.mouse.get_pos()
        selected_mode = normalize_map_mode(self.map_mode)
        for mode, rect in button_rects.items():
            selected = selected_mode == mode
            hovered = rect.collidepoint(mouse_pos)
            fill = PANEL_ACTIVE if selected else (PANEL_HOVER if hovered else PANEL_MUTED)
            border = ACCENT_COLOR if selected else PANEL_BORDER
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            pygame.draw.rect(self.screen, border, rect, width=2 if selected else 1, border_radius=6)
            self.draw_map_mode_icon(mode, rect, selected)
            if hovered:
                label = MAP_MODE_LABELS[mode]
                hotkey = MAP_MODE_HOTKEYS[mode]
                self.map_mode_tooltip = f"{label} ({hotkey})"

    def draw_map_mode_icon(self, mode: str, rect: pygame.Rect, selected: bool) -> None:
        color = TEXT_COLOR if selected else (210, 217, 226)
        center = rect.center
        if mode == "terrain":
            pygame.draw.rect(self.screen, WATER_COLOR, rect.inflate(-14, -14), border_radius=3)
            land = pygame.Rect(rect.x + 11, rect.y + 18, 17, 10)
            pygame.draw.ellipse(self.screen, LAND_COLOR, land)
        elif mode == "arable":
            pygame.draw.line(self.screen, color, (center[0], rect.y + 27), (center[0], rect.y + 13), 2)
            pygame.draw.arc(self.screen, HIGH_ARABLE_COLOR, (center[0] - 15, rect.y + 11, 16, 15), 0, math.pi, 2)
            pygame.draw.arc(self.screen, HIGH_ARABLE_COLOR, (center[0] - 1, rect.y + 11, 16, 15), 0, math.pi, 2)
        elif mode == "raw":
            points = [(center[0], rect.y + 10), (rect.x + 28, center[1]), (center[0], rect.y + 28), (rect.x + 10, center[1])]
            pygame.draw.polygon(self.screen, HIGH_RAW_COLOR, points)
            pygame.draw.polygon(self.screen, (65, 49, 30), points, width=2)
        elif mode == "manufactories":
            body = pygame.Rect(rect.x + 10, rect.y + 18, 20, 11)
            pygame.draw.rect(self.screen, HIGH_FACTORY_COLOR, body)
            roof = [(rect.x + 10, rect.y + 18), (rect.x + 15, rect.y + 13), (rect.x + 20, rect.y + 18), (rect.x + 25, rect.y + 13), (rect.x + 30, rect.y + 18)]
            pygame.draw.lines(self.screen, HIGH_FACTORY_COLOR, False, roof, 3)
            pygame.draw.rect(self.screen, color, pygame.Rect(rect.x + 13, rect.y + 21, 4, 4))
            pygame.draw.rect(self.screen, color, pygame.Rect(rect.x + 22, rect.y + 21, 4, 4))
        elif mode == "devastation":
            flame = [
                (center[0], rect.y + 9),
                (rect.x + 27, rect.y + 21),
                (center[0] + 6, rect.y + 30),
                (center[0], rect.y + 25),
                (center[0] - 6, rect.y + 30),
                (rect.x + 11, rect.y + 21),
            ]
            pygame.draw.polygon(self.screen, HIGH_DEVASTATION_COLOR, flame)
            pygame.draw.polygon(self.screen, (255, 176, 77), [(center[0], rect.y + 17), (center[0] + 4, rect.y + 26), (center[0] - 4, rect.y + 26)])
        elif mode == "tech":
            nodes = [(center[0], rect.y + 11), (rect.x + 12, rect.y + 26), (rect.x + 27, rect.y + 26)]
            pygame.draw.lines(self.screen, color, True, nodes, 2)
            for node in nodes:
                pygame.draw.circle(self.screen, (180, 165, 255), node, 4)
        elif mode == "diplo":
            pygame.draw.circle(self.screen, (255, 190, 218), (rect.x + 15, center[1]), 6)
            pygame.draw.circle(self.screen, (255, 190, 218), (rect.x + 25, center[1]), 6)
            pygame.draw.line(self.screen, color, (rect.x + 18, center[1]), (rect.x + 22, center[1]), 2)
        elif mode == "physical":
            pygame.draw.circle(self.screen, (225, 42, 42), (rect.x + 18, center[1]), 8)
            pygame.draw.circle(self.screen, (250, 220, 42), (rect.x + 22, center[1]), 8)
            pygame.draw.line(self.screen, (30, 30, 30), (center[0], rect.y + 10), (center[0], rect.y + 28), 2)

    def draw_quick_menu(self) -> None:
        if self.quick_menu_pos is None:
            return
        rect = self.quick_menu_bounds()
        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=6)
        item = self.quick_menu_rects.get("statistics")
        if item is None:
            return
        if item.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(self.screen, PANEL_HOVER, item, border_radius=5)
        surface = self.font.render("Statistics", True, TEXT_COLOR)
        self.screen.blit(surface, (item.x + 12, item.y + 8))

    def draw_windows(self) -> None:
        for window in self.windows:
            rect = window["rect"]
            if not isinstance(rect, pygame.Rect):
                continue
            self.draw_window_frame(window, rect)
            if window["kind"] == "statistics":
                self.draw_stats_window(rect)

    def draw_window_frame(self, window: Dict[str, object], rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=7)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=7)
        title_rect = self.window_title_rect(rect)
        pygame.draw.rect(self.screen, PANEL_MUTED, title_rect, border_radius=7)
        pygame.draw.line(self.screen, PANEL_BORDER, (rect.x, title_rect.bottom), (rect.right, title_rect.bottom), 1)
        title = str(window.get("title", "Window"))
        self.screen.blit(self.font.render(title, True, TEXT_COLOR), (rect.x + 12, rect.y + 7))
        close_rect = self.window_close_rect(rect)
        close_fill = PANEL_HOVER if close_rect.collidepoint(pygame.mouse.get_pos()) else PANEL_MUTED
        pygame.draw.rect(self.screen, close_fill, close_rect, border_radius=4)
        pygame.draw.line(self.screen, TEXT_COLOR, (close_rect.x + 5, close_rect.y + 5), (close_rect.right - 5, close_rect.bottom - 5), 2)
        pygame.draw.line(self.screen, TEXT_COLOR, (close_rect.right - 5, close_rect.y + 5), (close_rect.x + 5, close_rect.bottom - 5), 2)

    def handle_stats_window_click(self, rect: pygame.Rect, pos: Tuple[int, int]) -> None:
        if self.stats_open_dropdown is not None:
            for option_key, (metric_key, option_rect) in self.stats_dropdown_option_rects.items():
                if option_rect.collidepoint(pos):
                    if option_key.startswith("x:"):
                        self.stats_x_metric = metric_key
                    elif option_key.startswith("y:"):
                        self.stats_y_metric = metric_key
                    self.stats_open_dropdown = None
                    return

        for view, tab_rect in self.stats_tab_rects.items():
            if tab_rect.collidepoint(pos):
                self.stats_view = view
                self.stats_open_dropdown = None
                return

        for scope_key, scope_rect in self.stats_scope_rects.items():
            if scope_rect.collidepoint(pos):
                if scope_key in self.stats_selected_scopes:
                    self.stats_selected_scopes.remove(scope_key)
                else:
                    self.stats_selected_scopes.add(scope_key)
                self.stats_open_dropdown = None
                return

        for dropdown_id, dropdown_rect in self.stats_dropdown_rects.items():
            if dropdown_rect.collidepoint(pos):
                self.stats_open_dropdown = None if self.stats_open_dropdown == dropdown_id else dropdown_id
                return

        self.stats_open_dropdown = None

    def draw_stats_window(self, rect: pygame.Rect) -> None:
        self.stats_scope_rects.clear()
        self.stats_tab_rects.clear()
        self.stats_dropdown_rects.clear()
        self.stats_dropdown_option_rects.clear()

        content = pygame.Rect(rect.x + 12, rect.y + WINDOW_TITLE_HEIGHT + 12, rect.width - 24, rect.height - WINDOW_TITLE_HEIGHT - 24)
        self.draw_stats_tabs(content)
        selector = pygame.Rect(content.x, content.y + 42, 166, content.height - 42)
        data_rect = pygame.Rect(selector.right + 12, selector.y, content.right - selector.right - 12, selector.height)
        self.draw_stats_scope_selector(selector)
        if self.stats_view == "ledger":
            self.draw_stats_ledger(data_rect)
        else:
            self.draw_stats_graph(data_rect)

    def draw_stats_tabs(self, content: pygame.Rect) -> None:
        for index, view in enumerate(("ledger", "graph")):
            rect = pygame.Rect(content.x + index * 96, content.y, 88, 30)
            self.stats_tab_rects[view] = rect
            selected = self.stats_view == view
            pygame.draw.rect(self.screen, PANEL_ACTIVE if selected else PANEL_MUTED, rect, border_radius=5)
            pygame.draw.rect(self.screen, ACCENT_COLOR if selected else PANEL_BORDER, rect, width=1, border_radius=5)
            label = "Ledger" if view == "ledger" else "Graph"
            self.screen.blit(self.small_font.render(label, True, TEXT_COLOR), (rect.x + 14, rect.y + 8))

    def draw_stats_scope_selector(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (16, 20, 27), rect, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=5)
        self.screen.blit(self.small_font.render("Lineages", True, TEXT_COLOR), (rect.x + 10, rect.y + 8))

        y = rect.y + 34
        for scope_key, row in self.scope_entries():
            row_rect = pygame.Rect(rect.x + 8, y, rect.width - 16, 24)
            self.stats_scope_rects[scope_key] = row_rect
            selected = scope_key in self.stats_selected_scopes
            pygame.draw.rect(self.screen, PANEL_HOVER if row_rect.collidepoint(pygame.mouse.get_pos()) else PANEL_COLOR, row_rect, border_radius=4)
            box = pygame.Rect(row_rect.x + 4, row_rect.y + 5, 13, 13)
            pygame.draw.rect(self.screen, PANEL_ACTIVE if selected else PANEL_MUTED, box, border_radius=3)
            pygame.draw.rect(self.screen, ACCENT_COLOR if selected else PANEL_BORDER, box, width=1, border_radius=3)
            if selected:
                pygame.draw.line(self.screen, TEXT_COLOR, (box.x + 3, box.y + 7), (box.x + 6, box.bottom - 3), 2)
                pygame.draw.line(self.screen, TEXT_COLOR, (box.x + 6, box.bottom - 3), (box.right - 3, box.y + 3), 2)
            swatch = pygame.Rect(row_rect.x + 22, row_rect.y + 6, 11, 11)
            pygame.draw.rect(self.screen, hex_to_rgb(str(row["color"])), swatch, border_radius=2)
            label = str(row["label"])
            if row.get("defeated"):
                label = f"{label} Defeated"
            color = MUTED_TEXT_COLOR if row.get("defeated") else TEXT_COLOR
            self.blit_clipped(label, self.small_font, color, pygame.Rect(row_rect.x + 38, row_rect.y + 5, row_rect.width - 42, 16))
            y += 27
            if y > rect.bottom - 24:
                break

    def draw_stats_ledger(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (16, 20, 27), rect, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=5)
        rows = self.selected_scope_entries()
        if not rows:
            self.screen.blit(self.font.render("Select a lineage to view accounts.", True, MUTED_TEXT_COLOR), (rect.x + 18, rect.y + 20))
            return

        headers = [
            ("Account", 0, 84),
            ("Pop", 88, 46),
            ("Tiles", 136, 36),
            ("GDP", 174, 54),
            ("GDP/c", 230, 54),
            ("Food", 286, 54),
            ("Raw", 342, 48),
            ("Ref", 392, 48),
            ("Jobs F/E/M/A", 444, 104),
        ]
        origin_x = rect.x + 10
        y = rect.y + 10
        for text, offset, width in headers:
            self.blit_clipped(text, self.small_font, MUTED_TEXT_COLOR, pygame.Rect(origin_x + offset, y, width, 18))
        y += 22

        clip = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-4, -4))
        for _, row in rows:
            if y + 44 > rect.bottom:
                break
            pygame.draw.line(self.screen, PANEL_BORDER, (rect.x + 8, y - 3), (rect.right - 8, y - 3), 1)
            account_label = str(row["label"])
            if row.get("defeated"):
                account_label = f"{account_label} Defeated"
            values = [
                account_label,
                self.format_stat_value(row["inhabitants"], "int"),
                self.format_stat_value(row["occupied_tiles"], "int"),
                self.format_stat_value(row["gdp"], "float"),
                self.format_stat_value(row["gdp_per_capita"], "float"),
                self.format_stat_value(row["food_stockpile"], "float"),
                self.format_stat_value(row["raw_stockpile"], "float"),
                self.format_stat_value(row["refined_stockpile"], "float"),
                f"{row['farmers']}/{row['extractors']}/{row['manufacturers']}/{row['artisans']}",
            ]
            for value, (_, offset, width) in zip(values, headers):
                self.blit_clipped(value, self.small_font, TEXT_COLOR, pygame.Rect(origin_x + offset, y, width, 18))
            y += 20
            detail = (
                f"Prod food {self.format_stat_value(row['food_produced'], 'float')}  "
                f"raw {self.format_stat_value(row['raw_extracted'], 'float')}  "
                f"ref {self.format_stat_value(row['refined_produced'], 'float')}  "
                f"births {self.format_stat_value(row['births'], 'int')}  "
                f"mfg {self.format_stat_value(row['manufactories'], 'int')}  "
                f"tech {self.format_stat_value(row['avg_tech'], 'float')}/{self.format_stat_value(row['max_tech'], 'int')}  "
                f"dev {self.format_stat_value(row['avg_devastation'], 'float')}/{self.format_stat_value(row['max_devastation'], 'float')}"
            )
            self.blit_clipped(detail, self.small_font, MUTED_TEXT_COLOR, pygame.Rect(origin_x, y, rect.width - 18, 18))
            y += 26
        self.screen.set_clip(clip)

    def draw_stats_graph(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (16, 20, 27), rect, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=5)
        x_rect = pygame.Rect(rect.x + 48, rect.y + 12, 150, 26)
        y_rect = pygame.Rect(rect.x + 246, rect.y + 12, 170, 26)
        self.screen.blit(self.small_font.render("X", True, MUTED_TEXT_COLOR), (rect.x + 28, rect.y + 19))
        self.screen.blit(self.small_font.render("Y", True, MUTED_TEXT_COLOR), (rect.x + 226, rect.y + 19))
        self.draw_metric_dropdown("x", x_rect, self.stats_x_metric)
        self.draw_metric_dropdown("y", y_rect, self.stats_y_metric)

        plot = pygame.Rect(rect.x + 42, rect.y + 62, rect.width - 74, rect.height - 112)
        pygame.draw.rect(self.screen, (10, 13, 18), plot)
        pygame.draw.rect(self.screen, PANEL_BORDER, plot, width=1)
        series = self.graph_series(plot.width)
        if not series:
            self.screen.blit(self.font.render("Select numeric data to graph.", True, MUTED_TEXT_COLOR), (plot.x + 18, plot.y + 18))
            self.draw_open_metric_dropdowns()
            return

        all_x = [point[0] for _, _, points in series for point in points]
        all_y = [point[1] for _, _, points in series for point in points]
        x_min, x_max = self.padded_range(min(all_x), max(all_x))
        y_min, y_max = self.padded_range(min(all_y), max(all_y))
        self.draw_graph_grid(plot, x_min, x_max, y_min, y_max)

        mouse_pos = pygame.mouse.get_pos()
        closest = None
        for label, color, points in series:
            screen_points = [
                self.graph_to_screen(point[0], point[1], plot, x_min, x_max, y_min, y_max)
                for point in points
            ]
            if len(screen_points) >= 2:
                pygame.draw.lines(self.screen, color, False, screen_points, 2)
            for screen_point, raw_point in zip(screen_points, points):
                if plot.collidepoint(mouse_pos):
                    distance = math.hypot(screen_point[0] - mouse_pos[0], screen_point[1] - mouse_pos[1])
                    if closest is None or distance < closest[0]:
                        closest = (distance, label, raw_point)
            if screen_points:
                pygame.draw.circle(self.screen, color, screen_points[-1], 3)

        legend_x = plot.x + 8
        legend_y = plot.y + 8
        for label, color, _ in series[:6]:
            pygame.draw.rect(self.screen, color, pygame.Rect(legend_x, legend_y + 4, 10, 10), border_radius=2)
            self.blit_clipped(label, self.small_font, TEXT_COLOR, pygame.Rect(legend_x + 16, legend_y, 94, 18))
            legend_y += 18

        if closest is not None and closest[0] <= 16:
            _, label, point = closest
            self.graph_hover_text = (
                f"{label}: {self.metric_label(self.stats_x_metric)} "
                f"{point[0]:.2f}, {self.metric_label(self.stats_y_metric)} {point[1]:.2f}"
            )
        self.draw_open_metric_dropdowns()

    def draw_metric_dropdown(self, dropdown_id: str, rect: pygame.Rect, metric_key: str) -> None:
        self.stats_dropdown_rects[dropdown_id] = rect
        active = self.stats_open_dropdown == dropdown_id
        pygame.draw.rect(self.screen, PANEL_ACTIVE if active else PANEL_MUTED, rect, border_radius=4)
        pygame.draw.rect(self.screen, ACCENT_COLOR if active else PANEL_BORDER, rect, width=1, border_radius=4)
        self.blit_clipped(self.metric_label(metric_key), self.small_font, TEXT_COLOR, pygame.Rect(rect.x + 8, rect.y + 6, rect.width - 24, 16))
        pygame.draw.polygon(
            self.screen,
            TEXT_COLOR,
            [(rect.right - 14, rect.y + 10), (rect.right - 6, rect.y + 10), (rect.right - 10, rect.y + 16)],
        )

    def draw_open_metric_dropdowns(self) -> None:
        if self.stats_open_dropdown is None:
            return
        anchor = self.stats_dropdown_rects.get(self.stats_open_dropdown)
        if anchor is None:
            return
        metrics = [metric for metric in self.model.available_stat_metrics() if metric.get("graph")]
        columns = 2
        option_width = 178
        option_height = 24
        rows = math.ceil(len(metrics) / columns)
        popup = pygame.Rect(anchor.x, anchor.bottom + 4, columns * option_width, rows * option_height + 8)
        screen_width, screen_height = self.screen.get_size()
        if popup.right > screen_width - 8:
            popup.right = screen_width - 8
        if popup.bottom > screen_height - 8:
            popup.bottom = screen_height - 8
        pygame.draw.rect(self.screen, PANEL_COLOR, popup, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_BORDER, popup, width=1, border_radius=5)
        for index, metric in enumerate(metrics):
            column = index // rows
            row = index % rows
            option_rect = pygame.Rect(
                popup.x + 4 + column * option_width,
                popup.y + 4 + row * option_height,
                option_width - 8,
                option_height,
            )
            option_key = f"{self.stats_open_dropdown}:{metric['key']}"
            self.stats_dropdown_option_rects[option_key] = (str(metric["key"]), option_rect)
            if option_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.screen, PANEL_HOVER, option_rect, border_radius=4)
            selected_key = self.stats_x_metric if self.stats_open_dropdown == "x" else self.stats_y_metric
            color = ACCENT_COLOR if metric["key"] == selected_key else TEXT_COLOR
            self.blit_clipped(str(metric["label"]), self.small_font, color, pygame.Rect(option_rect.x + 6, option_rect.y + 5, option_rect.width - 12, 16))

    def scope_entries(self) -> List[Tuple[str, Dict[str, object]]]:
        snapshot = self.model.current_stats_snapshot()
        entries: List[Tuple[str, Dict[str, object]]] = [("global", snapshot["global"])]
        lineages = snapshot["lineages"]
        if isinstance(lineages, dict):
            for nation_id in sorted(lineages):
                entries.append((f"nation:{nation_id}", lineages[nation_id]))
        return entries

    def selected_scope_entries(self) -> List[Tuple[str, Dict[str, object]]]:
        return [
            (scope_key, row)
            for scope_key, row in self.scope_entries()
            if scope_key in self.stats_selected_scopes
        ]

    def row_for_scope(self, snapshot: Dict[str, object], scope_key: str) -> Optional[Dict[str, object]]:
        if scope_key == "global":
            return snapshot["global"]
        if not scope_key.startswith("nation:"):
            return None
        try:
            nation_id = int(scope_key.split(":", 1)[1])
        except ValueError:
            return None
        lineages = snapshot.get("lineages", {})
        if not isinstance(lineages, dict):
            return None
        return lineages.get(nation_id)

    def graph_series(self, max_points: int) -> List[Tuple[str, Color, List[Tuple[float, float]]]]:
        series = []
        current_rows = dict(self.scope_entries())
        ordered_keys = [scope_key for scope_key, _ in self.scope_entries() if scope_key in self.stats_selected_scopes]
        for scope_key in ordered_keys:
            current_row = current_rows.get(scope_key)
            if current_row is None:
                continue
            points = []
            for snapshot in self.model.stats_history:
                row = self.row_for_scope(snapshot, scope_key)
                if row is None:
                    continue
                x_value = row.get(self.stats_x_metric)
                y_value = row.get(self.stats_y_metric)
                if self.is_number(x_value) and self.is_number(y_value):
                    points.append((float(x_value), float(y_value)))
            if not points:
                continue
            points.sort(key=lambda point: point[0])
            points = self.downsample_points(points, max(2, max_points))
            series.append((str(current_row["label"]), hex_to_rgb(str(current_row["color"])), points))
        return series

    def downsample_points(
        self,
        points: List[Tuple[float, float]],
        max_points: int,
    ) -> List[Tuple[float, float]]:
        if len(points) <= max_points:
            return points
        stride = max(1, math.ceil(len(points) / max_points))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        return sampled

    def draw_graph_grid(
        self,
        plot: pygame.Rect,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        for index in range(6):
            x = plot.x + int(plot.width * index / 5)
            y = plot.y + int(plot.height * index / 5)
            pygame.draw.line(self.screen, (27, 34, 45), (x, plot.y), (x, plot.bottom), 1)
            pygame.draw.line(self.screen, (27, 34, 45), (plot.x, y), (plot.right, y), 1)
        x_label = f"{self.metric_label(self.stats_x_metric)} {x_min:.1f}-{x_max:.1f}"
        y_label = f"{self.metric_label(self.stats_y_metric)} {y_min:.1f}-{y_max:.1f}"
        self.blit_clipped(x_label, self.small_font, MUTED_TEXT_COLOR, pygame.Rect(plot.x, plot.bottom + 8, plot.width // 2, 18))
        self.blit_clipped(y_label, self.small_font, MUTED_TEXT_COLOR, pygame.Rect(plot.centerx, plot.bottom + 8, plot.width // 2, 18))

    def graph_to_screen(
        self,
        x_value: float,
        y_value: float,
        plot: pygame.Rect,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> Tuple[int, int]:
        x_ratio = (x_value - x_min) / (x_max - x_min)
        y_ratio = (y_value - y_min) / (y_max - y_min)
        return (
            plot.x + int(x_ratio * plot.width),
            plot.bottom - int(y_ratio * plot.height),
        )

    def padded_range(self, minimum: float, maximum: float) -> Tuple[float, float]:
        if minimum == maximum:
            padding = max(1.0, abs(minimum) * 0.1)
            return minimum - padding, maximum + padding
        padding = (maximum - minimum) * 0.06
        return minimum - padding, maximum + padding

    def is_number(self, value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def metric_label(self, metric_key: str) -> str:
        for metric in self.model.available_stat_metrics():
            if metric["key"] == metric_key:
                return str(metric["label"])
        return metric_key

    def metric_format(self, metric_key: str) -> str:
        for metric in self.model.available_stat_metrics():
            if metric["key"] == metric_key:
                return str(metric["format"])
        return "float"

    def format_stat_value(self, value: object, value_format: str) -> str:
        if value is None:
            return "--"
        if value_format == "int":
            return f"{int(value):,}"
        if value_format == "percent":
            return f"{float(value) * 100:.0f}%"
        if value_format == "float":
            return f"{float(value):,.1f}"
        return str(value)

    def blit_clipped(
        self,
        text: str,
        font,
        color: Color,
        rect: pygame.Rect,
    ) -> None:
        surface = font.render(text, True, color)
        if surface.get_width() <= rect.width:
            self.screen.blit(surface, rect.topleft)
            return
        clipped = text
        while clipped and font.size(clipped + "...")[0] > rect.width:
            clipped = clipped[:-1]
        self.screen.blit(font.render(clipped + "...", True, color), rect.topleft)

    def draw_tooltip(self) -> None:
        text = self.map_mode_tooltip or self.graph_hover_text
        if not text:
            return
        mouse_x, mouse_y = pygame.mouse.get_pos()
        surface = self.small_font.render(text, True, TEXT_COLOR)
        rect = surface.get_rect()
        rect.x = mouse_x - rect.width - 12
        rect.y = mouse_y - rect.height - 12
        if rect.x < 8:
            rect.x = mouse_x + 12
        if rect.y < 8:
            rect.y = mouse_y + 12
        rect.inflate_ip(12, 8)
        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=5)
        self.screen.blit(surface, (rect.x + 6, rect.y + 4))

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

    def draw_owned_tiles(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> None:
        tile = max(1, int(self.tile_size + 1))
        global_max_population = self.model.global_max_population()
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                population = self.model.population_by_pos.get((x, y))
                if population is None:
                    continue
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
                population = self.model.population_by_pos.get((x, y))
                if population is None:
                    continue

                left = int(self.camera_x + x * self.tile_size)
                top = int(self.camera_y + y * self.tile_size)
                right = int(self.camera_x + (x + 1) * self.tile_size)
                bottom = int(self.camera_y + (y + 1) * self.tile_size)

                if self.has_border(population, x - 1, y):
                    pygame.draw.line(self.screen, (0, 0, 0), (left, top), (left, bottom), thickness)
                if self.has_border(population, x + 1, y):
                    pygame.draw.line(self.screen, (0, 0, 0), (right, top), (right, bottom), thickness)
                if self.has_border(population, x, y - 1):
                    pygame.draw.line(self.screen, (0, 0, 0), (left, top), (right, top), thickness)
                if self.has_border(population, x, y + 1):
                    pygame.draw.line(self.screen, (0, 0, 0), (left, bottom), (right, bottom), thickness)

    def has_border(self, population, neighbor_x: int, neighbor_y: int) -> bool:
        if not (0 <= neighbor_x < self.model.width and 0 <= neighbor_y < self.model.height):
            return True
        neighbor = self.model.population_by_pos.get((neighbor_x, neighbor_y))
        if neighbor is None:
            return True
        return neighbor.nation is not population.nation

    def draw_attack_arrows(self) -> None:
        for arrow in self.model.attack_arrows:
            start = self.tile_center(arrow.source)
            end = self.tile_center(arrow.target)
            self.draw_arrow(start, end)

    def tile_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        return (
            int(self.camera_x + (x + 0.5) * self.tile_size),
            int(self.camera_y + (y + 0.5) * self.tile_size),
        )

    def draw_arrow(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 1:
            return

        ux = dx / distance
        uy = dy / distance
        arrowhead_length = max(8.0, self.tile_size * 0.45)
        arrowhead_width = max(5.0, self.tile_size * 0.22)
        shaft_end = (
            int(end[0] - ux * arrowhead_length * 0.75),
            int(end[1] - uy * arrowhead_length * 0.75),
        )
        line_width = max(3, min(8, int(self.tile_size * 0.16)))

        pygame.draw.line(self.screen, (0, 0, 0), start, shaft_end, line_width + 2)
        pygame.draw.line(self.screen, (220, 36, 36), start, shaft_end, line_width)

        perp_x = -uy
        perp_y = ux
        left = (
            int(end[0] - ux * arrowhead_length + perp_x * arrowhead_width),
            int(end[1] - uy * arrowhead_length + perp_y * arrowhead_width),
        )
        right = (
            int(end[0] - ux * arrowhead_length - perp_x * arrowhead_width),
            int(end[1] - uy * arrowhead_length - perp_y * arrowhead_width),
        )
        pygame.draw.polygon(self.screen, (0, 0, 0), [end, left, right])
        inset_left = (
            int(end[0] - ux * (arrowhead_length - 2) + perp_x * (arrowhead_width - 1)),
            int(end[1] - uy * (arrowhead_length - 2) + perp_y * (arrowhead_width - 1)),
        )
        inset_right = (
            int(end[0] - ux * (arrowhead_length - 2) - perp_x * (arrowhead_width - 1)),
            int(end[1] - uy * (arrowhead_length - 2) - perp_y * (arrowhead_width - 1)),
        )
        pygame.draw.polygon(self.screen, (220, 36, 36), [end, inset_left, inset_right])

    def draw_capital_stars(self) -> None:
        if self.tile_size < 6:
            return
        for nation in self.model.surviving_nations():
            if nation.capital_pos is None:
                continue
            center = self.tile_center(nation.capital_pos)
            self.draw_star(center, max(4, int(self.tile_size * 0.32)))

    def draw_star(self, center: Tuple[int, int], radius: int) -> None:
        points = []
        inner = max(2, int(radius * 0.45))
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            active_radius = radius if index % 2 == 0 else inner
            points.append(
                (
                    int(center[0] + math.cos(angle) * active_radius),
                    int(center[1] + math.sin(angle) * active_radius),
                )
            )
        pygame.draw.polygon(self.screen, (0, 0, 0), points)
        inset_points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            active_radius = max(1, radius - 2) if index % 2 == 0 else max(1, inner - 1)
            inset_points.append(
                (
                    int(center[0] + math.cos(angle) * active_radius),
                    int(center[1] + math.sin(angle) * active_radius),
                )
            )
        pygame.draw.polygon(self.screen, STAR_COLOR, inset_points)

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
        mode = normalize_map_mode(self.map_mode)
        if mode == "tech":
            return self.dark_investment_color(
                population.x_tech,
                max_value=0.3,
                light=(222, 210, 255),
                dark=(44, 22, 92),
            )
        if mode == "diplo":
            return self.dark_investment_color(
                population.y_dip,
                max_value=0.3,
                light=(255, 212, 232),
                dark=(95, 18, 58),
            )
        if mode == "physical":
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

        mode = normalize_map_mode(self.map_mode)
        if mode == "arable":
            return lerp_color(LOW_ARABLE_COLOR, HIGH_ARABLE_COLOR, float(self.model.arable_map[y, x]))
        if mode == "raw":
            return lerp_color(LOW_RAW_COLOR, HIGH_RAW_COLOR, float(self.model.raw_goods_map[y, x]))
        if mode == "manufactories":
            cell = self.model.resource_cell_at((x, y))
            value = 0.0 if cell is None or cell.manufactory_level <= 0 else 1.0
            return lerp_color(LOW_FACTORY_COLOR, HIGH_FACTORY_COLOR, value)
        if mode == "devastation":
            max_devastation = max(1e-9, self.model.economy_config.devastation_max)
            value = self.model.devastation_at((x, y)) / max_devastation
            return lerp_color(LOW_DEVASTATION_COLOR, HIGH_DEVASTATION_COLOR, value)
        return LAND_COLOR

    def draw_status_panel(self) -> None:
        rows = self.status_rows()

        width = 260
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
        return pygame.Rect(26, 24 + row_count * 24 + 8, 200, 8)

    def status_rows(self) -> list[str]:
        latest = getattr(self.model.datacollector, "latest_model_record", None)
        if latest is None:
            latest = self.model.datacollector.get_model_vars_dataframe().iloc[-1]
        hover_text = self.hover_text()
        return [
            f"Step {int(latest['Step'])}",
            f"{'Playing' if self.playing else 'Paused'}",
            f"Populations {int(latest['PopulationAgents'])}",
            f"Inhabitants {int(latest['TotalInhabitants'])}",
            f"Lineages {int(latest['SurvivingLineages'])}",
            f"GDP {float(latest['GDP']):.1f}",
            f"Food {float(latest['FoodStockpile']):.1f}",
            f"Refined {float(latest['RefinedStockpile']):.1f}",
            f"Manufactories {int(latest['Manufactories'])}",
            f"Conquests {int(latest['ConquestEvents'])}",
            f"Max tech {int(latest['MaxTech'])}",
            f"Dominant {latest['DominantTrait']}",
            f"Map {normalize_map_mode(self.map_mode).title()}",
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

        cell = self.model.resource_cell_at(tile_pos)
        population = self.model.population_at(tile_pos)
        if population is None:
            if cell is None:
                return f"Hover {tile_pos}: empty"
            return (
                f"Hover {tile_pos}: A{cell.arable_value:.2f} "
                f"R{cell.raw_goods_value:.2f} D{cell.devastation:.1f}"
            )
        return (
            f"Hover {tile_pos}: {population.inhabitant_count} "
            f"F{population.last_farmers} M{population.last_manufacturers} "
            f"D{cell.devastation:.1f}"
        )

    def screen_to_tile(self, screen_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x = int((screen_pos[0] - self.camera_x) // self.tile_size)
        y = int((screen_pos[1] - self.camera_y) // self.tile_size)
        if 0 <= x < self.model.width and 0 <= y < self.model.height:
            return x, y
        return None
