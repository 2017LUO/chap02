"""Standalone domain-knowledge visualizer for Crafter and Highway.

Run:
    python tests/domain_knowledge_visualizer.py
"""

from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import create_env  # noqa: E402
from src.chap02_dk_abs.domain_knowledge import (  # noqa: E402
    build_topology,
    extract_region_rules,
    extract_skill_descriptors,
)


BG = "#10243a"
PANEL = "#17324b"
PANEL_2 = "#22435f"
CANVAS_BG = "#142b42"
CANVAS_ALT = "#18344e"
GRID_LINE = "#2a516d"
INK = "#f4f8ff"
MUTED = "#b8c7d8"
CYAN = "#4dd8ff"
BLUE = "#4f8cff"
GREEN = "#4ff0a3"
AMBER = "#ffd166"
RED = "#ff5c77"
PURPLE = "#b482ff"


SCENES: dict[str, dict[str, Any]] = {
    "crafter": {
        "label": "Crafter",
        "title": "Crafter 领域知识可视化",
        "subtitle": "石镐制作任务：资源、工作台、危险区域与技能链",
        "config": {
            "backend": "gym",
            "env_id": "CrafterReward-v1",
            "wood_target_count": 4,
            "stone_required_for_stone_pickaxe": 1,
        },
        "accent": GREEN,
    },
    "highway": {
        "label": "Highway",
        "title": "Highway 领域知识可视化",
        "subtitle": "分层驾驶任务：车道拓扑、安全规则与离散技能策略",
        "config": {
            "backend": "gymnasium",
            "env_id": "highway-v0",
            "render_mode": "rgb_array",
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 100,
        },
        "accent": CYAN,
    },
}


class DomainKnowledgeVisualizer:
    """Separate runtime viewer for environment topology, skills and rules."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("领域知识可视化控制台")
        self.root.tk.call("tk", "scaling", 1.12)
        self.root.geometry("1500x900")
        self.root.minsize(1280, 760)
        self.root.configure(bg=BG)

        self.scene = "crafter"
        self.domain: dict[str, Any] = {}
        self.topology: dict[str, Any] = {}
        self.skills: dict[str, str] = {}
        self.rules: dict[str, Any] = {}
        self.obs: dict[str, Any] = {}
        self.task_state: dict[str, Any] = {}
        self.safety_state: dict[str, Any] = {}
        self.event_info: dict[str, Any] = {}
        self.runtime_rgb: Any = None
        self.runtime_image_bounds: tuple[float, float, int, int] = (0.0, 0.0, 0, 0)
        self.backend_status = ""
        self._image_ref: Optional[tk.PhotoImage] = None
        self.status_var = tk.StringVar(value="")
        self.header_title_var = tk.StringVar(value="")
        self.header_subtitle_var = tk.StringVar(value="")
        self.metric_var = tk.StringVar(value="")
        self.card_window_id: Optional[int] = None

        self._setup_styles()
        self._build_layout()
        self.load_scene("crafter")

    def _setup_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Panel.TLabel", background=PANEL, foreground=INK)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Dark.Vertical.TScrollbar", background=PANEL_2, troughcolor=PANEL, bordercolor=PANEL)
        style.configure("Dark.Horizontal.TScrollbar", background=PANEL_2, troughcolor=PANEL, bordercolor=PANEL)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.header = tk.Frame(self.root, bg=BG, padx=18, pady=14)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(1, weight=1)

        self.scene_buttons: dict[str, tk.Button] = {}
        switcher = tk.Frame(self.header, bg=BG)
        switcher.grid(row=0, column=0, sticky="w")
        for index, scene in enumerate(("crafter", "highway")):
            button = tk.Button(
                switcher,
                text=SCENES[scene]["label"],
                command=lambda name=scene: self.load_scene(name),
                bd=0,
                padx=18,
                pady=8,
                cursor="hand2",
                font=("Microsoft YaHei UI", 11, "bold"),
            )
            button.grid(row=0, column=index, padx=(0, 8))
            self.scene_buttons[scene] = button

        title_box = tk.Frame(self.header, bg=BG)
        title_box.grid(row=0, column=1, sticky="ew", padx=16)
        tk.Label(
            title_box,
            textvariable=self.header_title_var,
            bg=BG,
            fg=INK,
            font=("Microsoft YaHei UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            textvariable=self.header_subtitle_var,
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(2, 0))

        tk.Button(
            self.header,
            text="刷新知识",
            command=lambda: self.load_scene(self.scene),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            bg="#24506f",
            activebackground="#2f668a",
            fg=INK,
            activeforeground=INK,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=2, sticky="e")

        body = tk.Frame(self.root, bg=BG, padx=18, pady=0)
        body.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=5)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)

        left_panel = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)

        tk.Label(
            left_panel,
            text="环境拓扑",
            bg=PANEL,
            fg=INK,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        topology_view = tk.Frame(left_panel, bg=PANEL)
        topology_view.grid(row=1, column=0, sticky="nsew")
        topology_view.columnconfigure(0, weight=1)
        topology_view.rowconfigure(0, weight=1)
        self.topology_canvas = tk.Canvas(topology_view, bg=CANVAS_BG, highlightthickness=0, height=500)
        self.topology_canvas.grid(row=0, column=0, sticky="nsew")
        topology_y = ttk.Scrollbar(
            topology_view,
            orient="vertical",
            command=self.topology_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        topology_y.grid(row=0, column=1, sticky="ns")
        topology_x = ttk.Scrollbar(
            topology_view,
            orient="horizontal",
            command=self.topology_canvas.xview,
            style="Dark.Horizontal.TScrollbar",
        )
        topology_x.grid(row=1, column=0, sticky="ew")
        self.topology_canvas.configure(xscrollcommand=topology_x.set, yscrollcommand=topology_y.set)

        runtime_panel = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        runtime_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        runtime_panel.columnconfigure(0, weight=1)
        runtime_panel.rowconfigure(1, weight=0)
        runtime_panel.rowconfigure(2, weight=1, minsize=170)
        tk.Label(
            runtime_panel,
            text="场景运行可视化",
            bg=PANEL,
            fg=INK,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.runtime_canvas = tk.Canvas(runtime_panel, bg=CANVAS_BG, highlightthickness=0, height=330)
        self.runtime_canvas.grid(row=1, column=0, sticky="ew")

        json_box = tk.Frame(runtime_panel, bg=PANEL, padx=0, pady=0)
        json_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        json_box.columnconfigure(0, weight=1)
        json_box.rowconfigure(1, weight=1)
        tk.Label(
            json_box,
            text="知识摘要",
            bg=PANEL,
            fg=INK,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.json_text = tk.Text(
            json_box,
            height=7,
            wrap="word",
            bg="#10283f",
            fg="#e7f1ff",
            insertbackground=INK,
            relief="flat",
            padx=10,
            pady=8,
            font=("Consolas", 10),
        )
        self.json_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        json_scroll = ttk.Scrollbar(
            json_box,
            orient="vertical",
            command=self.json_text.yview,
            style="Dark.Vertical.TScrollbar",
        )
        json_scroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        self.json_text.configure(yscrollcommand=json_scroll.set)

        right_panel = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        tk.Label(
            right_panel,
            text="技能策略与区域规则",
            bg=PANEL,
            fg=INK,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        card_view = tk.Frame(right_panel, bg=PANEL)
        card_view.grid(row=1, column=0, sticky="nsew")
        card_view.columnconfigure(0, weight=1)
        card_view.rowconfigure(0, weight=1)
        self.card_canvas = tk.Canvas(card_view, bg=PANEL, highlightthickness=0)
        self.card_canvas.grid(row=0, column=0, sticky="nsew")
        card_scroll = ttk.Scrollbar(
            card_view,
            orient="vertical",
            command=self.card_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        card_scroll.grid(row=0, column=1, sticky="ns")
        self.card_canvas.configure(yscrollcommand=card_scroll.set)
        self.card_frame = tk.Frame(self.card_canvas, bg=PANEL)
        self.card_window_id = self.card_canvas.create_window((0, 0), window=self.card_frame, anchor="nw")
        self.card_frame.bind("<Configure>", self._sync_card_scrollregion)
        self.card_canvas.bind("<Configure>", self._sync_card_width)

        footer = tk.Frame(self.root, bg=BG, padx=18, pady=0)
        footer.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        footer.columnconfigure(0, weight=1)
        tk.Label(
            footer,
            textvariable=self.metric_var,
            bg=BG,
            fg=MUTED,
            font=("Consolas", 11),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            textvariable=self.status_var,
            bg=BG,
            fg=GREEN,
            font=("Microsoft YaHei UI", 11),
        ).grid(row=0, column=1, sticky="e")

    def load_scene(self, scene: str) -> None:
        self.scene = scene
        scene_cfg = SCENES[scene]
        self.header_title_var.set(scene_cfg["title"])
        self.header_subtitle_var.set(scene_cfg["subtitle"])
        self._style_scene_buttons()

        env = None
        self.runtime_rgb = None
        self.runtime_image_bounds = (0.0, 0.0, 0, 0)
        self._image_ref = None
        env_config = dict(scene_cfg["config"])
        preferred_backend = str(env_config.get("backend", "scripted"))
        try:
            try:
                env = create_env(scene, env_config)
                self.obs = env.reset(seed=0)
                self.backend_status = f"backend={preferred_backend}"
            except Exception as exc:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                fallback_config = dict(env_config)
                fallback_config["backend"] = "scripted"
                env = create_env(scene, fallback_config)
                self.obs = env.reset(seed=0)
                self.backend_status = f"backend=scripted；{preferred_backend} 初始化失败：{exc}"
            self.domain = env.get_domain_knowledge()
            try:
                self.task_state = env.get_task_state()
                self.safety_state = env.get_safety_state()
                self.event_info = env.get_event_info()
            except Exception:
                self.task_state = {}
                self.safety_state = {}
                self.event_info = {}
            self.runtime_rgb = self._extract_runtime_rgb(env)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        self.topology = build_topology(scene, self.domain)
        self.skills = extract_skill_descriptors(scene, self.domain)
        self.rules = extract_region_rules(scene, self.domain)

        self._draw_topology()
        self._draw_runtime()
        self._draw_cards()
        self._draw_json()
        self.status_var.set(f"{scene_cfg['label']} 知识已加载 · {self.backend_status}")
        self.metric_var.set(
            f"topology={len(self.topology)} groups   skills={len(self.skills)}   rules={len(self.rules)}"
        )

    def _extract_runtime_rgb(self, env: Any) -> Any:
        image = self.obs.get("image")
        if image is not None and not isinstance(image, dict):
            return image
        render_rgb = getattr(env, "render_rgb", None)
        if callable(render_rgb):
            try:
                return render_rgb()
            except Exception:
                return None
        return None

    def _style_scene_buttons(self) -> None:
        for scene, button in self.scene_buttons.items():
            active = scene == self.scene
            accent = SCENES[scene]["accent"]
            button.configure(
                bg=accent if active else "#203e59",
                activebackground=accent,
                fg="#04101a" if active else MUTED,
                activeforeground="#04101a",
            )

    def _sync_card_scrollregion(self, _: Optional[tk.Event] = None) -> None:
        self.card_canvas.configure(scrollregion=self.card_canvas.bbox("all"))

    def _sync_card_width(self, event: tk.Event) -> None:
        if self.card_window_id is not None:
            self.card_canvas.itemconfigure(self.card_window_id, width=max(1, event.width))
        self._sync_card_scrollregion()

    def _draw_topology(self) -> None:
        self.topology_canvas.delete("all")
        width = max(self.topology_canvas.winfo_width(), 440)
        visible_height = max(self.topology_canvas.winfo_height(), 420)
        height = max(visible_height, 760 if self.scene == "highway" else 720)
        self._draw_background_grid(width, height)
        if self.scene == "crafter":
            self._draw_crafter_topology(width, height)
        else:
            self._draw_highway_topology(width, height)
        self.topology_canvas.configure(scrollregion=self.topology_canvas.bbox("all"))

    def _draw_runtime(self) -> None:
        self.runtime_canvas.delete("all")
        width = max(self.runtime_canvas.winfo_width(), 360)
        height = max(self.runtime_canvas.winfo_height(), 300)
        self._draw_runtime_background(width, height)
        if self._draw_runtime_rgb(width, height):
            self._draw_runtime_overlay(width, height)
            return
        if self.scene == "crafter":
            self._draw_crafter_runtime(width, height)
        else:
            self._draw_highway_runtime(width, height)

    def _draw_runtime_rgb(self, width: int, height: int) -> bool:
        image = self._normalize_rgb_image(self.runtime_rgb)
        if image is None:
            self.runtime_image_bounds = (0.0, 0.0, 0, 0)
            return False
        image_h = len(image)
        image_w = len(image[0])
        if image_h < 2 or image_w < 2:
            self.runtime_image_bounds = (0.0, 0.0, 0, 0)
            return False

        top = 74
        target_w = max(120, width - 40)
        target_h = max(120, height - top - 88)
        scale = min(target_w / image_w, target_h / image_h)
        display_w = max(1, int(image_w * scale))
        display_h = max(1, int(image_h * scale))

        display = tk.PhotoImage(width=display_w, height=display_h)
        display.put(
            " ".join(self._resized_rgb_rows(image, display_w, display_h)),
            to=(0, 0, display_w, display_h),
        )
        self._image_ref = display
        x = max(20, (width - display_w) / 2)
        y = top
        self.runtime_canvas.create_image(x, y, image=display, anchor="nw")
        self.runtime_canvas.create_rectangle(
            x - 1,
            y - 1,
            x + display_w + 1,
            y + display_h + 1,
            outline="#d8e8f8",
        )
        self.runtime_image_bounds = (x, y, display_w, display_h)
        return True

    def _resized_rgb_rows(self, image: list[Any], display_w: int, display_h: int) -> list[str]:
        source_h = len(image)
        source_w = len(image[0])
        rows = []
        for y in range(display_h):
            source_y = min(source_h - 1, int(y * source_h / display_h))
            source_row = image[source_y]
            colors = []
            for x in range(display_w):
                source_x = min(source_w - 1, int(x * source_w / display_w))
                pixel = source_row[source_x]
                if not isinstance(pixel, (list, tuple)) or len(pixel) < 3:
                    colors.append("#000000")
                    continue
                r, g, b = [max(0, min(255, int(value))) for value in pixel[:3]]
                colors.append(f"#{r:02x}{g:02x}{b:02x}")
            rows.append("{" + " ".join(colors) + "}")
        return rows

    def _normalize_rgb_image(self, image: Any) -> Optional[list[Any]]:
        if image is None or isinstance(image, dict):
            return None
        if hasattr(image, "tolist"):
            image = image.tolist()
        if not isinstance(image, list) or not image or not isinstance(image[0], list):
            return None
        first_row = image[0]
        if not first_row or not isinstance(first_row[0], (list, tuple)):
            return None
        return image

    def _draw_runtime_overlay(self, width: int, height: int) -> None:
        if self.scene == "crafter":
            self._runtime_title("Crafter 运行快照", "官方 RGB 观测 / 背包 / 安全状态", GREEN)
            self._draw_crafter_overlay_stats(width, height)
        else:
            self._runtime_title("Highway 运行快照", "官方 RGB 渲染 / 车道 / 风险状态", CYAN)
            self._draw_highway_overlay_stats(width, height)

    def _draw_crafter_overlay_stats(self, width: int, height: int) -> None:
        inventory = dict(self.obs.get("inventory") or {})
        player = dict(self.obs.get("player_state") or {})
        phase = str(self.task_state.get("phase", "collect_wood"))
        safety = self.safety_state
        _, image_y, _, image_h = self.runtime_image_bounds
        stat_y = min(height - 58, image_y + image_h + 12)
        xs, stat_w = self._runtime_stat_columns(width, 3)
        self._runtime_stat(xs[0], stat_y, "phase", phase, GREEN, stat_w)
        self._runtime_stat(xs[1], stat_y, "health", str(player.get("health", safety.get("health", "-"))), CYAN, stat_w)
        self._runtime_stat(xs[2], stat_y, "danger", f"{float(safety.get('danger_level', 0.0) or 0.0):.2f}", RED, stat_w)
        inv_line = "  ".join(
            f"{name}:{inventory.get(name, 0)}"
            for name in ("wood", "stone", "table", "wood_pickaxe", "stone_pickaxe")
        )
        self.runtime_canvas.create_text(
            24,
            height - 20,
            anchor="w",
            fill="#d8e8f8",
            font=("Consolas", 9, "bold"),
            text=inv_line[: max(20, int(width / 7))],
        )

    def _draw_highway_overlay_stats(self, width: int, height: int) -> None:
        task = self.task_state
        safety = self.safety_state
        _, image_y, _, image_h = self.runtime_image_bounds
        stat_y = min(height - 58, image_y + image_h + 12)
        xs, stat_w = self._runtime_stat_columns(width, 3)
        self._runtime_stat(xs[0], stat_y, "lane", str(task.get("lane_id", 0)), CYAN, stat_w)
        self._runtime_stat(xs[1], stat_y, "speed", f"{float(task.get('speed', 0.0) or 0.0):.1f}", GREEN, stat_w)
        self._runtime_stat(xs[2], stat_y, "risk", f"{float(safety.get('risk_score', 0.0) or 0.0):.2f}", RED, stat_w)
        self.runtime_canvas.create_text(
            24,
            height - 20,
            anchor="w",
            fill=MUTED,
            font=("Microsoft YaHei UI", 8),
            text=(
                f"front={safety.get('front_vehicle_dist')}  "
                f"left_safe={safety.get('left_lane_safe')}  "
                f"right_safe={safety.get('right_lane_safe')}  "
                f"recommended={safety.get('recommended_safe_action')}"
            )[: max(24, int(width / 6))],
        )

    def _draw_runtime_background(self, width: int, height: int) -> None:
        for y in range(0, height, 30):
            fill = CANVAS_BG if (y // 30) % 2 == 0 else CANVAS_ALT
            self.runtime_canvas.create_rectangle(0, y, width, y + 30, fill=fill, outline="")
        for x in range(0, width, 36):
            self.runtime_canvas.create_line(x, 0, x, height, fill=GRID_LINE)
        for y in range(0, height, 36):
            self.runtime_canvas.create_line(0, y, width, y, fill=GRID_LINE)

    def _draw_crafter_runtime(self, width: int, height: int) -> None:
        inventory = dict(self.obs.get("inventory") or {})
        player = dict(self.obs.get("player_state") or {})
        phase = str(self.task_state.get("phase", "collect_wood"))
        safety = self.safety_state

        self._runtime_title("Crafter 运行快照", "局部地图 / 背包 / 安全状态", GREEN)
        grid_size = min(44, max(22, int((min(width - 48, height - 190)) / 7)))
        x0 = width / 2 - grid_size * 3.5
        y0 = 86
        palette = {
            "grass": "#284b32",
            "tree": "#2fb06d",
            "stone": "#798897",
            "table": "#b98d55",
            "zombie": "#d64f6f",
            "player": CYAN,
            "sand": "#7c7150",
        }
        objects = {
            (1, 1): "tree",
            (5, 1): "stone",
            (2, 5): "table",
            (5, 5): "zombie",
            (3, 3): "player",
            (0, 4): "tree",
            (6, 2): "stone",
        }
        labels = {
            "tree": "T",
            "stone": "S",
            "table": "W",
            "zombie": "Z",
            "player": "P",
        }
        for row in range(7):
            for col in range(7):
                kind = objects.get((col, row), "grass" if (row + col) % 5 else "sand")
                x = x0 + col * grid_size
                y = y0 + row * grid_size
                self.runtime_canvas.create_rectangle(
                    x,
                    y,
                    x + grid_size - 2,
                    y + grid_size - 2,
                    fill=palette[kind],
                    outline="#2d5a78",
                    width=1,
                )
                if kind in labels:
                    self.runtime_canvas.create_text(
                        x + grid_size / 2,
                        y + grid_size / 2,
                        fill=INK,
                        font=("Consolas", 13, "bold"),
                        text=labels[kind],
                    )

        stat_y = min(height - 78, y0 + grid_size * 7 + 18)
        xs, stat_w = self._runtime_stat_columns(width, 3)
        self._runtime_stat(xs[0], stat_y, "phase", phase, GREEN, stat_w)
        self._runtime_stat(xs[1], stat_y, "health", str(player.get("health", safety.get("health", "-"))), CYAN, stat_w)
        self._runtime_stat(xs[2], stat_y, "danger", f"{float(safety.get('danger_level', 0.0) or 0.0):.2f}", RED, stat_w)
        inv_line = "  ".join(
            f"{name}:{inventory.get(name, 0)}"
            for name in ("wood", "stone", "table", "wood_pickaxe", "stone_pickaxe")
        )
        self.runtime_canvas.create_text(24, height - 42, anchor="w", fill="#e7f1ff", font=("Consolas", 10, "bold"), text=inv_line)
        self.runtime_canvas.create_text(
            24,
            height - 20,
            anchor="w",
            fill=MUTED,
            font=("Microsoft YaHei UI", 9),
            text="T=tree  S=stone  W=workbench  Z=danger  P=player",
        )

    def _draw_highway_runtime(self, width: int, height: int) -> None:
        task = self.task_state
        safety = self.safety_state
        nearby = list(self.obs.get("nearby_vehicles", []))
        lanes_count = int(task.get("lanes_count", 4) or 4)
        lane_id = int(task.get("lane_id", 0) or 0)
        lane_h = min(52, max(34, int((height - 206) / max(lanes_count, 1))))
        road_x0 = 28
        road_x1 = width - 28
        road_y0 = 86
        ego_x = width * 0.36

        self._runtime_title("Highway 运行快照", "车道 / 周车 / 风险状态", CYAN)
        for lane in range(lanes_count):
            y0 = road_y0 + lane * lane_h
            fill = "#25445f" if lane % 2 == 0 else "#203b55"
            self.runtime_canvas.create_rectangle(road_x0, y0, road_x1, y0 + lane_h, fill=fill, outline="#3a6381")
            if lane > 0:
                self.runtime_canvas.create_line(road_x0, y0, road_x1, y0, fill="#d7edf6", dash=(12, 10), width=2)
            self.runtime_canvas.create_text(road_x0 - 12, y0 + lane_h / 2, fill=INK, font=("Consolas", 10, "bold"), text=str(lane))

        ego_y = road_y0 + lane_id * lane_h + lane_h / 2
        self._runtime_vehicle(ego_x, ego_y, CYAN, "ego")
        for vehicle in nearby[:10]:
            lane = vehicle.get("lane_id")
            if lane is None:
                continue
            try:
                distance = float(vehicle.get("distance", 0.0) or 0.0)
                v_lane = int(lane)
            except (TypeError, ValueError):
                continue
            x = ego_x + distance * 2.2
            if x < road_x0 + 22 or x > road_x1 - 22 or v_lane < 0 or v_lane >= lanes_count:
                continue
            y = road_y0 + v_lane * lane_h + lane_h / 2
            self._runtime_vehicle(x, y, AMBER, "veh")

        stat_y = road_y0 + lanes_count * lane_h + 18
        row2_y = stat_y + 32
        xs, stat_w = self._runtime_stat_columns(width, 3)
        self._runtime_stat(xs[0], stat_y, "lane", str(lane_id), CYAN, stat_w)
        self._runtime_stat(xs[1], stat_y, "speed", f"{float(task.get('speed', 0.0) or 0.0):.1f}", GREEN, stat_w)
        self._runtime_stat(xs[2], stat_y, "risk", f"{float(safety.get('risk_score', 0.0) or 0.0):.2f}", RED, stat_w)
        self._runtime_stat(xs[0], row2_y, "front", str(safety.get("front_vehicle_dist")), AMBER, stat_w)
        self._runtime_stat(xs[1], row2_y, "left_safe", str(safety.get("left_lane_safe")), GREEN, stat_w)
        self._runtime_stat(xs[2], row2_y, "right_safe", str(safety.get("right_lane_safe")), GREEN, stat_w)
        self.runtime_canvas.create_text(
            24,
            height - 24,
            anchor="w",
            fill=MUTED,
            font=("Microsoft YaHei UI", 9),
            text=f"recommended={safety.get('recommended_safe_action')}  event={self.event_info.get('event_type')}",
        )

    def _runtime_title(self, title: str, subtitle: str, accent: str) -> None:
        self.runtime_canvas.create_text(24, 28, anchor="w", fill=accent, font=("Microsoft YaHei UI", 18, "bold"), text=title)
        self.runtime_canvas.create_text(24, 56, anchor="w", fill=MUTED, font=("Microsoft YaHei UI", 11), text=subtitle)

    def _runtime_stat_columns(self, width: int, count: int) -> tuple[list[float], int]:
        gap = 12
        box_w = min(142, max(96, int((width - 48 - gap * (count - 1)) / max(count, 1))))
        return [24 + index * (box_w + gap) for index in range(count)], box_w

    def _runtime_stat(self, x: float, y: float, label: str, value: str, accent: str, box_w: int = 142) -> None:
        self.runtime_canvas.create_rectangle(x, y, x + box_w, y + 25, fill="#21415f", outline="#3a6381")
        self.runtime_canvas.create_text(x + 9, y + 13, anchor="w", fill=MUTED, font=("Consolas", 9), text=label)
        self.runtime_canvas.create_text(x + box_w - 10, y + 13, anchor="e", fill=accent, font=("Consolas", 10, "bold"), text=value[:16])

    def _runtime_vehicle(self, x: float, y: float, color: str, label: str) -> None:
        self.runtime_canvas.create_rectangle(x - 20, y - 10, x + 20, y + 10, fill=color, outline=INK, width=1)
        self.runtime_canvas.create_rectangle(x - 12, y - 15, x + 11, y - 7, fill=color, outline=INK, width=1)
        self.runtime_canvas.create_text(x, y + 23, fill=INK, font=("Consolas", 8, "bold"), text=label)

    def _draw_background_grid(self, width: int, height: int) -> None:
        for y in range(0, height, 28):
            shade = CANVAS_BG if (y // 28) % 2 == 0 else CANVAS_ALT
            self.topology_canvas.create_rectangle(0, y, width, y + 28, fill=shade, outline="")
        for x in range(0, width, 36):
            self.topology_canvas.create_line(x, 0, x, height, fill=GRID_LINE)
        for y in range(0, height, 36):
            self.topology_canvas.create_line(0, y, width, y, fill=GRID_LINE)

    def _draw_crafter_topology(self, width: int, height: int) -> None:
        regions = list(self.topology.get("regions") or [])
        objects = list(self.topology.get("objects") or [])
        rules = self.rules
        skills = self.skills
        center = (width * 0.50, height * 0.46)
        radius = min(width, height) * 0.30
        region_positions: dict[str, tuple[float, float]] = {}

        self._draw_glow_text(34, 34, "资源-规则-技能知识网络", GREEN)
        self.topology_canvas.create_text(
            34,
            64,
            anchor="w",
            fill=MUTED,
            font=("Microsoft YaHei UI", 11),
            text="区域节点连接规则，技能策略围绕任务链展开",
        )

        for index, region in enumerate(regions):
            angle = -math.pi / 2 + index * 2 * math.pi / max(len(regions), 1)
            x = center[0] + math.cos(angle) * radius
            y = center[1] + math.sin(angle) * radius
            region_positions[region] = (x, y)
            self._line(center[0], center[1], x, y, "#315a78", width=2)
            self._node(x, y, region, GREEN, 94, 42)
            rule_text = " / ".join(str(item) for item in rules.get(region, []))
            if rule_text:
                self.topology_canvas.create_text(
                    x,
                    y + 38,
                    fill="#c9f6dd",
                    font=("Microsoft YaHei UI", 9),
                    text=rule_text[:34],
                )

        self._node(center[0], center[1], "stone_pickaxe\n任务目标", AMBER, 120, 58)

        object_y = height - 72
        start_x = max(70, width / 2 - len(objects) * 54)
        for index, obj in enumerate(objects):
            x = start_x + index * 112
            self._pill(x, object_y, obj, PURPLE)
            nearest_region = regions[index % len(regions)] if regions else None
            if nearest_region:
                rx, ry = region_positions[nearest_region]
                self._line(rx, ry + 20, x, object_y - 18, "#334e68", width=1)

        chain = [
            "collect_wood",
            "place_table",
            "make_wood_pickaxe",
            "collect_stone",
            "make_stone_pickaxe",
        ]
        chain_x = 58
        for index, skill_id in enumerate(chain):
            y = 116 + index * 58
            self._skill_chip(chain_x, y, index + 1, skill_id, skills.get(skill_id, ""))
            if index > 0:
                self._arrow(chain_x + 20, y - 36, chain_x + 20, y - 18, CYAN)

    def _draw_highway_topology(self, width: int, height: int) -> None:
        lanes = list(self.topology.get("lanes") or [0, 1, 2, 3])
        adjacent = self.topology.get("adjacent_lanes") or {}
        lane_h = min(72, max(50, (height - 360) // max(len(lanes), 1)))
        road_x0 = 72
        road_x1 = width - 72
        road_y0 = 116

        self._draw_glow_text(34, 34, "车道拓扑与分层动作空间", CYAN)
        self.topology_canvas.create_text(
            34,
            64,
            anchor="w",
            fill=MUTED,
            font=("Microsoft YaHei UI", 11),
            text="车道邻接关系决定换道动作，安全规则约束高层技能选择",
        )

        for index, lane in enumerate(lanes):
            y0 = road_y0 + index * lane_h
            fill = "#25445f" if index % 2 == 0 else "#203b55"
            self.topology_canvas.create_rectangle(road_x0, y0, road_x1, y0 + lane_h, fill=fill, outline="#3a6381")
            if index > 0:
                self.topology_canvas.create_line(road_x0, y0, road_x1, y0, fill="#d7edf6", dash=(14, 10), width=2)
            self.topology_canvas.create_text(
                road_x0 - 28,
                y0 + lane_h / 2,
                fill=INK,
                font=("Consolas", 15, "bold"),
                text=f"L{lane}",
            )
            links = adjacent.get(lane, adjacent.get(str(lane), []))
            self._pill(road_x1 - 86, y0 + lane_h / 2, f"adj {links}", BLUE)

        ego_lane_index = min(1, len(lanes) - 1)
        ego_y = road_y0 + ego_lane_index * lane_h + lane_h / 2
        self._vehicle(road_x0 + 190, ego_y, CYAN, "ego")
        self._vehicle(road_x0 + 370, ego_y, AMBER, "front")
        if ego_lane_index > 0:
            self._arrow(road_x0 + 190, ego_y - 22, road_x0 + 190, ego_y - lane_h + 26, GREEN)
            self.topology_canvas.create_text(
                road_x0 + 220,
                ego_y - lane_h / 2,
                anchor="w",
                fill="#c9f6dd",
                font=("Microsoft YaHei UI", 11, "bold"),
                text="lane_left if safe",
            )
        if ego_lane_index < len(lanes) - 1:
            self._arrow(road_x0 + 190, ego_y + 22, road_x0 + 190, ego_y + lane_h - 26, GREEN)
            self.topology_canvas.create_text(
                road_x0 + 220,
                ego_y + lane_h / 2,
                anchor="w",
                fill="#c9f6dd",
                font=("Microsoft YaHei UI", 11, "bold"),
                text="lane_right if safe",
            )

        rule_y = road_y0 + len(lanes) * lane_h + 52
        for index, (name, values) in enumerate(self.rules.items()):
            self._rule_badge(92, rule_y + index * 78, name, values, width - 184)

    def _draw_cards(self) -> None:
        for child in self.card_frame.winfo_children():
            child.destroy()

        self._section_title("技能策略", f"{len(self.skills)} skills", SCENES[self.scene]["accent"])
        for index, (skill_id, desc) in enumerate(self.skills.items(), start=1):
            self._card(
                title=f"{index:02d}  {skill_id}",
                body=desc,
                accent=SCENES[self.scene]["accent"],
            )

        self._section_title("区域规则", f"{len(self.rules)} rule groups", AMBER)
        for key, values in self.rules.items():
            self._card(
                title=str(key),
                body=" / ".join(str(item) for item in self._as_list(values)),
                accent=AMBER,
            )

        craft_rules = self.domain.get("craft_rules")
        if isinstance(craft_rules, dict) and craft_rules:
            self._section_title("制作规则", f"{len(craft_rules)} craft rules", PURPLE)
            for key, values in craft_rules.items():
                self._card(
                    title=str(key),
                    body=" / ".join(str(item) for item in self._as_list(values)),
                    accent=PURPLE,
                )
        self._sync_card_scrollregion()
        self.card_canvas.yview_moveto(0)

    def _section_title(self, title: str, meta: str, accent: str) -> None:
        row = tk.Frame(self.card_frame, bg=PANEL, padx=14, pady=0)
        row.pack(fill="x", pady=(6, 2))
        tk.Label(
            row,
            text=title,
            bg=PANEL,
            fg=INK,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(side="left")
        tk.Label(
            row,
            text=meta,
            bg=PANEL,
            fg=accent,
            font=("Consolas", 10, "bold"),
        ).pack(side="right")

    def _card(self, title: str, body: str, accent: str) -> None:
        outer = tk.Frame(self.card_frame, bg=PANEL, padx=6, pady=1)
        outer.pack(fill="x")
        card = tk.Frame(outer, bg=PANEL_2, padx=9, pady=6, highlightthickness=1, highlightbackground="#3a6381")
        card.pack(fill="x")
        top = tk.Frame(card, bg=PANEL_2)
        top.pack(fill="x")
        tk.Canvas(top, width=8, height=8, bg=PANEL_2, highlightthickness=0).pack(side="left", padx=(0, 6))
        dot = top.winfo_children()[0]
        dot.create_oval(1, 1, 7, 7, fill=accent, outline="")
        tk.Label(
            top,
            text=title,
            bg=PANEL_2,
            fg=INK,
            font=("Consolas", 10, "bold"),
        ).pack(side="left", anchor="w")
        tk.Label(
            card,
            text=str(body),
            bg=PANEL_2,
            fg="#c7d8ea",
            justify="left",
            wraplength=max(180, self.card_canvas.winfo_width() - 54),
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(2, 0), fill="x")

    def _draw_json(self) -> None:
        payload = {
            "scene": self.scene,
            "backend": self.backend_status,
            "topology_keys": list(self.topology.keys()),
            "skills": list(self.skills.keys()),
            "rules": list(self.rules.keys()),
            "task_state": self.task_state,
            "safety_state": self.safety_state,
        }
        if "craft_rules" in self.domain:
            payload["craft_rules"] = list(dict(self.domain["craft_rules"]).keys())
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    def _node(self, x: float, y: float, text: str, accent: str, w: int, h: int) -> None:
        self.topology_canvas.create_oval(x - w / 2 - 9, y - h / 2 - 9, x + w / 2 + 9, y + h / 2 + 9, fill="", outline=accent, width=1)
        self.topology_canvas.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill="#21415f", outline=accent, width=2)
        self.topology_canvas.create_text(x, y, fill=INK, font=("Microsoft YaHei UI", 11, "bold"), text=text)

    def _pill(self, x: float, y: float, text: str, accent: str) -> None:
        w = max(72, len(str(text)) * 8 + 24)
        self.topology_canvas.create_rectangle(x - w / 2, y - 16, x + w / 2, y + 16, fill="#244763", outline=accent, width=1)
        self.topology_canvas.create_text(x, y, fill=INK, font=("Consolas", 10, "bold"), text=text)

    def _skill_chip(self, x: float, y: float, index: int, skill_id: str, desc: str) -> None:
        self.topology_canvas.create_oval(x, y - 18, x + 36, y + 18, fill="#21415f", outline=CYAN, width=2)
        self.topology_canvas.create_text(x + 18, y, fill=CYAN, font=("Consolas", 12, "bold"), text=str(index))
        self.topology_canvas.create_text(x + 50, y - 7, anchor="w", fill=INK, font=("Consolas", 11, "bold"), text=skill_id)
        self.topology_canvas.create_text(x + 50, y + 14, anchor="w", fill=MUTED, font=("Microsoft YaHei UI", 9), text=desc[:34])

    def _rule_badge(self, x: float, y: float, name: str, values: Any, badge_w: float = 185) -> None:
        text = " / ".join(str(item) for item in self._as_list(values))
        badge_w = max(185, badge_w)
        self.topology_canvas.create_rectangle(x, y, x + badge_w, y + 64, fill="#244763", outline=AMBER, width=1)
        self.topology_canvas.create_text(x + 12, y + 16, anchor="w", fill=AMBER, font=("Consolas", 10, "bold"), text=name[:22])
        self.topology_canvas.create_text(
            x + 12,
            y + 42,
            anchor="w",
            fill=INK,
            font=("Microsoft YaHei UI", 9),
            text=text,
            width=badge_w - 24,
        )

    def _vehicle(self, x: float, y: float, color: str, label: str) -> None:
        self.topology_canvas.create_rectangle(x - 28, y - 14, x + 28, y + 14, fill=color, outline=INK, width=1)
        self.topology_canvas.create_rectangle(x - 16, y - 20, x + 14, y - 8, fill=color, outline=INK, width=1)
        self.topology_canvas.create_text(x, y + 30, fill=INK, font=("Consolas", 10, "bold"), text=label)

    def _line(self, x0: float, y0: float, x1: float, y1: float, color: str, width: int = 1) -> None:
        self.topology_canvas.create_line(x0, y0, x1, y1, fill=color, width=width)

    def _arrow(self, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
        self.topology_canvas.create_line(x0, y0, x1, y1, fill=color, width=3, arrow=tk.LAST, arrowshape=(12, 14, 5))

    def _draw_glow_text(self, x: float, y: float, text: str, accent: str) -> None:
        self.topology_canvas.create_text(x + 1, y + 1, anchor="w", fill="#18344e", font=("Microsoft YaHei UI", 20, "bold"), text=text)
        self.topology_canvas.create_text(x, y, anchor="w", fill=accent, font=("Microsoft YaHei UI", 20, "bold"), text=text)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return []
        return [value]


def main() -> None:
    root = tk.Tk()
    app = DomainKnowledgeVisualizer(root)

    def redraw(_: Optional[tk.Event] = None) -> None:
        app._draw_topology()
        app._draw_runtime()

    app.topology_canvas.bind("<Configure>", redraw)
    app.runtime_canvas.bind("<Configure>", redraw)
    root.mainloop()


if __name__ == "__main__":
    main()
