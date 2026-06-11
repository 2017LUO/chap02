"""Crafter 环境可视化。

运行方式：
    conda run -n DT python tests/crafter/manual_control.py
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import CrafterEnv  # noqa: E402
from src.chap02_dk_abs.domain_knowledge import (  # noqa: E402
    build_topology,
    extract_region_rules,
    extract_skill_descriptors,
)
from src.chap02_dk_abs.planning import generate_llm_action_behavior_plan, validate_plan  # noqa: E402


TARGET_ACHIEVEMENTS = [
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
]


class CrafterManualControl:
    """用按钮手动验证 Crafter 封装的动作、技能和状态字段。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Crafter 石镐任务手动控制")
        self.env: Optional[CrafterEnv] = None
        self.backend_var = tk.StringVar(value="gym")
        self.backend_status_var = tk.StringVar(value="")
        self.seed_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="")
        self.auto_button_var = tk.StringVar(value="auto DK-ABS")
        self._image_ref: Optional[tk.PhotoImage] = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.step_count = 0
        self.llm_request_count = 0
        self.last_action_name = "reset"
        self.last_unlocked: list[str] = []
        self.last_llm_plan = ""
        self.auto_running = False
        self.done = False
        self.obs: dict[str, Any] = {}

        self._build_layout()
        self.reset_env()

    def _build_layout(self) -> None:
        """构建手动控制窗口。"""
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=680, height=440, bg="#16231b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.detail_text = tk.Text(left, height=16, wrap="word")
        self.detail_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="ns")

        seed_box = ttk.LabelFrame(right, text="重置")
        seed_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(seed_box, text="backend").grid(row=0, column=0, padx=4, pady=4)
        ttk.Combobox(seed_box, textvariable=self.backend_var, values=["gym", "scripted"], width=9, state="readonly").grid(
            row=0, column=1, padx=4, pady=4
        )
        ttk.Label(seed_box, text="seed").grid(row=1, column=0, padx=4, pady=4)
        ttk.Entry(seed_box, textvariable=self.seed_var, width=8).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(seed_box, text="reset", command=self.reset_env).grid(row=1, column=2, padx=4, pady=4)
        ttk.Label(seed_box, textvariable=self.backend_status_var, justify="left", wraplength=250).grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 4)
        )

        action_box = ttk.LabelFrame(right, text="原子动作")
        action_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        action_buttons = [
            ("noop", "noop"),
            ("←", "move_left"),
            ("→", "move_right"),
            ("↑", "move_up"),
            ("↓", "move_down"),
            ("do/interact", "interact"),
            ("sleep", "sleep"),
            ("place_stone", "place_stone"),
            ("place_table", "place_table"),
            ("place_furnace", "place_furnace"),
            ("place_plant", "place_plant"),
            ("make_wood_pickaxe", "make_wood_pickaxe"),
            ("make_stone_pickaxe", "make_stone_pickaxe"),
        ]
        for index, (label, action_name) in enumerate(action_buttons):
            ttk.Button(action_box, text=label, command=lambda name=action_name: self.run_action(name)).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=4, pady=3
            )

        repeat_box = ttk.LabelFrame(right, text="连续移动")
        repeat_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        repeat_buttons = [
            ("← x5", "move_left"),
            ("→ x5", "move_right"),
            ("↑ x5", "move_up"),
            ("↓ x5", "move_down"),
        ]
        for index, (label, action_name) in enumerate(repeat_buttons):
            ttk.Button(repeat_box, text=label, command=lambda name=action_name: self.run_repeat(name, 5)).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=4, pady=3
            )

        skill_box = ttk.LabelFrame(right, text="技能策略")
        skill_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        skill_buttons = [
            ("explore", "explore", {"max_steps": 4}),
            ("collect_wood", "collect_wood", {"target_count": 4, "max_steps": 120}),
            ("place_table", "place_table", {}),
            ("make_wood_pickaxe", "make_wood_pickaxe", {}),
            ("collect_stone", "collect_stone", {"target_count": 1, "max_steps": 120}),
            ("make_stone_pickaxe", "make_stone_pickaxe", {}),
            ("avoid_zombie", "avoid_zombie", {"max_steps": 4}),
        ]
        for index, (label, skill_id, skill_args) in enumerate(skill_buttons):
            ttk.Button(skill_box, text=label, command=lambda sid=skill_id, args=skill_args: self.run_skill(sid, args)).grid(
                row=index, column=0, sticky="ew", padx=4, pady=3
            )

        auto_box = ttk.LabelFrame(right, text="自动演示")
        auto_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(auto_box, textvariable=self.auto_button_var, command=self.toggle_auto_run).grid(
            row=0, column=0, sticky="ew", padx=4, pady=3
        )

        ttk.Label(right, textvariable=self.status_var, justify="left", wraplength=260).grid(
            row=5, column=0, sticky="ew", pady=(4, 0)
        )
        self.root.bind("<Left>", lambda _: self.run_action("move_left"))
        self.root.bind("<Right>", lambda _: self.run_action("move_right"))
        self.root.bind("<Up>", lambda _: self.run_action("move_up"))
        self.root.bind("<Down>", lambda _: self.run_action("move_down"))
        self.root.bind("<space>", lambda _: self.run_action("interact"))

    def reset_env(self) -> None:
        """按输入随机种子重置环境。"""
        seed = int(self.seed_var.get() or 0)
        backend = self.backend_var.get()
        self.auto_running = False
        self.auto_button_var.set("auto DK-ABS")
        try:
            self.env = CrafterEnv({"backend": backend, "env_id": "CrafterReward-v1"})
            self.backend_status_var.set(f"backend: {backend}")
        except Exception as exc:
            # DT 当前若 PIL/Crafter DLL 有问题，会在这里明确提示并回退到脚本后端。
            self.env = CrafterEnv({"backend": "scripted"})
            self.backend_status_var.set(f"Gym 初始化失败，已回退 scripted:\n{exc}")
        self.obs = self.env.reset(seed=seed)
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.step_count = 0
        self.llm_request_count = 0
        self.last_action_name = "reset"
        self.last_unlocked = []
        self.last_llm_plan = ""
        self.done = False
        self._refresh()

    def run_action(self, action_name: str) -> None:
        """执行一个原子动作并刷新可视化状态。"""
        if self.done or self.env is None:
            return
        try:
            before = self._completed_achievement_ids()
            self.obs, self.last_reward, self.done, _ = self.env.step(action_name)
            self.total_reward += float(self.last_reward)
            self.last_unlocked = self._newly_unlocked(before)
            self.step_count += 1
            self.last_action_name = action_name
        except Exception as exc:
            self.status_var.set(f"action failed: {action_name}\n{exc}")
            return
        self._refresh()

    def run_repeat(self, action_name: str, count: int) -> None:
        """连续执行同一动作，避免单步被障碍挡住时看不出变化。"""
        for _ in range(count):
            if self.done:
                break
            self.run_action(action_name)

    def run_skill(self, skill_id: str, skill_args: dict[str, Any]) -> None:
        """执行一个技能策略并刷新可视化状态。"""
        if self.done or self.env is None:
            return
        try:
            before = self._completed_achievement_ids()
            self.obs, self.last_reward, self.done, info = self.env.apply_skill(skill_id, skill_args)
            self.total_reward += float(self.last_reward)
            self.last_unlocked = self._newly_unlocked(before)
            self.step_count += int(info.get("skill_steps", 1) or 1)
            self.last_action_name = skill_id
        except Exception as exc:
            self.status_var.set(f"skill failed: {skill_id}\n{exc}")
            return
        self._refresh()

    def toggle_auto_run(self) -> None:
        """Start or stop automatic DK-ABS high-level execution."""
        if self.env is None:
            return
        if self.auto_running:
            self.auto_running = False
            self.auto_button_var.set("auto DK-ABS")
            self._refresh()
            return
        if self.done:
            self.reset_env()
        self.auto_running = True
        self.auto_button_var.set("stop auto")
        self.root.after(100, self._auto_tick)

    def _auto_tick(self) -> None:
        """Ask the simulated LLM for the next behavior, then execute it."""
        if not self.auto_running or self.env is None:
            return
        task_state = self.env.get_task_state()
        if self.done or task_state.get("phase") == "task_complete":
            self.auto_running = False
            self.auto_button_var.set("auto DK-ABS")
            self._refresh()
            return
        try:
            domain = self._current_domain_knowledge()
            planner_config = self._auto_planner_config()
            llm_result = generate_llm_action_behavior_plan("crafter", task_state, domain, planner_config)
            plan = llm_result.plan
            ok, errors = validate_plan("crafter", plan, max_plan_length=20)
            if not ok or not plan:
                self.status_var.set(f"auto plan invalid: {errors}")
                self.auto_running = False
                self.auto_button_var.set("auto DK-ABS")
                return
            behavior = plan[0]
            self.llm_request_count += 1
            self.last_llm_plan = " -> ".join(item.skill_id for item in plan[:5])
            self.run_skill(behavior.skill_id, behavior.to_skill_args())
        except Exception as exc:
            self.status_var.set(f"auto failed:\n{exc}")
            self.auto_running = False
            self.auto_button_var.set("auto DK-ABS")
            return
        if self.auto_running:
            self.root.after(900, self._auto_tick)

    def _current_domain_knowledge(self) -> dict[str, Any]:
        """Build the same domain knowledge block used by chapter-2 experiments."""
        if self.env is None:
            return {}
        domain = self.env.get_domain_knowledge()
        return {
            **domain,
            "environment_topology": build_topology("crafter", domain),
            "region_rules": extract_region_rules("crafter", domain),
            "skill_policies": extract_skill_descriptors("crafter", domain),
        }

    @staticmethod
    def _auto_planner_config() -> dict[str, Any]:
        """Planner config for automatic visual demonstration."""
        return {
            "mode": "llm_simulated",
            "fallback_to_rule": True,
            "replan_each_step": True,
            "wood_target_count": 4,
            "stone_target_count": 1,
            "max_skill_steps": 350,
            "llm": {
                "provider": "simulated",
                "model": "gpt-4o-mini-sim",
                "temperature": 0.0,
                "response_format": "json_object",
            },
        }

    def _completed_achievement_ids(self) -> list[str]:
        """Return currently completed target and official achievements."""
        task_state = self.obs.get("task_state", {})
        inventory = task_state.get("inventory", {}) or self.obs.get("inventory", {})
        achievements = task_state.get("achievements", {}) or self.obs.get("achievements", {})
        completed = set()
        for name in TARGET_ACHIEVEMENTS:
            if achievements.get(name):
                completed.add(name)
        if int(inventory.get("wood", 0) or 0) > 0:
            completed.add("collect_wood")
        if int(inventory.get("table", 0) or 0) > 0:
            completed.add("place_table")
        if int(inventory.get("wood_pickaxe", 0) or 0) > 0:
            completed.add("make_wood_pickaxe")
        if int(inventory.get("stone", 0) or 0) > 0:
            completed.add("collect_stone")
        if int(inventory.get("stone_pickaxe", 0) or 0) > 0:
            completed.add("make_stone_pickaxe")
        for name, done in achievements.items():
            if done:
                completed.add(str(name))
        return sorted(completed)

    def _newly_unlocked(self, before: list[str]) -> list[str]:
        """Return achievements that became completed after the latest action."""
        before_set = set(before)
        return [name for name in self._completed_achievement_ids() if name not in before_set]

    def _achievement_progress_text(self) -> tuple[str, str, str]:
        """Build compact achievement labels for the status panel and canvas."""
        completed = self._completed_achievement_ids()
        completed_set = set(completed)
        target_done = [name for name in TARGET_ACHIEVEMENTS if name in completed_set]
        extra_done = [name for name in completed if name not in TARGET_ACHIEVEMENTS]
        target_text = ", ".join(target_done) if target_done else "none"
        extra_text = ", ".join(extra_done[:4]) if extra_done else "none"
        if len(extra_done) > 4:
            extra_text += f", +{len(extra_done) - 4}"
        return f"{len(target_done)}/{len(TARGET_ACHIEVEMENTS)}", target_text, extra_text

    def _refresh(self) -> None:
        """刷新画布、状态栏和 JSON 详情。"""
        self._draw_canvas()
        task_state = self.obs.get("task_state", {})
        safety_state = self.obs.get("safety_state", {})
        event_info = self.obs.get("event_info", {})
        player_pos = event_info.get("player_pos")
        achievement_ratio, target_done, extra_done = self._achievement_progress_text()
        unlocked_text = ", ".join(self.last_unlocked) if self.last_unlocked else "none"
        self.status_var.set(
            "score: {score:.2f}\nlast_reward: {reward:.2f}\nachievements: {achievements}\ntarget_done: {target_done}\nextra_done: {extra_done}\nunlocked: {unlocked}\nstep: {step}\nllm_requests: {llm}\nlast: {last}\npos: {pos}\nphase: {phase}\ndone: {done}\nhealth: {health}\nevent: {event}\nreason: {reason}\nplan: {plan}".format(
                score=self.total_reward,
                step=self.step_count,
                llm=self.llm_request_count,
                last=self.last_action_name,
                pos=player_pos,
                phase=task_state.get("phase"),
                reward=self.last_reward,
                achievements=achievement_ratio,
                target_done=target_done,
                extra_done=extra_done,
                unlocked=unlocked_text,
                done=self.done,
                health=safety_state.get("health"),
                event=event_info.get("event_type"),
                reason=event_info.get("done_reason"),
                plan=self.last_llm_plan,
            )
        )
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, json.dumps(self.obs, ensure_ascii=False, indent=2, default=str))

    def _draw_canvas(self) -> None:
        """绘制制作阶段、背包数量和安全状态。"""
        self.canvas.delete("all")
        task_state = self.obs.get("task_state", {})
        inventory = task_state.get("inventory", {})
        achievements = task_state.get("achievements", {})
        safety_state = self.obs.get("safety_state", {})
        phase = task_state.get("phase", "collect_wood")

        backend_name = self.backend_var.get() if self.env is not None else "unknown"
        self.canvas.create_text(24, 22, anchor="w", fill="#f2f4df", font=("Microsoft YaHei UI", 16, "bold"), text="Crafter 制作石镐")
        self.canvas.create_text(24, 52, anchor="w", fill="#b9c6b1", font=("Microsoft YaHei UI", 10), text=f"backend={backend_name}；Gym 视图以玩家为中心，移动时看 pos/背景变化")

        if self._draw_rgb_observation():
            self._draw_state_summary(x=392, y=92, task_state=task_state, inventory=inventory, safety_state=safety_state)
            return

        self._draw_scripted_summary(task_state, inventory, achievements, safety_state, phase)

    def _draw_scripted_summary(
        self,
        task_state: dict[str, Any],
        inventory: dict[str, Any],
        achievements: dict[str, Any],
        safety_state: dict[str, Any],
        phase: str,
    ) -> None:
        """脚本后端没有真实画面时，绘制阶段和背包摘要。"""

        stages = ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe"]
        x0, y0 = 24, 92
        box_w, box_h, gap = 116, 58, 12
        for index, stage in enumerate(stages):
            x = x0 + index * (box_w + gap)
            complete = achievements.get(stage, False) or (stage == "make_stone_pickaxe" and inventory.get("stone_pickaxe", 0) > 0)
            current = stage == phase
            fill = "#317a45" if complete else "#e0b84b" if current else "#2b3a32"
            outline = "#f8df86" if current else "#73907b"
            self.canvas.create_rectangle(x, y0, x + box_w, y0 + box_h, fill=fill, outline=outline, width=2)
            self.canvas.create_text(x + box_w / 2, y0 + 22, fill="#f7f7ed", font=("Consolas", 9, "bold"), text=stage)
            self.canvas.create_text(x + box_w / 2, y0 + 43, fill="#f7f7ed", font=("Consolas", 8), text="done" if complete else "current" if current else "pending")

        achievement_ratio, target_done, _ = self._achievement_progress_text()
        unlocked_text = ", ".join(self.last_unlocked) if self.last_unlocked else "none"
        self.canvas.create_text(24, 168, anchor="w", fill="#f2f4df", font=("Microsoft YaHei UI", 12, "bold"), text=f"Score {self.total_reward:.2f}   Achievements {achievement_ratio}")
        self.canvas.create_text(24, 190, anchor="w", fill="#b9c6b1", font=("Consolas", 9), text=f"done: {target_done}")
        self.canvas.create_text(430, 190, anchor="w", fill="#b9c6b1", font=("Consolas", 9), text=f"unlocked: {unlocked_text}")

        self.canvas.create_text(24, 200, anchor="w", fill="#f2f4df", font=("Microsoft YaHei UI", 13, "bold"), text="Inventory")
        items = ["wood", "table", "wood_pickaxe", "stone", "stone_pickaxe"]
        for index, item in enumerate(items):
            y = 234 + index * 34
            count = int(inventory.get(item, 0))
            self.canvas.create_text(32, y, anchor="w", fill="#d9ead3", font=("Consolas", 10), text=f"{item}: {count}")
            self.canvas.create_rectangle(170, y - 10, 360, y + 10, fill="#23342b", outline="#52685a")
            width = min(190, count * 55)
            self.canvas.create_rectangle(170, y - 10, 170 + width, y + 10, fill="#70b06f", outline="")

        health = int(safety_state.get("health", 8) or 0)
        danger = float(safety_state.get("danger_level", 0.0) or 0.0)
        self.canvas.create_text(430, 212, anchor="w", fill="#f2f4df", font=("Microsoft YaHei UI", 13, "bold"), text="Safety")
        self.canvas.create_text(430, 244, anchor="w", fill="#d9ead3", font=("Consolas", 11), text=f"health: {health}")
        self.canvas.create_rectangle(430, 270, 620, 292, fill="#23342b", outline="#52685a")
        self.canvas.create_rectangle(430, 270, 430 + max(0, min(190, health * 190 / 8)), 292, fill="#6fbf73", outline="")
        self.canvas.create_text(430, 324, anchor="w", fill="#d9ead3", font=("Consolas", 11), text=f"danger: {danger:.2f}")
        self.canvas.create_rectangle(430, 350, 620, 372, fill="#23342b", outline="#52685a")
        self.canvas.create_rectangle(430, 350, 430 + int(danger * 190), 372, fill="#d66a4f", outline="")

    def _draw_rgb_observation(self) -> bool:
        """把 Gym Crafter 的 RGB 数组直接绘制到 Tk 画布。"""
        image = self.obs.get("image")
        if image is None or isinstance(image, dict):
            return False
        if hasattr(image, "tolist"):
            image = image.tolist()
        if not isinstance(image, list) or not image or not isinstance(image[0], list):
            return False
        height = len(image)
        width = len(image[0])
        if height < 16 or width < 16:
            return False
        scale = max(1, min(5, 320 // max(width, height)))
        photo = tk.PhotoImage(width=width, height=height)
        rows = []
        for row in image:
            colors = []
            for pixel in row:
                if not isinstance(pixel, (list, tuple)) or len(pixel) < 3:
                    colors.append("#000000")
                    continue
                r, g, b = [max(0, min(255, int(value))) for value in pixel[:3]]
                colors.append(f"#{r:02x}{g:02x}{b:02x}")
            rows.append("{" + " ".join(colors) + "}")
        photo.put(" ".join(rows), to=(0, 0, width, height))
        display = photo.zoom(scale, scale) if scale > 1 else photo
        self._image_ref = display
        x, y = 24, 88
        self.canvas.create_image(x, y, image=display, anchor="nw")
        self.canvas.create_rectangle(x - 1, y - 1, x + width * scale + 1, y + height * scale + 1, outline="#f2f4df")
        return True

    def _draw_state_summary(
        self,
        x: int,
        y: int,
        task_state: dict[str, Any],
        inventory: dict[str, Any],
        safety_state: dict[str, Any],
    ) -> None:
        """真实 Gym 画面旁边绘制统一接口状态摘要。"""
        self.canvas.create_text(x, y, anchor="w", fill="#f2f4df", font=("Microsoft YaHei UI", 13, "bold"), text="Unified State")
        achievement_ratio, target_done, extra_done = self._achievement_progress_text()
        unlocked_text = ", ".join(self.last_unlocked[:3]) if self.last_unlocked else "none"
        lines = [
            f"score: {self.total_reward:.2f}",
            f"last_reward: {self.last_reward:.2f}",
            f"achievements: {achievement_ratio}",
            f"target_done: {target_done}",
            f"extra_done: {extra_done}",
            f"unlocked: {unlocked_text}",
            f"step: {self.step_count}",
            f"last: {self.last_action_name}",
            f"pos: {self.obs.get('event_info', {}).get('player_pos')}",
            f"phase: {task_state.get('phase')}",
            f"health: {safety_state.get('health')}",
            f"danger: {float(safety_state.get('danger_level', 0.0) or 0.0):.2f}",
            f"wood: {inventory.get('wood', 0)}",
            f"stone: {inventory.get('stone', 0)}",
            f"table: {inventory.get('table', 0)}",
            f"wood_pickaxe: {inventory.get('wood_pickaxe', 0)}",
            f"stone_pickaxe: {inventory.get('stone_pickaxe', 0)}",
        ]
        for index, line in enumerate(lines):
            self.canvas.create_text(x, y + 32 + index * 22, anchor="w", fill="#d9ead3", font=("Consolas", 10), text=line)


def main() -> None:
    """启动 Crafter 手动控制窗口。"""
    root = tk.Tk()
    CrafterManualControl(root)
    root.mainloop()


if __name__ == "__main__":
    main()
