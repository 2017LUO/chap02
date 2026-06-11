"""Highway 环境可视化。

运行方式：
    conda run -n DT python tests/highway/manual_control.py
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

from envs import HighwayEnvWrapper  # noqa: E402
from src.chap02_dk_abs.domain_knowledge import (  # noqa: E402
    build_topology,
    extract_region_rules,
    extract_skill_descriptors,
)
from src.chap02_dk_abs.planning import generate_llm_action_behavior_plan, validate_plan  # noqa: E402


HIGHWAY_LLM_ACTIONS = ["lane_left", "keep_lane", "lane_right", "speed_up", "slow_down"]
ACTION_LABELS = {
    "lane_left": "lane_left",
    "keep_lane": "idle / keep_lane",
    "lane_right": "lane_right",
    "speed_up": "faster / speed_up",
    "slow_down": "slower / slow_down",
}


class HighwayManualControl:
    """用验证 Highway 封装的动作、车道和安全字段。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Highway 大语言模型分层决策")
        self.env: Optional[HighwayEnvWrapper] = None
        self.backend_var = tk.StringVar(value="gymnasium")
        self.backend_status_var = tk.StringVar(value="")
        self.seed_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="")
        self.auto_button_var = tk.StringVar(value="auto LLM")
        self._image_ref: Optional[tk.PhotoImage] = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.step_count = 0
        self.llm_request_count = 0
        self.last_action_name = "reset"
        self.last_llm_plan = ""
        self.last_llm_rationale = ""
        self.auto_running = False
        self.auto_lane_change_done = False
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

        self.canvas = tk.Canvas(left, width=760, height=430, bg="#1f2429", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.detail_text = tk.Text(left, height=16, wrap="word")
        self.detail_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="ns")

        seed_box = ttk.LabelFrame(right, text="重置")
        seed_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(seed_box, text="backend").grid(row=0, column=0, padx=4, pady=4)
        ttk.Combobox(seed_box, textvariable=self.backend_var, values=["gymnasium", "scripted"], width=10, state="readonly").grid(
            row=0, column=1, padx=4, pady=4
        )
        ttk.Label(seed_box, text="seed").grid(row=1, column=0, padx=4, pady=4)
        ttk.Entry(seed_box, textvariable=self.seed_var, width=8).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(seed_box, text="reset", command=self.reset_env).grid(row=1, column=2, padx=4, pady=4)
        ttk.Label(seed_box, textvariable=self.backend_status_var, justify="left", wraplength=270).grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 4)
        )

        action_box = ttk.LabelFrame(right, text="原始动作")
        action_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        actions = [
            ("lane_left", "lane_left"),
            ("idle", "idle"),
            ("lane_right", "lane_right"),
            ("faster", "faster"),
            ("slower", "slower"),
        ]
        for index, (label, action_name) in enumerate(actions):
            ttk.Button(action_box, text=label, command=lambda name=action_name: self.run_action(name)).grid(
                row=index, column=0, sticky="ew", padx=4, pady=3
            )

        skill_box = ttk.LabelFrame(right, text="技能策略")
        skill_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        skills = [
            ("keep_lane", "keep_lane", {}),
            ("safe_follow x3", "safe_follow", {"max_steps": 3}),
            ("speed_up", "speed_up", {}),
            ("slow_down", "slow_down", {}),
            ("lane_left", "lane_left", {}),
            ("lane_right", "lane_right", {}),
        ]
        for index, (label, skill_id, skill_args) in enumerate(skills):
            ttk.Button(skill_box, text=label, command=lambda sid=skill_id, args=skill_args: self.run_skill(sid, args)).grid(
                row=index, column=0, sticky="ew", padx=4, pady=3
            )

        auto_box = ttk.LabelFrame(right, text="自动演示")
        auto_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(auto_box, textvariable=self.auto_button_var, command=self.toggle_auto_run).grid(
            row=0, column=0, sticky="ew", padx=4, pady=3
        )
        ttk.Label(auto_box, text="5 actions: lane_left / keep_lane / lane_right / speed_up / slow_down", wraplength=270).grid(
            row=1, column=0, sticky="ew", padx=4, pady=(0, 4)
        )

        ttk.Label(right, textvariable=self.status_var, justify="left", wraplength=270).grid(
            row=4, column=0, sticky="ew", pady=(4, 0)
        )
        self.root.bind("<Left>", lambda _: self.run_action("lane_left"))
        self.root.bind("<Right>", lambda _: self.run_action("lane_right"))
        self.root.bind("<Up>", lambda _: self.run_action("faster"))
        self.root.bind("<Down>", lambda _: self.run_action("slower"))
        self.root.bind("<space>", lambda _: self.run_action("idle"))

    def reset_env(self) -> None:
        """按输入随机种子重置环境。"""
        seed = int(self.seed_var.get() or 0)
        backend = self.backend_var.get()
        self.auto_running = False
        self.auto_button_var.set("auto LLM")
        try:
            self.env = HighwayEnvWrapper(
                {
                    "backend": backend,
                    "env_id": "highway-v0",
                    "render_mode": "rgb_array",
                    "lanes_count": 4,
                    "vehicles_count": 50,
                    "duration": 100,
                }
            )
            self.backend_status_var.set(f"backend: {backend}")
        except Exception as exc:
            self.env = HighwayEnvWrapper({"backend": "scripted", "lanes_count": 4, "vehicles_count": 50, "duration": 100})
            self.backend_status_var.set(f"官方 highway-env 初始化失败，已回退 scripted:\n{exc}")
        self.obs = self.env.reset(seed=seed)
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.step_count = 0
        self.llm_request_count = 0
        self.last_action_name = "reset"
        self.last_llm_plan = ""
        self.last_llm_rationale = ""
        self.auto_lane_change_done = False
        self.done = False
        self._refresh()

    def run_action(self, action_name: str) -> None:
        """执行一个 Highway 原始离散动作。"""
        if self.done or self.env is None:
            return
        try:
            self.obs, self.last_reward, self.done, _ = self.env.step(action_name)
            self.total_reward += float(self.last_reward)
            self.step_count += 1
            self.last_action_name = action_name
        except Exception as exc:
            self.status_var.set(f"action failed: {action_name}\n{exc}")
            return
        self._refresh()

    def run_skill(self, skill_id: str, skill_args: dict[str, Any]) -> None:
        """执行一个 Highway 技能策略。"""
        if self.done or self.env is None:
            return
        try:
            self.obs, self.last_reward, self.done, info = self.env.apply_skill(skill_id, skill_args)
            self.total_reward += float(self.last_reward)
            self.step_count += int(info.get("skill_steps", 1) or 1)
            self.last_action_name = skill_id
        except Exception as exc:
            self.status_var.set(f"skill failed: {skill_id}\n{exc}")
            return
        self._refresh()

    def toggle_auto_run(self) -> None:
        """Start or stop automatic LLM step-by-step driving."""
        if self.env is None:
            return
        if self.auto_running:
            self.auto_running = False
            self.auto_button_var.set("auto LLM")
            self._refresh()
            return
        if self.done:
            self.reset_env()
        self.auto_running = True
        self.auto_button_var.set("stop auto")
        self.root.after(100, self._auto_tick)

    def _auto_tick(self) -> None:
        """Ask the simulated LLM to rank five actions, then execute the first."""
        if not self.auto_running or self.env is None:
            return
        task_state = self.env.get_task_state()
        if self.done or int(task_state.get("time_step", 0) or 0) >= int(task_state.get("max_episode_steps", 100) or 100):
            self.auto_running = False
            self.auto_button_var.set("auto LLM")
            self._refresh()
            return
        try:
            llm_result = generate_llm_action_behavior_plan(
                "highway",
                self._current_planning_state(),
                self._current_domain_knowledge(),
                self._auto_planner_config(),
            )
            plan = llm_result.plan
            ok, errors = validate_plan("highway", plan, max_plan_length=5)
            if not ok or not plan:
                self.status_var.set(f"auto plan invalid: {errors}")
                self.auto_running = False
                self.auto_button_var.set("auto LLM")
                return
            plan, promoted = self._promote_demo_lane_change(plan)
            chosen = plan[0]
            self.llm_request_count += 1
            self.last_llm_plan = " > ".join(ACTION_LABELS.get(item.skill_id, item.skill_id) for item in plan[:5])
            self.last_llm_rationale = (
                f"{llm_result.provider}/{llm_result.model}"
                + ("; demo lane-change promoted" if promoted else "")
            )
            self.run_skill(chosen.skill_id, chosen.to_skill_args())
            if chosen.skill_id in {"lane_left", "lane_right"}:
                self.auto_lane_change_done = True
        except Exception as exc:
            self.status_var.set(f"auto failed:\n{exc}")
            self.auto_running = False
            self.auto_button_var.set("auto LLM")
            return
        if self.auto_running:
            self.root.after(350, self._auto_tick)

    def _promote_demo_lane_change(self, plan: list[Any]) -> tuple[list[Any], bool]:
        """Promote one safe lane change early so the LLM demo shows hierarchy."""
        if self.env is None or self.auto_lane_change_done:
            return plan, False
        next_llm_request = self.llm_request_count + 1
        if next_llm_request < 8 or next_llm_request > 20:
            return plan, False
        lane_change = self._best_safe_lane_change()
        if lane_change is None or plan[0].skill_id == lane_change:
            return plan, False
        for index, item in enumerate(plan):
            if item.skill_id == lane_change:
                return [item] + plan[:index] + plan[index + 1 :], True
        return plan, False

    def _best_safe_lane_change(self) -> Optional[str]:
        """Pick a safe adjacent lane, preferring the side with more front space."""
        if self.env is None:
            return None
        task_state = self.env.get_task_state()
        safety_state = self.env.get_safety_state()
        lane_id = int(task_state.get("lane_id", 0) or 0)
        lanes_count = int(task_state.get("lanes_count", 4) or 4)
        options: list[tuple[str, float]] = []
        if bool(safety_state.get("left_lane_safe")) and lane_id > 0:
            options.append(("lane_left", self._front_gap_for_lane(lane_id - 1)))
        if bool(safety_state.get("right_lane_safe")) and lane_id < lanes_count - 1:
            options.append(("lane_right", self._front_gap_for_lane(lane_id + 1)))
        if not options:
            return None
        return max(options, key=lambda item: item[1])[0]

    def _front_gap_for_lane(self, lane_id: int) -> float:
        gaps: list[float] = []
        for vehicle in self.obs.get("nearby_vehicles", []):
            try:
                if int(vehicle.get("lane_id")) != lane_id:
                    continue
                distance = float(vehicle.get("distance", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if distance > 0.0:
                gaps.append(distance)
        return min(gaps) if gaps else 1_000.0

    def _current_domain_knowledge(self) -> dict[str, Any]:
        """Build a five-action Highway prompt domain for visual LLM control."""
        if self.env is None:
            return {}
        domain = self.env.get_domain_knowledge()
        skill_policies = extract_skill_descriptors("highway", domain)
        five_skill_policies = {skill_id: skill_policies[skill_id] for skill_id in HIGHWAY_LLM_ACTIONS if skill_id in skill_policies}
        return {
            **domain,
            "environment_topology": build_topology("highway", domain),
            "region_rules": extract_region_rules("highway", domain),
            "skill_policies": five_skill_policies,
            "llm_action_space": {skill_id: ACTION_LABELS[skill_id] for skill_id in HIGHWAY_LLM_ACTIONS},
        }

    def _current_planning_state(self) -> dict[str, Any]:
        """Expose ego, surrounding vehicles and safety fields to the LLM prompt."""
        if self.env is None:
            return {}
        state = dict(self.env.get_task_state())
        domain = self.env.get_domain_knowledge()
        lanes = (domain.get("environment_topology") or {}).get("lanes") or [0, 1, 2, 3]
        nearby = list(self.obs.get("nearby_vehicles", []))
        nearby.sort(key=lambda item: abs(float(item.get("distance", 0.0) or 0.0)))
        state.update(
            {
                "lanes_count": len(lanes),
                "ego_state": self.obs.get("ego_state", {}),
                "safety_state": self.env.get_safety_state(),
                "event_info": self.env.get_event_info(),
                "nearby_vehicles": nearby[:10],
                "allowed_actions": HIGHWAY_LLM_ACTIONS,
                "decision_rule": "rank the five actions and execute the first item only for this environment step",
            }
        )
        return state

    @staticmethod
    def _auto_planner_config() -> dict[str, Any]:
        """Planner config for one-step Highway visual demonstration."""
        return {
            "mode": "llm_simulated",
            "fallback_to_rule": True,
            "replan_each_step": True,
            "planning_horizon": 5,
            "llm": {
                "provider": "simulated",
                "model": "gpt-4o-mini-sim",
                "temperature": 0.0,
                "response_format": "json_object",
            },
        }

    def _refresh(self) -> None:
        """刷新画布、状态栏和 JSON 详情。"""
        self._draw_canvas()
        task_state = self.obs.get("task_state", {})
        safety_state = self.obs.get("safety_state", {})
        event_info = self.obs.get("event_info", {})
        self.status_var.set(
            "score: {score:.2f}\nlast_reward: {reward:.2f}\nllm_requests: {llm}\nlast: {last}\nllm_rank: {rank}\nt: {t}/{max_t}\nlane: {lane}\nspeed: {speed:.1f}\nfront: {front}\nleft_safe: {left}\nright_safe: {right}\ndone: {done}\nrisk: {risk:.2f}\nevent: {event}\nreason: {reason}".format(
                score=self.total_reward,
                t=task_state.get("time_step", 0),
                max_t=task_state.get("max_episode_steps", 100),
                lane=task_state.get("lane_id", 0),
                speed=float(task_state.get("speed", 0.0) or 0.0),
                reward=self.last_reward,
                llm=self.llm_request_count,
                last=self.last_action_name,
                rank=self.last_llm_plan or "none",
                front=safety_state.get("front_vehicle_dist"),
                left=safety_state.get("left_lane_safe"),
                right=safety_state.get("right_lane_safe"),
                done=self.done,
                risk=float(safety_state.get("risk_score", 0.0) or 0.0),
                event=event_info.get("event_type"),
                reason=event_info.get("done_reason"),
            )
        )
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, json.dumps(self.obs, ensure_ascii=False, indent=2, default=str))

    def _draw_canvas(self) -> None:
        """绘制车道、自车、周围车辆和安全距离。"""
        self.canvas.delete("all")
        task_state = self.obs.get("task_state", {})
        safety_state = self.obs.get("safety_state", {})
        nearby = self.obs.get("nearby_vehicles", [])

        lanes_count = 4
        road_x0, road_y0 = 40, 70
        road_w, lane_h = 660, 76
        ego_x = 180

        self.canvas.create_text(40, 26, anchor="w", fill="#f4f7fb", font=("Microsoft YaHei UI", 16, "bold"), text="Highway 大语言模型分层决策")
        self.canvas.create_text(
            455,
            27,
            anchor="w",
            fill="#f4f7fb",
            font=("Consolas", 11, "bold"),
            text=f"score {self.total_reward:.2f}  llm {self.llm_request_count}",
        )
        self.canvas.create_text(40, 52, anchor="w", fill="#c5ced8", font=("Microsoft YaHei UI", 10), text=f"backend={self.backend_var.get()}；默认使用官方 highway-env 渲染")

        if self._draw_official_render():
            self._draw_overlay_summary(task_state, safety_state)
            return

        for lane in range(lanes_count):
            y0 = road_y0 + lane * lane_h
            fill = "#2c3338" if lane % 2 == 0 else "#303940"
            self.canvas.create_rectangle(road_x0, y0, road_x0 + road_w, y0 + lane_h, fill=fill, outline="")
            self.canvas.create_text(12, y0 + lane_h / 2, fill="#aeb9c4", font=("Consolas", 10), text=str(lane))
            if lane > 0:
                self.canvas.create_line(road_x0, y0, road_x0 + road_w, y0, fill="#d7dde3", dash=(12, 10), width=2)

        lane_id = int(task_state.get("lane_id", 0) or 0)
        ego_y = road_y0 + lane_id * lane_h + lane_h / 2
        self._draw_vehicle(ego_x, ego_y, "#43a5f5", "ego")

        for vehicle in nearby:
            lane = vehicle.get("lane_id")
            if lane is None:
                continue
            distance = float(vehicle.get("distance", 0.0) or 0.0)
            x = ego_x + distance * 2.1
            if x < road_x0 + 10 or x > road_x0 + road_w - 10:
                continue
            y = road_y0 + int(lane) * lane_h + lane_h / 2
            self._draw_vehicle(x, y, "#e2a84b", str(vehicle.get("id", "veh")))

        safe_distance = 10.0
        self.canvas.create_line(ego_x + safe_distance * 2.1, ego_y - 30, ego_x + safe_distance * 2.1, ego_y + 30, fill="#80d083", dash=(4, 4))
        self.canvas.create_line(ego_x - safe_distance * 2.1, ego_y - 30, ego_x - safe_distance * 2.1, ego_y + 30, fill="#80d083", dash=(4, 4))

        risk = float(safety_state.get("risk_score", 0.0) or 0.0)
        self.canvas.create_text(40, 392, anchor="w", fill="#f4f7fb", font=("Consolas", 11), text=f"front_dist={safety_state.get('front_vehicle_dist')}  ttc={safety_state.get('time_to_collision')}  recommended={safety_state.get('recommended_safe_action')}")
        self.canvas.create_rectangle(520, 382, 700, 404, fill="#22292e", outline="#65737e")
        self.canvas.create_rectangle(520, 382, 520 + int(risk * 180), 404, fill="#d45c54", outline="")
        self.canvas.create_text(610, 393, fill="#f4f7fb", font=("Consolas", 10, "bold"), text=f"risk {risk:.2f}")
        self.canvas.create_text(40, 418, anchor="w", fill="#c5ced8", font=("Consolas", 9), text=f"llm_rank={self.last_llm_plan or 'none'}")

    def _draw_vehicle(self, x: float, y: float, fill: str, label: str) -> None:
        """在画布上绘制简化车辆图标。"""
        self.canvas.create_rectangle(x - 22, y - 12, x + 22, y + 12, fill=fill, outline="#f4f7fb", width=1)
        self.canvas.create_rectangle(x - 14, y - 17, x + 12, y - 8, fill=fill, outline="#f4f7fb", width=1)
        self.canvas.create_text(x, y + 25, fill="#f4f7fb", font=("Consolas", 8), text=label)

    def _draw_official_render(self) -> bool:
        """绘制官方 highway-env 的 RGB 渲染帧。"""
        if self.env is None:
            return False
        image = self.env.render_rgb()
        if image is None:
            return False
        if hasattr(image, "tolist"):
            image = image.tolist()
        if not isinstance(image, list) or not image or not isinstance(image[0], list):
            return False
        height = len(image)
        width = len(image[0])
        scale = max(1, min(2, 660 // max(width, 1)))
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
        x, y = 40, 82
        self.canvas.create_image(x, y, image=display, anchor="nw")
        self.canvas.create_rectangle(x - 1, y - 1, x + width * scale + 1, y + height * scale + 1, outline="#f4f7fb")
        return True

    def _draw_overlay_summary(self, task_state: dict[str, Any], safety_state: dict[str, Any]) -> None:
        """在官方渲染下方显示统一接口状态摘要。"""
        lines = [
            f"score={self.total_reward:.2f}",
            f"last_reward={self.last_reward:.2f}",
            f"llm={self.llm_request_count}",
            f"t={task_state.get('time_step', 0)}/100",
            f"lane={task_state.get('lane_id', 0)}",
            f"speed={float(task_state.get('speed', 0.0) or 0.0):.1f}",
            f"front_dist={safety_state.get('front_vehicle_dist')}",
            f"risk={float(safety_state.get('risk_score', 0.0) or 0.0):.2f}",
            f"recommended={safety_state.get('recommended_safe_action')}",
        ]
        self.canvas.create_text(40, 270, anchor="w", fill="#f4f7fb", font=("Consolas", 11), text="  ".join(lines))
        lane_lines = [
            f"left_safe={safety_state.get('left_lane_safe')}",
            f"right_safe={safety_state.get('right_lane_safe')}",
            f"last={self.last_action_name}",
        ]
        self.canvas.create_text(40, 300, anchor="w", fill="#f4f7fb", font=("Consolas", 11), text="  ".join(lane_lines))
        self.canvas.create_text(40, 330, anchor="w", fill="#c5ced8", font=("Consolas", 10), text=f"llm_rank={self.last_llm_plan or 'none'}")


def main() -> None:
    """启动 Highway 手动控制窗口。"""
    root = tk.Tk()
    HighwayManualControl(root)
    root.mainloop()


if __name__ == "__main__":
    main()
