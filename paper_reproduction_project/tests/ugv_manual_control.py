"""Manual/code switch controller for UGV Unity scenes.

Run:
    conda run -n DT python tests/ugv_manual_control.py

For Unity Editor control, start this script, click Connect with backend=unity,
then press Play in the Unity scene. If Unity is already playing, stop Play first.
Use backend=scripted for a quick local dry run.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import create_env  # noqa: E402
from envs.unity_ugv_env import UnityUGVEnv  # noqa: E402
from src.chap02_dk_abs.domain_knowledge import build_topology, extract_region_rules, extract_skill_descriptors  # noqa: E402
from src.chap02_dk_abs.planning import generate_llm_action_behavior_plan, validate_plan  # noqa: E402


MANUAL_INTERVAL_MS = 120
tk: Any = None
ttk: Any = None


class UGVManualControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("UGV Unity Controller")
        self.env: Optional[UnityUGVEnv] = None
        self.obs: dict[str, Any] = {}
        self.pressed: set[str] = set()
        self.auto_code_running = False
        self.connecting = False
        self._connect_generation = 0
        self.done = False
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = "none"
        self.last_plan = "none"

        self.scene_var = tk.StringVar(value="ugv_parking")
        self.backend_var = tk.StringVar(value="unity")
        self.mode_var = tk.StringVar(value="manual")
        self.worker_var = tk.StringVar(value="0")
        self.port_var = tk.StringVar(value="5004")
        self.seed_var = tk.StringVar(value="0")
        self.timeout_var = tk.StringVar(value="120")
        self.file_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="未连接。先点击 Connect / Reset，再回到 Unity 点击 Play。")

        self._build_layout()
        self._bind_keys()
        self._refresh()
        self.root.after(MANUAL_INTERVAL_MS, self._manual_tick)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=760, height=420, bg="#20242a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.detail_text = tk.Text(left, height=14, wrap="word")
        self.detail_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="ns")

        setup = ttk.LabelFrame(right, text="Connection")
        setup.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(setup, text="scene").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(setup, textvariable=self.scene_var, values=["ugv_parking", "ugv_patrol"], width=14, state="readonly").grid(row=0, column=1, padx=4, pady=3)
        ttk.Label(setup, text="backend").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(setup, textvariable=self.backend_var, values=["unity", "udp", "scripted"], width=14, state="readonly").grid(row=1, column=1, padx=4, pady=3)
        ttk.Label(setup, text="worker").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(setup, textvariable=self.worker_var, width=16).grid(row=2, column=1, padx=4, pady=3)
        ttk.Label(setup, text="base port").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(setup, textvariable=self.port_var, width=16).grid(row=3, column=1, padx=4, pady=3)
        ttk.Label(setup, text="seed").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(setup, textvariable=self.seed_var, width=16).grid(row=4, column=1, padx=4, pady=3)
        ttk.Label(setup, text="timeout sec").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(setup, textvariable=self.timeout_var, width=16).grid(row=5, column=1, padx=4, pady=3)
        ttk.Label(setup, text="player path").grid(row=6, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(setup, textvariable=self.file_var, width=24).grid(row=6, column=1, padx=4, pady=3)
        self.connect_button = ttk.Button(setup, text="Connect / Reset", command=self.connect_or_reset)
        self.connect_button.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=5)

        mode = ttk.LabelFrame(right, text="Control mode")
        mode.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode, text="Manual keyboard", variable=self.mode_var, value="manual", command=self._set_mode).grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Radiobutton(mode, text="Code planner", variable=self.mode_var, value="code", command=self._set_mode).grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Button(mode, text="Run next code skill", command=self.run_next_code_skill).grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(mode, text="Toggle auto code", command=self.toggle_auto_code).grid(row=3, column=0, sticky="ew", padx=4, pady=4)

        manual = ttk.LabelFrame(right, text="Manual action")
        manual.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(manual, text="W/Up forward, S/Down reverse").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(manual, text="A/Left steer left, D/Right steer right").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(manual, text="Space stops. Holding keys streams actions.").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Button(manual, text="Stop now", command=self.stop_now).grid(row=3, column=0, sticky="ew", padx=4, pady=4)

        ttk.Label(right, textvariable=self.status_var, justify="left", wraplength=300).grid(row=3, column=0, sticky="ew")

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

    def connect_or_reset(self) -> None:
        if self.connecting:
            self.status_var.set(
                "正在等待 Unity 连接。\n"
                "如果 Unity 已经是蓝色 Play 状态，请先停止 Play，再保持本窗口等待，然后重新点击 Unity Play。"
            )
            return
        self.close_env()
        scene = self.scene_var.get()
        seed = int(self.seed_var.get() or 0)
        config: dict[str, Any] = {
            "backend": self.backend_var.get(),
            "worker_id": int(self.worker_var.get() or 0),
            "base_port": int(self.port_var.get() or 5004),
            "seed": seed,
            "timeout_wait": int(self.timeout_var.get() or 120),
            "control_mode": self.mode_var.get(),
        }
        file_name = self.file_var.get().strip()
        if file_name:
            config["file_name"] = file_name
        self.connecting = True
        self._connect_generation += 1
        generation = self._connect_generation
        self.connect_button.state(["disabled"])
        if config["backend"] == "udp":
            self.status_var.set(
                "UDP 控制模式：不等待 ML-Agents 连接。\n"
                "请保持 Unity 场景在 Play 状态，然后直接用 W/A/S/D 控车。\n"
                "Unity 会监听 127.0.0.1:5055。"
            )
        elif config["backend"] == "unity" and not file_name:
            self.status_var.set(
                "正在等待 Unity Editor...\n"
                "重要：如果 Unity 已经在 Play，请先停止 Play。\n"
                "保持本窗口等待后，再回到 Unity 点击 Play。\n"
                "Unity Project Settings > ML-Agents > Connect Trainer 必须开启。"
            )
        else:
            self.status_var.set("正在连接...")
        threading.Thread(target=self._connect_worker, args=(generation, scene, config, seed), daemon=True).start()

    def _connect_worker(self, generation: int, scene: str, config: dict[str, Any], seed: int) -> None:
        env: Optional[UnityUGVEnv] = None
        try:
            env = create_env(scene, config)  # type: ignore[assignment]
            env.set_control_mode(str(config.get("control_mode", "manual")))
            obs = env.reset(seed=seed)
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self.root.after(0, self._on_connect_failed, generation, exc)
            return
        self.root.after(0, self._on_connect_success, generation, env, obs)

    def _on_connect_success(self, generation: int, env: UnityUGVEnv, obs: dict[str, Any]) -> None:
        if generation != self._connect_generation:
            try:
                env.close()
            except Exception:
                pass
            return
        self.connecting = False
        self.connect_button.state(["!disabled"])
        self.env = env
        self.obs = obs
        self.done = False
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = "reset"
        self.last_plan = "none"
        self._refresh()

    def _on_connect_failed(self, generation: int, exc: Exception) -> None:
        if generation != self._connect_generation:
            return
        self.connecting = False
        self.connect_button.state(["!disabled"])
        self.status_var.set(
            "连接失败：\n"
            f"{exc}\n\n"
            "正确顺序：先停止 Unity Play，再点击本窗口 Connect / Reset，最后回到 Unity 点击 Play。\n"
            "同时检查 Unity Project Settings > ML-Agents > Connect Trainer 是否开启。"
        )

    def close_env(self) -> None:
        self._connect_generation += 1
        self.connecting = False
        if hasattr(self, "connect_button"):
            self.connect_button.state(["!disabled"])
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None

    def _set_mode(self) -> None:
        if self.env is not None:
            self.env.set_control_mode(self.mode_var.get())
        if self.mode_var.get() == "manual":
            self.auto_code_running = False
        self._refresh()

    def _on_key_press(self, event: tk.Event) -> None:
        self.pressed.add(str(event.keysym).lower())
        if str(event.keysym).lower() == "space":
            self.stop_now()

    def _on_key_release(self, event: tk.Event) -> None:
        self.pressed.discard(str(event.keysym).lower())

    def _manual_tick(self) -> None:
        if self.env is not None and self.mode_var.get() == "manual" and not self.done:
            steer, motor = self._current_manual_action()
            if self.pressed or abs(steer) > 0.001 or abs(motor) > 0.001:
                self._send_manual(steer, motor, "manual_keys")
        self.root.after(MANUAL_INTERVAL_MS, self._manual_tick)

    def _current_manual_action(self) -> tuple[float, float]:
        steer = 0.0
        motor = 0.0
        if "left" in self.pressed or "a" in self.pressed:
            steer -= 1.0
        if "right" in self.pressed or "d" in self.pressed:
            steer += 1.0
        if "up" in self.pressed or "w" in self.pressed:
            motor += 1.0
        if "down" in self.pressed or "s" in self.pressed:
            motor -= 1.0
        if "space" in self.pressed:
            steer = 0.0
            motor = 0.0
        return steer, motor

    def _send_manual(self, steer: float, motor: float, label: str) -> None:
        if self.env is None:
            return
        try:
            self.obs, self.last_reward, self.done, _ = self.env.manual_step(steer, motor, label=label)
            self.total_reward += float(self.last_reward)
            self.last_action = f"{label}: steer={steer:.2f}, motor={motor:.2f}"
            self._refresh()
        except Exception as exc:
            self.status_var.set(f"Manual action failed:\n{exc}")

    def stop_now(self) -> None:
        self.pressed.clear()
        self._send_manual(0.0, 0.0, "manual_stop")

    def run_next_code_skill(self) -> None:
        if self.env is None:
            return
        self.mode_var.set("code")
        self.env.set_control_mode("code")
        try:
            plan = self._build_code_plan()
            if not plan:
                self.status_var.set("No code plan available.")
                return
            behavior = plan[0]
            self.obs, self.last_reward, self.done, _ = self.env.apply_skill(behavior.skill_id, behavior.to_skill_args())
            self.total_reward += float(self.last_reward)
            self.last_action = behavior.skill_id
            self._refresh()
        except Exception as exc:
            self.status_var.set(f"Code action failed:\n{exc}")

    def toggle_auto_code(self) -> None:
        self.auto_code_running = not self.auto_code_running
        if self.auto_code_running:
            self.mode_var.set("code")
            self._set_mode()
            self.root.after(200, self._auto_code_tick)
        self._refresh()

    def _auto_code_tick(self) -> None:
        if not self.auto_code_running or self.env is None or self.done or self.mode_var.get() != "code":
            self.auto_code_running = False
            self._refresh()
            return
        self.run_next_code_skill()
        if self.auto_code_running:
            self.root.after(600, self._auto_code_tick)

    def _build_code_plan(self):
        if self.env is None:
            return []
        scene = self.scene_var.get()
        task_state = self.env.get_task_state()
        domain = self.env.get_domain_knowledge()
        domain = {
            **domain,
            "environment_topology": build_topology(scene, domain),
            "region_rules": extract_region_rules(scene, domain),
            "skill_policies": extract_skill_descriptors(scene, domain),
        }
        result = generate_llm_action_behavior_plan(
            scene,
            task_state,
            domain,
            {
                "mode": "llm_simulated",
                "fallback_to_rule": True,
                "planning_horizon": 8,
                "llm": {"provider": "simulated", "model": "gpt-4o-mini-sim", "temperature": 0.0},
            },
        )
        ok, errors = validate_plan(scene, result.plan, max_plan_length=16)
        if not ok:
            raise RuntimeError(f"Invalid plan: {errors}")
        self.last_plan = " -> ".join(item.skill_id for item in result.plan[:8])
        return result.plan

    def _refresh(self) -> None:
        self._draw_canvas()
        if self.env is None:
            return
        task = self.obs.get("task_state", {})
        safety = self.obs.get("safety_state", {})
        control = self.obs.get("control_state", {})
        self.status_var.set(
            "scene: {scene}\nbackend: {backend}\nmode: {mode}\nphase: {phase}\nrisk: {risk}\nreward: {reward:.2f}\ntotal: {total:.2f}\ndone: {done}\nlast: {last}\nplan: {plan}\nrecommended: {rec}".format(
                scene=self.scene_var.get(),
                backend=self.backend_var.get(),
                mode=control.get("control_mode", self.mode_var.get()),
                phase=task.get("phase"),
                risk=safety.get("risk_score"),
                reward=self.last_reward,
                total=self.total_reward,
                done=self.done,
                last=self.last_action,
                plan=self.last_plan,
                rec=safety.get("recommended_safe_action"),
            )
        )
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, json.dumps(self.obs, ensure_ascii=False, indent=2, default=str))

    def _draw_canvas(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(24, 24, anchor="w", fill="#f5f7fa", font=("Consolas", 18, "bold"), text="UGV Unity Controller")
        self.canvas.create_text(24, 54, anchor="w", fill="#b8c0cc", font=("Consolas", 11), text="manual and code modes share the same steer/motor action channel")

        if self.env is None:
            lines = [
                "连接顺序",
                "1. 先停止 Unity 的 Play 状态",
                "2. 在本窗口点击 Connect / Reset",
                "3. 看到正在等待 Unity 后，再回到 Unity 点击 Play",
                "4. 连接成功后，用 W/A/S/D 或方向键控制无人车",
                "",
                "如果你已经先点了 Unity Play，请停止 Play 后重新按以上顺序来一次。",
            ]
            y = 112
            for index, line in enumerate(lines):
                fill = "#8bd7b5" if index == 0 else "#d8e0ea"
                size = 15 if index == 0 else 12
                weight = "bold" if index == 0 else "normal"
                self.canvas.create_text(34, y, anchor="w", fill=fill, font=("Consolas", size, weight), text=line)
                y += 34 if index == 0 else 28
            return

        task = self.obs.get("task_state", {})
        safety = self.obs.get("safety_state", {})
        control = self.obs.get("control_state", {})
        vehicle = self.obs.get("vehicle_state", {})
        stage = str(task.get("phase", "not_connected"))
        mode = str(control.get("control_mode", self.mode_var.get()))
        risk = float(safety.get("risk_score", 0.0) or 0.0)

        stages = self._stage_names()
        x0, y0 = 34, 105
        box_w, box_h, gap = 88, 54, 10
        for index, name in enumerate(stages):
            x = x0 + index * (box_w + gap)
            fill = "#2f8f68" if name == stage else "#313842"
            outline = "#8bd7b5" if name == stage else "#59616e"
            self.canvas.create_rectangle(x, y0, x + box_w, y0 + box_h, fill=fill, outline=outline, width=2)
            self.canvas.create_text(x + box_w / 2, y0 + 27, fill="#f5f7fa", font=("Consolas", 8, "bold"), text=name)

        self.canvas.create_text(34, 205, anchor="w", fill="#f5f7fa", font=("Consolas", 13, "bold"), text=f"mode={mode}  last={self.last_action}")
        self.canvas.create_text(34, 235, anchor="w", fill="#cbd3df", font=("Consolas", 11), text=f"speed={vehicle.get('speed')}  motor={vehicle.get('motor_command')}  steer={vehicle.get('steer_command')}")
        self.canvas.create_text(34, 265, anchor="w", fill="#cbd3df", font=("Consolas", 11), text=f"traffic={safety.get('traffic_light_state')}  region={safety.get('current_region_id')}  recommended={safety.get('recommended_safe_action')}")

        self.canvas.create_rectangle(34, 320, 360, 348, fill="#161b22", outline="#59616e")
        self.canvas.create_rectangle(34, 320, 34 + int(326 * max(0.0, min(1.0, risk))), 348, fill="#d15b56", outline="")
        self.canvas.create_text(197, 334, fill="#f5f7fa", font=("Consolas", 11, "bold"), text=f"risk {risk:.2f}")
        self.canvas.create_text(34, 380, anchor="w", fill="#b8c0cc", font=("Consolas", 10), text=f"code plan: {self.last_plan}")

    def _stage_names(self) -> list[str]:
        if self.scene_var.get() == "ugv_patrol":
            return ["outer_patrol", "checkpoint", "intersection", "route_switch", "inner_patrol", "parking_area", "parking"]
        return ["main_road", "obstacle_zone", "target_ball", "intersection", "street", "parking_area", "parking"]

    def destroy(self) -> None:
        self.close_env()
        self.root.destroy()


def main() -> None:
    root = create_tk_root()
    app = UGVManualControlApp(root)
    root.protocol("WM_DELETE_WINDOW", app.destroy)
    root.mainloop()


def ensure_tk_loaded() -> None:
    global tk, ttk
    if tk is not None and ttk is not None:
        return
    import tkinter as tk_module
    from tkinter import ttk as ttk_module

    tk = tk_module
    ttk = ttk_module


def create_tk_root() -> tk.Tk:
    """Create Tk reliably in Conda environments with Library/lib Tcl/Tk data."""
    conda_tcl_root = Path(sys.prefix) / "Library" / "lib"
    if conda_tcl_root.exists():
        original_cwd = Path.cwd()
        old_tcl = os.environ.get("TCL_LIBRARY")
        old_tk = os.environ.get("TK_LIBRARY")
        os.environ["TCL_LIBRARY"] = "tcl8.6"
        os.environ["TK_LIBRARY"] = "tk8.6"
        try:
            os.chdir(conda_tcl_root)
            ensure_tk_loaded()
            return tk.Tk()
        finally:
            os.chdir(original_cwd)
            if old_tcl is None:
                os.environ.pop("TCL_LIBRARY", None)
            else:
                os.environ["TCL_LIBRARY"] = old_tcl
            if old_tk is None:
                os.environ.pop("TK_LIBRARY", None)
            else:
                os.environ["TK_LIBRARY"] = old_tk
    ensure_tk_loaded()
    return tk.Tk()


if __name__ == "__main__":
    main()
