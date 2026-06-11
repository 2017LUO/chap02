#!/usr/bin/env python3
"""
KEEP_LAST_wait.py  – Keep‑Last baseline **with wait‑for‑LLM semantics**
---------------------------------------------------------------------
Compared with the previous “plus” version, the only behavioural change is:
• **During LLM inference we do *not* execute any action.** The loop polls the
  decision thread and simply sleeps until the skill is ready, then starts
  executing it. All other extensions (reward‑10 counter, collision/stuck
  handling, CSV logging) are retained.

Waiting avoids the placeholder *keep‑last* action bias and ensures that each
sub‑task starts *exactly* with the skill returned by the LLM.
"""

import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from CLC_Rl.skills_manager.skill_manager import SkillLibrary
from utils.loader import get_config, get_environment
from utils.scene_id import SceneID
from utils.seed import set_random_seed
from prompt.Test import decide

# =============================================================================
# Constants
# =============================================================================
FALLBACK_SKILL = 0
STEP_SCALE = 4                  # env.step → logical step multiplier
MAX_STEPS = 600                 # ≥ : treat as stuck → rerun episode
WAIT_SLEEP = 0.02               # polling sleep (seconds) while waiting LLM

# --- reward‑10 counting (3rd sub‑task) ------------------------------------- #
THIRD_TASK_ID = 2                # cur_id of the sub‑task to watch
REWARD_VALUE = 10               # reward value to count
TARGET_COUNT = 3                # "exact‑three" flag threshold

# =============================================================================
# 1. Async decide thread management
# =============================================================================
DecisionState = Dict[int, Dict[str, object]]  # task_id → {thread, holder}


def start_decision(task_id: int, state: DecisionState) -> None:
    """Launch decide(task_id) in a daemon thread unless already running."""
    if task_id in state:
        return

    holder: Dict[str, object] = {}

    def _worker() -> None:
        t0 = time.perf_counter()
        four_skills, cur_skill = decide(task_id)
        holder.update({
            "four_skills": four_skills,
            "cur_skill":   cur_skill,
            "elapsed":     time.perf_counter() - t0,  # wall‑clock seconds
        })

    th = threading.Thread(target=_worker, daemon=True)
    state[task_id] = {"thread": th, "holder": holder}
    th.start()


# =============================================================================
# 2. Single‑episode runner (wait‑for‑LLM version)
# =============================================================================
EpiReturn = Tuple[
    float,        # ep_reward
    List[int],    # success flags per task (6)
    int,          # total_steps (scaled)
    List[int],    # per‑task steps  (scaled, 6)
    List[float],  # per‑task LLM times (6)
    List[int],    # cur_skill sequence
    int,          # third_reward10_count
    int,          # third_reward10_three (1/0)
    int,          # collision_flag (1/0)
]


def run_episode(env, scene_id: SceneID, lib: SkillLibrary) -> EpiReturn:
    image_stack, image, ray = env.reset()

    # ---- state vars ----
    pre_skill = FALLBACK_SKILL
    cur_skill = pre_skill
    cur_id = prev_id = 0
    scene_id.reset()

    subtask_counter = 0
    raw_steps = 0
    task_idx = 0
    transition_cnt = 0

    task_steps_raw = [0] * 6
    llm_times: List[Optional[float]] = [None] * 6
    cur_skill_calls: List[int] = []

    decision_state: DecisionState = {}
    start_decision(0, decision_state)   # kick‑off for first task

    done, ep_reward = False, 0.0
    pos_x = pos_z = 0.0

    # ---- third‑task reward‑10 counting ----
    in_third_task = False
    third_reward10_count = 0

    # ---- collision tracking flag ----
    collision_flag = 0

    while not done:
        # 1) identify sub‑task by position
        cur_id = scene_id(pos_x, pos_z)

        # 2) task switch handling
        if cur_id != prev_id:

            # leaving previous task bookkeeping
            if prev_id == THIRD_TASK_ID:
                in_third_task = False

            task_steps_raw[min(task_idx, 5)] = subtask_counter
            transition_cnt += 1
            task_idx = min(task_idx + 1, 5)
            subtask_counter = 0
            prev_id = cur_id

            # async decide for new task
            start_decision(cur_id, decision_state)

            if cur_id == THIRD_TASK_ID:
                in_third_task = True
                third_reward10_count = 0

        # 3) Wait until LLM decision for *current* task is ready
        ds = decision_state.get(cur_id)
        if not (ds and "cur_skill" in ds["holder"]):
            time.sleep(WAIT_SLEEP)
            continue  # skip env.step – real waiting, no action executed

        # 4) first time we get the decision → record latency & skill
        if llm_times[cur_id] is None:
            llm_times[cur_id] = ds["holder"]["elapsed"]  # type: ignore
            cur_skill = ds["holder"]["cur_skill"]        # type: ignore
            cur_skill_calls.append(cur_skill)

        # 5) environment step using decided skill
        action = lib.act(cur_skill, image_stack, ray)
        image_stack, image, ray, reward, done, pos_x, pos_z = env.step(action)

        # 6) update counters
        if in_third_task and reward >= REWARD_VALUE:
            third_reward10_count += 1

        if done and reward < 0:
            collision_flag = 1

        ep_reward += reward
        raw_steps += 1
        subtask_counter += 1

        # ---- stuck protection ----
        if raw_steps * STEP_SCALE >= MAX_STEPS:
            done = True  # caller interprets as stuck

    # ---- episode end ----
    task_steps_raw[min(task_idx, 5)] = subtask_counter

    success_flags = [0] * 6
    for i in range(min(transition_cnt, 5)):
        success_flags[i] = 1
    if cur_id == 5:
        success_flags[5] = 1

    total_steps = raw_steps * STEP_SCALE
    task_steps = [s * STEP_SCALE for s in task_steps_raw]
    llm_times_sec = [t if t is not None else 0.0 for t in llm_times]

    return (
        ep_reward,
        success_flags,
        total_steps,
        task_steps,
        llm_times_sec,
        cur_skill_calls,
        third_reward10_count,
        int(third_reward10_count == TARGET_COUNT),
        collision_flag,
    )


# =============================================================================
# 3. Main loop with collision & stuck handling (unchanged)
# =============================================================================

def main(agent: str, env_name: str, episodes: int, out_dir: Path):
    cfg = get_config(agent)
    set_random_seed(cfg.seed)

    sceneID = SceneID()
    env = get_environment(env_name, cfg.seed, train_mode=False)
    lib = SkillLibrary(device=cfg.device)

    # --- logging setup ---
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out_dir / "run.log",
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("KEEP")

    # --- CSV setup ---
    csv_path = out_dir / "episode_log.csv"
    first_write = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if first_write:
        csv_writer.writerow([
            "episode", "runtime_sec", "steps", "reward",
            "failure_reason",
            *[f"task{i}_success" for i in range(1, 7)],
            *[f"task{i}_steps" for i in range(1, 7)],
            *[f"llm_time_task{i}" for i in range(1, 7)],
            "cur_skill_calls",
            "third_reward10_count",
            "third_reward10_three",
            "collision_flag",
        ])

    ep = 1
    while ep <= episodes:
        attempt = 1
        while True:
            t0 = time.perf_counter()
            (reward, task_flags, steps, task_steps, llm_times,
             cur_skill_seq, third_cnt, third_ok, collision_flag) = run_episode(
                env, sceneID, lib)
            runtime = time.perf_counter() - t0

            # ---- stuck? rerun same episode number ----
            if steps >= MAX_STEPS:
                logger.warning(
                    f"Ep {ep}: attempt {attempt} stuck ({steps}≥{MAX_STEPS}); rerun")
                attempt += 1
                continue
            break

        failure_reason = "collision" if collision_flag else ""

        csv_writer.writerow([
            ep, f"{runtime:.6f}", steps, f"{reward:.3f}",
            failure_reason,
            *task_flags,
            *task_steps,
            *[f"{t:.6f}" for t in llm_times],
            ",".join(map(str, cur_skill_seq)),
            third_cnt,
            third_ok,
            collision_flag,
        ])
        csv_file.flush()

        logger.info(
            f"Ep {ep:3d} | {runtime:6.2f}s | step {steps:5d} | R {reward:8.2f} | "
            f"fail {failure_reason or 'none'} | "
            + " ".join(f"T{i}:{f}" for i, f in enumerate(task_flags, 1)) + " | "
            + f"3rd‑task R10 count={third_cnt} | three? {'YES' if third_ok else 'NO'} | "
            + f"collision={collision_flag} | "
            + "LLM:" + ",".join(f"{t:.2f}s" for t in llm_times) + " | "
            + f"SK={cur_skill_seq}")
        ep += 1

    csv_file.close()
    logger.info("=== DONE ===")


# =============================================================================
# 4. CLI wrapper
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser("Keep‑Last baseline evaluator (WAIT version)")
    p.add_argument("--agent", required=True)
    p.add_argument("--env",   required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--out",   type=str, default="DILU_results")
    args = p.parse_args()
    # python DK-ABS.py --agent sac --env ugv --episodes 500
    main(args.agent, args.env, args.episodes, Path(args.out))
