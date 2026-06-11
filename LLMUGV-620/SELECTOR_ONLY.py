#!/usr/bin/env python3
# EFSR.py
# ======================================================================

import argparse, csv, logging, time, threading
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as tmp
from joblib._multiprocessing_helpers import mp

from CLC_Rl.selector.online_trainer import OnlineTrainer
from CLC_Rl.selector.selector import Selector
from CLC_Rl.skills_manager.skill_manager import SkillLibrary
from CLC_Rl.switcher.constants import (ACT_KEEP, ACT_SELECTOR, ACT_WAIT,
                                       IDX2NAME, NAME2IDX)
from CLC_Rl.switcher.online_switcher import OnlineSwitcher
from utils.loader import get_config, get_environment
from utils.scene_id import SceneID
from utils.seed import set_random_seed

# ---------- LLM decide ----------
# from prompt.LLM_Agent import decide
from prompt.Test import decide

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
INIT_LLM_SKILL = 0
STEP_SCALE = 4
SWITCH_PENALTY = {ACT_WAIT: -10.0, ACT_KEEP: 0.0, ACT_SELECTOR: -5.0}


# ----------------------------------------------------------------------
# 子进程训练器
# ----------------------------------------------------------------------
def _start_selector_worker(sample_q, ckpt):
    trainer = OnlineTrainer(sample_q, checkpoint_path=ckpt,
                            device='cuda', log_file='selector_train.log',
                            tb_dir='runs/selector')
    trainer.run()


def _start_lcm_worker(sample_q, ckpt):
    from CLC_Rl.switcher.trainer import LCMTrainer
    trainer = LCMTrainer(sample_q, ckpt)
    trainer.run()


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def np84_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """(84,84,3) float32[0–1] → (3,84,84) float32"""
    return torch.as_tensor(img_np, dtype=torch.float32).permute(2, 0, 1)


def _first_skill(pred: Any) -> Any:
    return max(pred.items(), key=lambda kv: kv[1])[0] if isinstance(pred, dict) else pred[0]


def _log_pred_from_imgs(logger: logging.Logger, tag: str, scene: int,
                        img_buf: deque, candidate: List[int],
                        selector: Selector) -> int:
    preds = [selector.predict(img, candidate) for img in img_buf]
    winner, _ = Counter([_first_skill(p) for p in preds]).most_common(1)[0]
    return winner


# ----------------------------------------------------------------------
# 单回合
# ----------------------------------------------------------------------
def run_episode(env, scene_id_fn: SceneID, lib: SkillLibrary,
                q_sel: mp.Queue, q_lcm: mp.Queue,
                selector: Selector, switcher: OnlineSwitcher,
                logger: logging.Logger) -> Tuple[
        float, List[int], int, List[int], List[float], List[int],
        List[int], List[str], List[Optional[int]], List[List[int]], int]:

    # ---------- Reset ----------
    image_stack, image, ray = env.reset()
    pos_x = pos_z = 0.0
    scene_id_fn.reset()
    tensor_img = np84_to_tensor(image)
    img_buf: deque = deque([tensor_img] * 3, maxlen=3)

    start_img = tensor_img.clone()
    start_lidar = torch.as_tensor(ray) if ray is not None else torch.empty(0)

    ep_reward = 0.0
    subtask_reward = 0.0

    cur_id = 0
    task_idx = transition_cnt = subtask_counter = 0
    cur_llm_skill = last_llm_skill = INIT_LLM_SKILL
    candidate_skill: List[int] = [0, 1, 2, 3]
    task_steps_raw = [0] * 6
    llm_times: List[Optional[float]] = [None] * 6
    activate_seq: List[int] = []

    winners: List[int] = []
    switch_skills: List[str] = []
    llm_skills: List[Optional[int]] = []
    llm_candidates: List[List[int]] = []

    # --- NEW: 碰撞状态 ---
    collision_occurred = False                          # --- NEW
    collision_task_idx: Optional[int] = None            # --- NEW

    # ---------- LLM 线程管理 ----------
    decision_state: Dict[int, Dict] = {}

    def _decide_async(tid: int, holder: dict):
        holder['four_skills'], holder['llm_skill'] = decide(tid)

    def start_decision(tid: int):
        h: Dict[str, Any] = {}
        th = threading.Thread(target=_decide_async, args=(tid, h))
        th.start()
        decision_state[tid] = {'thread': th, 'holder': h,
                               'start': time.perf_counter(), 'finished': False}

    # ---------- 场景 0 ----------
    selector_skill = _log_pred_from_imgs(logger, 'init', 0,
                                         img_buf, candidate_skill, selector)
    winners.append(selector_skill)

    act_name = switcher.select(start_img, start_lidar)
    switch_skills.append(act_name)
    switch_act_int = NAME2IDX[act_name]
    logger.info('Switcher scene=%d action=%s selector_skill=%d last_llm_skill=%d', 0, act_name, selector_skill, last_llm_skill)

    start_decision(0)
    pending_img = tensor_img.clone()

    if act_name == IDX2NAME[ACT_WAIT]:
        ds = decision_state[0]
        ds['thread'].join()
        ds['finished'] = True
        llm_times[0] = time.perf_counter() - ds['start']
        cur_llm_skill = ds['holder']['llm_skill']
        candidate_skill = ds['holder']['four_skills']
        activate_skill = cur_llm_skill
        q_sel.put((tensor_img, int(cur_llm_skill)))
        llm_skills.append(cur_llm_skill)
        llm_candidates.append(candidate_skill.copy())
        pending_img = None
        logger.info('LLMResult   scene=%d llm=%s time=%.2f cand=%s',
                    cur_id, cur_llm_skill, llm_times[cur_id], candidate_skill)
    elif act_name == IDX2NAME[ACT_KEEP]:
        activate_skill = last_llm_skill
        llm_skills.append(None)
        llm_candidates.append([])
    else:
        activate_skill = selector_skill
        llm_skills.append(None)
        llm_candidates.append([])

    raw_steps = 0
    img_buf.clear()
    img_buf.append(tensor_img)

    # ==================================================================
    # 主循环
    # ==================================================================
    while True:
        # ------- 环境一步 -------
        action = lib.act(activate_skill, image_stack, ray)
        image_stack, image, ray, reward, done, pos_x, pos_z = env.step(action)

        tensor_img = np84_to_tensor(image)
        img_buf.append(tensor_img)

        subtask_reward += reward
        ep_reward += reward
        raw_steps += 1
        subtask_counter += 1
        activate_seq.append(activate_skill)

        # --- NEW: 碰撞检测 ---
        if not collision_occurred and done and reward < 0:
            collision_occurred = True
            collision_task_idx = task_idx
            logger.warning('Collision detected at task %d, step %d', task_idx, raw_steps)  # --- NEW

        # ------- done：写经验并退出 -------
        if done:
            if q_lcm is not None:
                end_lidar = torch.as_tensor(ray) if ray is not None else torch.empty(0)
                total_rew = subtask_reward + SWITCH_PENALTY[switch_act_int]
                q_lcm.put((start_img, start_lidar, switch_act_int,
                           total_rew, tensor_img, end_lidar, True))
            break

        # ------- 子任务切换 -------
        new_id = scene_id_fn(pos_x, pos_z)

        if new_id != cur_id:
            last_llm_skill = cur_llm_skill
            # 为上一子任务写经验
            if q_lcm is not None:
                end_lidar = torch.as_tensor(ray) if ray is not None else torch.empty(0)
                total_rew = subtask_reward + SWITCH_PENALTY[switch_act_int]
                q_lcm.put((start_img, start_lidar, switch_act_int,
                           total_rew, tensor_img, end_lidar, False))

            task_steps_raw[min(task_idx, 5)] = subtask_counter
            transition_cnt += 1
            task_idx = min(task_idx + 1, 5)
            subtask_counter = 0
            subtask_reward = 0.0

            # 进入新子任务
            start_img = tensor_img.clone()
            start_lidar = torch.as_tensor(ray) if ray is not None else torch.empty(0)
            cur_id = new_id

            selector_skill = _log_pred_from_imgs(logger, 'switch', cur_id,
                                                 img_buf, candidate_skill, selector)
            winners.append(selector_skill)
            img_buf.clear()
            img_buf.append(tensor_img)

            if cur_id not in decision_state:
                start_decision(cur_id)
            ds = decision_state[cur_id]

            act_name = switcher.select(start_img, start_lidar)
            switch_skills.append(act_name)
            switch_act_int = NAME2IDX[act_name]
            logger.info('Switcher scene=%d action=%s selector_skill=%d last_llm_skill=%d', cur_id, act_name, selector_skill, last_llm_skill)

            if act_name == IDX2NAME[ACT_WAIT]:
                ds['thread'].join()
                ds['finished'] = True
                llm_times[cur_id] = time.perf_counter() - ds['start']
                cur_llm_skill = ds['holder']['llm_skill']
                candidate_skill = ds['holder']['four_skills']
                activate_skill = cur_llm_skill
                q_sel.put((tensor_img, int(cur_llm_skill)))
                llm_skills.append(cur_llm_skill)
                llm_candidates.append(candidate_skill.copy())
                pending_img = None
                logger.info('LLMResult   scene=%d llm=%s time=%.2f cand=%s',
                            cur_id, cur_llm_skill, llm_times[cur_id], candidate_skill)
            elif act_name == IDX2NAME[ACT_KEEP]:
                activate_skill = last_llm_skill
                llm_skills.append(None)
                llm_candidates.append([])
                pending_img = tensor_img.clone()
            else:
                activate_skill = selector_skill
                llm_skills.append(None)
                llm_candidates.append([])
                pending_img = tensor_img.clone()

        # ------- 检查异步 LLM 完成 -------
        ds = decision_state.get(cur_id)
        if ds and not ds['finished'] and not ds['thread'].is_alive():
            llm_times[cur_id] = time.perf_counter() - ds['start']
            cur_llm_skill = ds['holder']['llm_skill']
            candidate_skill = ds['holder']['four_skills']
            activate_skill = cur_llm_skill
            ds['finished'] = True
            q_sel.put((pending_img, int(cur_llm_skill)))
            llm_skills[cur_id] = cur_llm_skill
            llm_candidates[cur_id] = candidate_skill.copy()
            pending_img = None
            logger.info('LLMResult   scene=%d llm=%s time=%.2f cand=%s',
                        cur_id, cur_llm_skill, llm_times[cur_id], candidate_skill)

    # ---------- 收尾 ----------
    task_steps_raw[min(task_idx, 5)] = subtask_counter

    # --- NEW: success_flags 计算 ---
    success_flags = [0] * 6
    if collision_occurred:
        # 只有在碰撞前 完整退出过的子任务算成功
        for i in range(collision_task_idx or 0):
            success_flags[i] = 1
    else:
        for i in range(transition_cnt):
            success_flags[i] = 1
        if cur_id == 5:
            success_flags[5] = 1
    # --------------------------------

    total_steps = raw_steps * STEP_SCALE
    task_steps = [s * STEP_SCALE for s in task_steps_raw]
    llm_times_sec = [t if t is not None else 0.0 for t in llm_times]

    for ds in decision_state.values():
        ds['thread'].join()

    collision_flag = int(collision_occurred)

    logger.info('RoundSummary succ_rate=%.2f collision=%d winners=%s switch=%s '
                'llm=%s llm_t=%s cand=%s',
                sum(success_flags) / 6, collision_flag,
                winners, switch_skills, llm_skills, llm_times_sec, llm_candidates)

    return (ep_reward, success_flags, total_steps, task_steps,
            llm_times_sec, activate_seq,
            winners, switch_skills, llm_skills, llm_candidates, collision_flag)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(agent: str, env_name: str, episodes: int, out_dir: Path):
    cfg = get_config(agent)
    set_random_seed(cfg.seed)
    env = get_environment(env_name, cfg.seed, train_mode=False)
    sceneID = SceneID()
    lib = SkillLibrary(device=cfg.device)

    # ---------- 日志 ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'run.log'
    logging.basicConfig(filename=str(log_file),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger('LCM')

    # 在线训练
    q_sel = mp.Queue(maxsize=50_000)
    tmp.spawn(_start_selector_worker, args=(q_sel, 'checkpoints/selector_best.pth'),
              nprocs=1, join=False)
    selector = Selector('checkpoints/selector_best.pth', device='cuda')

    q_lcm = mp.Queue(maxsize=50_000)
    tmp.spawn(_start_lcm_worker, args=(q_lcm, 'checkpoints/switcher_best.pth'),
              nprocs=1, join=False)
    switcher = OnlineSwitcher('checkpoints/switcher_best.pth', device='cuda')

    # ---------- CSV ----------
    csv_path = out_dir / 'log.csv'
    first = not csv_path.exists()
    csv_f = csv_path.open('a', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    if first:
        writer.writerow(['ep', 'time', 'step', 'reward',
                         *[f'T{i}_succ' for i in range(1, 7)],
                         *[f'T{i}_step' for i in range(1, 7)],
                         *[f'LLM{i}' for i in range(1, 7)],
                         'seq',
                         'winner_seq', 'switch_seq', 'llm_seq',
                         'cand_seq', 'succ_rate', 'collision'])

    # ---------- Episode Loop ----------
    for ep in range(1, episodes + 1):
        t0 = time.perf_counter()
        (reward, flags, steps, tsteps, llm_t, seq,
         winners, switch_skills, llm_skills, llm_cands, collision) = run_episode(
            env, sceneID, lib, q_sel, q_lcm, selector, switcher, logger)
        used = time.perf_counter() - t0
        succ_rate = sum(flags) / 6

        writer.writerow([ep, f'{used:.2f}', steps, f'{reward:.1f}',
                         *flags, *tsteps, *[f'{x:.2f}' for x in llm_t],
                         ','.join(map(str, seq)),
                         ','.join(map(str, winners)),
                         ','.join(map(str, switch_skills)),
                         ','.join('' if s is None else str(s) for s in llm_skills),
                         ';'.join(','.join(map(str, c)) for c in llm_cands),
                         f'{succ_rate:.2f}', collision])
        csv_f.flush()

        logger.info('EP%03d summary: time=%.2fs reward=%.1f succ_rate=%.2f collision=%d',
                    ep, used, reward, succ_rate, collision)

    csv_f.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == '__main__':
    tmp.set_start_method('spawn', force=True)
    ap = argparse.ArgumentParser('LCM with Selector & Switcher (float32 images)')
    ap.add_argument('--agent', required=True)
    ap.add_argument('--env', required=True)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--out', type=str, default='SELECTOR_ONLY_results')
    args = ap.parse_args()
    main(args.agent, args.env, args.episodes, Path(args.out))
