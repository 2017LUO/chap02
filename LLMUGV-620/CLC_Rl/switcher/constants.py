# constants.py
"""
高层三元动作 + 推理/探索超参
-------------------------------------------------
ACT_WAIT      : 等待 LLM
ACT_KEEP      : 沿用上一个技能
ACT_SELECTOR  : 始终允许执行“置信度最高的技能”，
                无论 LLM 是否已返回。
"""
# ---------- 动作枚举 ----------
ACT_WAIT = 0
ACT_KEEP = 1
ACT_SELECTOR = 2          # 外部将其映射到置信度最高技能
N_SWITCH_ACT = 3          # 网络输出维度


# ---------- ε‑greedy 探索 ----------
EPS_START = 1.0           # 初期 100 % 随机
EPS_END = 0.05          # 收敛后仍保留 5 % 随机
EPS_DECAY = 200         # 指数衰减常数（步数）

IDX2NAME = {
    ACT_WAIT:     "ACT_WAIT",
    ACT_KEEP:     "ACT_KEEP",
    ACT_SELECTOR: "ACT_SELECTOR",
}
NAME2IDX = {v: k for k, v in IDX2NAME.items()}
