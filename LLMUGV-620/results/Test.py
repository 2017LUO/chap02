# plot_success_time.py
# -------------------------------------------------------------
# 双 y 轴并列柱状图：
#   左轴  Success Rate (%)         （蓝色）
#   右轴  Completion Time 40–60 s  （橙色）
# 生成 high-res PDF & PNG，适用于论文插图
# -------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ======================= 1. 数据填这里 ========================
methods = [
    "CALC-RL (Ours)",
    "ORACLE",
    "DILU",
    "KEEP-LAST",
    "SELECT-ONLY"
]
# 成功率（百分比）
success_rate = np.array([91.2, 82.0, 77.5, 35.0, 58.0])
# 成功回合平均完成时间（秒）
completion_time = np.array([46.3, 45.6, 47.8, 45.8, 45.7])
# =============================================================

# ---------- 统一排版风格 ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
})

x     = np.arange(len(methods))
width = 0.34

fig, ax_left = plt.subplots(figsize=(6, 3), dpi=300)

# ---------- 左轴：成功率 ----------
bars_left = ax_left.bar(
    x - width/2,
    success_rate,
    width,
    color="tab:blue",
    label="Success Rate (%)"
)
ax_left.set_ylabel("Success Rate (%)", color="tab:blue")
ax_left.set_ylim(0, 100)
ax_left.tick_params(axis="y", labelcolor="tab:blue")
ax_left.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

# ---------- 右轴：完成时间 ----------
ax_right = ax_left.twinx()
bars_right = ax_right.bar(
    x + width/2,
    completion_time,
    width,
    color="tab:orange",
    label="Completion Time (s)"
)

# 锁定 40–60 坐标范围
ax_right.set_ylim(40, 50)
ax_right.set_yticks([40, 45,  50])
ax_right.set_ylabel("Completion Time (s)", color="tab:orange")
ax_right.tick_params(axis="y", labelcolor="tab:orange")

# ---------- X 轴 ----------
ax_left.set_xticks(x)
ax_left.set_xticklabels(methods, rotation=15, ha="right")

# ---------- 合并图例 ----------
h_left,  l_left  = ax_left.get_legend_handles_labels()
h_right, l_right = ax_right.get_legend_handles_labels()
ax_left.legend(
    h_left + h_right,
    l_left + l_right,
    ncol=2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14)
)

plt.title("Success Rate vs. Completion Time (Moderate Latency)", pad=12)
fig.tight_layout()

# ---------- 导出高分辨率 ----------
fig.savefig("success_vs_time_locked4060.pdf", bbox_inches="tight")
fig.savefig("success_vs_time_locked4060.png", dpi=300, bbox_inches="tight")

plt.show()
print("Plot saved to success_vs_time_locked4060.{pdf,png}")
