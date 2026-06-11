#!/usr/bin/env python3
"""
Plot a smoothed Runtime-vs-Episode curve (Seaborn)
– small window, ASCII-only labels
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------- parameters ----------
csv_path   = Path("episode_log_success.csv")   # data file
smooth_win = 101                              # odd, < len(series)
poly_order = 2
figsize    = (6, 3)                            # SMALL window
dpi_out    = 150
fig_name   = "runtime_smoothed_small.png"

# ---------- load data ----------
df = pd.read_csv(csv_path)
episodes = df["episode"] if "episode" in df.columns else df.index + 1
runtimes = df["runtime_sec"]

# ---------- smooth ----------
y_smooth = savgol_filter(runtimes, smooth_win, poly_order, mode="interp")
mean_rt  = runtimes.mean()

# ---------- plot ----------
sns.set_theme(style="ticks", context="paper")
fig, ax = plt.subplots(figsize=figsize, dpi=dpi_out, layout="constrained")

sns.lineplot(x=episodes, y=y_smooth,
             linewidth=1.6, label="Smoothed runtime", ax=ax)

ax.axhline(mean_rt, ls="--", lw=0.8, color="gray",
           alpha=0.6, label=f"Mean ≈ {mean_rt:.2f} s")

ax.set_xlabel("Episode")
ax.set_ylabel("Runtime (s)")

ax.set_ylim(40, 60)          # <<< y-axis range fixed here

ax.grid(ls="--", lw=0.4, alpha=0.25)
ax.legend(loc="upper right", framealpha=0.9)
sns.despine(ax=ax)

fig.savefig(fig_name)
plt.show()

