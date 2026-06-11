#!/usr/bin/env python3
"""
Plot a smoothed Runtime-vs-Episode comparison curve (Seaborn).

* Blue  — original episode_log_success.csv
* Orange — episode_log_success_offset.csv (runtime_sec + 3–6 s)

运行示例
--------
python plot_compare.py
python plot_compare.py -o runtime_compare.png --win 101 --ylow 40 --yhigh 60
"""

from pathlib import Path
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------
def smooth(series, win, order):
    """Savitzky-Golay smoothing (window 必须为奇数)."""
    if win % 2 == 0:
        win += 1  # 保证奇数
    win = min(win, len(series) // 2 * 2 + 1)  # 不超过长度
    return savgol_filter(series, win, order, mode="interp")


def load_runtime(csv_path):
    """返回 episode 序列和 runtime_sec 序列（float)."""
    df = pd.read_csv(csv_path)
    episodes = df["episode"] if "episode" in df.columns else df.index + 1
    runtimes = pd.to_numeric(df["runtime_sec"], errors="coerce")
    return episodes, runtimes


# ---------------------------------------------------------------------
def main(args):
    csv_orig = Path("episode_log_success.csv")
    csv_offset = Path("DILU_episode_log_success.csv")

    # ---------- 数据 ----------
    ep1, rt1 = load_runtime(csv_orig)
    ep2, rt2 = load_runtime(csv_offset)

    y1 = smooth(rt1, args.win, 2)
    y2 = smooth(rt2, args.win, 2)

    mean1, mean2 = rt1.mean(), rt2.mean()

    # ---------- 绘图 ----------
    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150, layout="constrained")

    sns.lineplot(x=ep1, y=y1, lw=1.6, label="Smoothed (orig)",  ax=ax)
    sns.lineplot(x=ep2, y=y2, lw=1.6, label="Smoothed (+DILU)", ax=ax)

    ax.axhline(mean1, ls="--", lw=0.8, color="gray",
               alpha=0.55, label=f"Mean orig ≈ {mean1:.2f}s")
    ax.axhline(mean2, ls=":", lw=0.8, color="gray",
               alpha=0.55, label=f"Mean DILU ≈ {mean2:.2f}s")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Runtime (s)")
    ax.set_ylim(args.ylow, args.yhigh)

    ax.grid(ls="--", lw=0.4, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    sns.despine(ax=ax)

    fig.savefig(args.out, dpi=150)
    plt.show()
    print(f"[✓] Plot saved to {args.out}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare smoothed runtime curves for original vs. +DILU CSVs.")
    parser.add_argument("-o", "--out", default="runtime_compare.png",
                        help="output PNG path (default: runtime_compare.png)")
    parser.add_argument("--win", type=int, default=51,
                        help="Savitzky-Golay window size (odd, default: 101)")
    parser.add_argument("--ylow",  type=float, default=42.0,
                        help="y-axis lower limit (default: 40)")
    parser.add_argument("--yhigh", type=float, default=50.0,
                        help="y-axis upper limit (default: 60)")
    main(parser.parse_args())
