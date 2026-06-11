#!/usr/bin/env python3
"""
保留 episode_log.csv 中：
    ① task1_success … task6_success 全为 1
    ② runtime_sec ≤ 100
两条件都满足的回合，并绘制运行时间曲线（可滑动平均）。

用法示例
--------
python filter_success_and_plot_time.py
python filter_success_and_plot_time.py -i oracle_results/episode_log.csv \
                                       -o oracle_results/episode_log_success.csv \
                                       -p oracle_results/time_curve.png \
                                       -w 7 -t 120
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """带中心对齐的滑动平均."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def load_and_filter(csv_path: Path, time_threshold: float) -> pd.DataFrame:
    """
    读取 CSV，并同时满足：
        * 6 个子任务成功
        * runtime_sec ≤ time_threshold
    """
    df = pd.read_csv(csv_path)

    time_ok = pd.to_numeric(df["runtime_sec"], errors="coerce") <= time_threshold
    success_ok = (df[[f"task{i}_success" for i in range(1, 7)]] == 1).all(axis=1)

    df_filtered = df[time_ok & success_ok].reset_index(drop=True)
    return df_filtered


def plot_time(df: pd.DataFrame, window: int, png_path: Path):
    """仅对保留下来的回合绘制 runtime_sec 曲线（含滑动平均）."""
    episodes = df["episode"]
    runtime  = df["runtime_sec"].astype(float)
    runtime_sm = rolling_mean(runtime, window)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, runtime,  linewidth=0.8, alpha=0.35, label="Runtime (raw)")
    plt.plot(episodes, runtime_sm, linewidth=1.8, label=f"Runtime (MA{window})")
    plt.xlabel("Episode")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime with 6-subtask Success & ≤100 s")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[✓] Time curve saved to {png_path}")


# ----------------------------------------------------------------------
def main(args):
    df_ok = load_and_filter(Path(args.input), args.threshold)
    df_ok.to_csv(args.output, index=False)
    print(f"[✓] Filtered CSV saved to {args.output}  "
          f"({len(df_ok)} rows kept)")

    if df_ok.empty:
        print("Warning: no episodes meet the criteria—no plot created.")
    else:
        plot_time(df_ok, args.window, Path(args.plot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Keep only episodes where all 6 subtasks succeed "
                     "AND runtime_sec ≤ threshold, then plot runtime curve."))
    parser.add_argument("-i", "--input",  default="DILU_episode_log.csv",
                        help="input CSV (default: episode_log.csv)")
    parser.add_argument("-o", "--output", default="DILU_episode_log_success.csv",
                        help="filtered CSV (default: episode_log_success.csv)")
    parser.add_argument("-p", "--plot",   default="time_curve.png",
                        help="output PNG path (default: time_curve.png)")
    parser.add_argument("-w", "--window", type=int, default=5,
                        help="moving-average window (default: 5)")
    parser.add_argument("-t", "--threshold", type=float, default=100.0,
                        help="max runtime_sec to keep (default: 100)")
    main(parser.parse_args())
