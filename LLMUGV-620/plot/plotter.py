import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_training_plot(episode_rewards, window_size, save_dir, episode_number):
    """
    绘制并保存训练奖励曲线图。

    参数：
    - episode_rewards: list, 每个回合的累计奖励列表。
    - window_size: int, 用于计算平滑曲线的窗口大小。
    - save_dir: str, 图像保存的目录路径。
    - episode_number: int, 当前回合编号（用于生成文件名）。
    """
    plt.figure(figsize=(12, 6))

    # 计算滑动平均奖励，数据不足时直接使用原始奖励
    if len(episode_rewards) >= window_size:
        smoothed = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        smoothed = episode_rewards

    # 绘制平滑奖励曲线和原始奖励曲线
    sns.lineplot(x=np.arange(len(smoothed)), y=smoothed, linewidth=2, label=f"Smoothed Reward ({window_size}-episode window)")
    sns.lineplot(x=np.arange(len(episode_rewards)), y=episode_rewards, alpha=0.3, linewidth=1, label="Raw Reward")

    # 图表美化
    plt.xlabel("Episodes", fontsize=12, weight='bold')
    plt.ylabel("Average Reward", fontsize=12, weight='bold')
    plt.title("SAC Training Progress", fontsize=14, pad=20, weight='bold')
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    filename = os.path.join(save_dir, f'sac_training_plot_{episode_number}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
