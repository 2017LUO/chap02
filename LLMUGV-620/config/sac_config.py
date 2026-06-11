import torch
import os

# ---------------------------
# 设备设置
# ---------------------------

seed = 42

# ---------------------------
# 设备设置
# ---------------------------
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# 训练超参数
# ---------------------------
max_episodes = 5000           # 最大回合数
batch_size = 256              # 每次采样的批量大小
warmup_steps = 1000            # 预热步数（使用随机动作）
buffer_capacity = 1000000     # 经验回放缓存容量

# ---------------------------
# SAC 算法
# ---------------------------
gamma = 0.999        # 折扣因子
tau = 0.005          # 目标网络软更新权重
alpha = 0.2          # 软 Actor-Critic 目标熵调节因子（温度参数）
actor_lr = 1e-4      # Actor 学习率
critic_lr = 3e-4     # Critic 学习率
alpha_lr = 1e-4      # 温度参数 α 的学习率
target_entropy = -2  # 目标熵，通常设置为 `-action_dim`

# 目标网络软更新间隔
update_interval = 2  # 每 2 次 Critic 更新，进行一次目标网络的软更新

# ---------------------------
# 环境和动作空间
# ---------------------------
action_dim = 2              # 动作空间维度
max_action = 1.0            # 动作取值上限
lidar_dim = 802

# ---------------------------
# 日志与模型保存相关参数
# ---------------------------
# 保存模型和图像的目录，建议使用绝对路径或基于当前工作目录
save_dir = os.path.join(os.getcwd(), 'models_and_plots')
model_dir = os.path.join(os.getcwd(), 'pth')

# 绘图时滑动平均窗口大小
window_size = 100

# 训练提前终止条件：最近 window_size 回合平均奖励达到该值时停止训练
target_reward = 20

# ---------------------------
# 其他信息（环境名称 & 算法名称）
# ---------------------------
env_name = "ugv"  # 这里会被 Linux_train_skills.py 动态替换
agent_name = "sac"  # 这里会被 Linux_train_skills.py 动态替换

