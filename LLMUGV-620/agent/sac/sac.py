import torch
import torch.optim as optim
from agent.sac.sac_net import Actor, Critic
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class SAC:
    def __init__(self, config):
        """
        初始化 SAC 算法。
        config 包含以下超参数：
            - device, max_action, gamma, tau, alpha, update_interval
            - env_name, agent_name,
            - action_dim, actor_lr, critic_lr, alpha_lr,
            - lidar_dim（此处代表雷达数据的维度，例如802）
        """
        self.device = config.device
        self.max_action = config.max_action
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.update_interval = config.update_interval  # 目标网络软更新间隔

        # 初始化 TensorBoard 记录器
        self.writer = SummaryWriter(f"runs/{config.env_name}_{config.agent_name}")

        # Actor 与 Critic 网络（均融合图像和雷达数据）
        self.actor = Actor(config.action_dim, self.max_action, config.lidar_dim).to(self.device)
        self.critic = Critic(config.action_dim, config.lidar_dim).to(self.device)
        self.critic_target = Critic(config.action_dim, config.lidar_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Temperature (α) 优化
        self.target_entropy = -config.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config.alpha_lr)

        # 软更新计数
        self.update_counter = 0

    # def select_action(self, state, ray):
    #     """
    #     选择动作（用于训练）。
    #     参数：
    #         state: 图像状态，形状 (4,84,84)
    #         ray:   雷达数据，形状 (802,)
    #     """
    #     # 将数据转换为张量，并添加 batch 维度
    #     state_img = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # [1, 4, 84, 84]
    #     state_ray = torch.FloatTensor(ray).to(self.device).unsqueeze(0)      # [1, 802]
    #     action, _ = self.actor.sample(state_img, state_ray)
    #     return action.detach().cpu().numpy()

    def select_action(self, state, ray):
        """
        选择确定性动作（推理阶段使用）。
        参数：
            state: 图像状态，形状 (4,84,84) 或批量 (B,4,84,84)
            ray:   雷达数据，形状 (802,)   或批量 (B,802)
        返回：
            numpy.ndarray，形状 (action_dim,) 或 (B, action_dim)
        """
        # 1. 转张量，若是单帧则添加 batch 维度
        state_img = torch.as_tensor(state, dtype=torch.float32,
                                    device=self.device)
        state_ray = torch.as_tensor(ray, dtype=torch.float32,
                                    device=self.device)
        if state_img.dim() == 3:  # (4,84,84)
            state_img = state_img.unsqueeze(0)
            state_ray = state_ray.unsqueeze(0)

        # 2. 前向推理（均值 μ）
        with torch.no_grad():
            mu, _ = self.actor(state_img, state_ray)  # Actor.forward
            action = torch.tanh(mu) * self.max_action  # [-max_action, max_action]

        # 3. 去 batch 维度 & 回到 CPU numpy
        return action.cpu().numpy()

    def update(self, batch, episode):
        """
        更新 SAC 网络。
        batch: (states, rays, actions, rewards, next_states, next_rays, dones)
        episode: 当前训练回合（用于 TensorBoard 记录）。
        """
        state_imgs, rays, actions, rewards, next_state_imgs, next_rays, dones = batch

        # 更新 Critic 网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_state_imgs, next_rays)
            target_q1, target_q2 = self.critic_target(next_state_imgs, next_rays, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_imgs, rays, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.writer.add_scalar("Loss/Critic", critic_loss.item(), episode)
        self.writer.add_scalar("Q-value/Q1_mean", current_q1.mean().item(), episode)
        self.writer.add_scalar("Q-value/Q2_mean", current_q2.mean().item(), episode)

        # 更新 Actor 网络
        actions_pi, log_probs = self.actor.sample(state_imgs, rays)
        q1_pi, q2_pi = self.critic(state_imgs, rays, actions_pi)
        actor_loss = (self.alpha * log_probs - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
        for name, param in self.actor.named_parameters():
            self.writer.add_histogram(f"Actor/{name}", param, episode)

        # 更新 Temperature (α)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()
        self.writer.add_scalar("Temperature/Alpha", self.alpha, episode)

        # 目标网络软更新
        if self.update_counter % self.update_interval == 0:
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        self.update_counter += 1