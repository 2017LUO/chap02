import torch
import torch.nn as nn


# Actor网络定义（图像与雷达数据融合）
class Actor(nn.Module):
    def __init__(self, action_dim, max_action, lidar_dim, hidden_dim=256):
        """
        参数说明：
          action_dim: 动作空间的维度
          max_action: 动作取值范围上限
          lidar_dim: 雷达数据原始输入维度（此处为802）
          hidden_dim: 雷达编码及后续全连接层的隐藏单元数（默认256）
        """
        super().__init__()
        self.max_action = max_action

        # 图像特征提取（修改后的CNN输出为256维特征）
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten(),
            nn.Linear(512, 256)  # 将原本512维映射到256维
        )
        self.cnn_out_ln = nn.LayerNorm([256])

        # 雷达数据编码（使用之前给出的MLP结构）
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 融合图像与雷达特征（256+hidden_dim=256+256=512）
        fusion_dim = 256 + hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state_img, state_lidar):
        # 图像分支
        img_feat = self.cnn_feature(state_img)
        img_feat = self.cnn_out_ln(img_feat)
        # 雷达分支
        lidar_feat = self.lidar_encode(state_lidar)
        # 融合（按特征维度拼接）
        fused = torch.cat([img_feat, lidar_feat], dim=1)
        x = self.fc(fused)
        mu = self.mu(x)
        # 限制 log_std 范围，避免梯度问题
        log_std = torch.clamp(self.log_std(x), -10, 2)
        return mu, log_std

    def sample(self, state_img, state_lidar):
        mu, log_std = self.forward(state_img, state_lidar)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # 重参数化采样
        action = torch.tanh(z) * self.max_action  # 限制动作范围
        # 计算动作的对数概率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)


# Critic网络定义（双 Q 网络，融合图像与雷达数据）
class Critic(nn.Module):
    def __init__(self, action_dim, lidar_dim, hidden_dim=256):
        """
        参数说明：
          action_dim: 动作空间维度
          lidar_dim: 雷达数据原始输入维度（802）
          hidden_dim: 雷达编码隐藏单元数，与Actor保持一致
        """
        super().__init__()
        # 图像特征提取（修改后的CNN输出为256维特征）
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten(),
            nn.Linear(512, 256)  # 将512维映射为256维
        )
        self.cnn_out_ln = nn.LayerNorm([256])

        # 雷达数据编码（沿用之前的MLP结构）
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 融合后状态特征维度（图像256 + 雷达hidden_dim=256+256=512），再与动作拼接
        fusion_dim = 256 + hidden_dim
        fc_input_dim = fusion_dim + action_dim

        # Q1 网络
        self.fc_q1 = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 网络
        self.fc_q2 = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_img, state_lidar, action):
        # 图像分支
        img_feat = self.cnn_feature(state_img)
        img_feat = self.cnn_out_ln(img_feat)
        # 雷达分支
        lidar_feat = self.lidar_encode(state_lidar)
        # 融合图像与雷达特征
        fused = torch.cat([img_feat, lidar_feat], dim=1)
        # 将状态特征与动作拼接
        sa = torch.cat([fused, action], dim=1)
        q1 = self.fc_q1(sa)
        q2 = self.fc_q2(sa)
        return q1, q2


