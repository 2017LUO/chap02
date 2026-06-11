import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """ Experience Replay Buffer with radar data """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, image, radar, action, reward, next_image, next_radar, done):
        # 将状态和雷达数据一起存储
        self.buffer.append((
            np.array(image, dtype=np.float32),  # 存储状态
            np.array(radar, dtype=np.float32),  # 存储雷达数据
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_image, dtype=np.float32),
            np.array(next_radar, dtype=np.float32),  # 存储下一状态的雷达数据
            np.array(done, dtype=np.float32)
        ))

    def sample(self, batch_size):
        images, radars, actions, rewards, next_images, next_radars, dones = zip(*random.sample(self.buffer, batch_size))

        # 返回包含状态、雷达、动作、奖励、下一状态、下一雷达数据和done标志的批次
        return (
            torch.FloatTensor(np.array(images)),
            torch.FloatTensor(np.array(radars)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_images)),
            torch.FloatTensor(np.array(next_radars)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
