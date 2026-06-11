import numpy as np
import cv2


class UGVEnvWrapper:
    def __init__(self, env, max_action=1.0, stack_frames=4, warmup_steps=10000, action_dim=2):
        """
        环境初始化，主要进行图像预处理和堆叠设置。
        :param env: Unity 环境实例
        :param max_action: 最大动作值
        :param stack_frames: 堆叠帧数
        :param warmup_steps: 预热步骤数
        :param action_dim: 动作空间维度
        """
        self.env = env
        self.max_action = max_action
        self.stack_frames = stack_frames
        self.warmup_steps = warmup_steps
        self.action_dim = action_dim
        self.total_steps = 0

        # 堆叠状态变量初始化
        self.frame_buffer = []
        self.stacked_image = None

        # 初始化环境的其他部分
        self.ma_obs_names, self.ma_obs_shapes, self.ma_d_action_sizes, self.ma_c_action_size = env.init()
        self.ma_names = list(self.ma_obs_shapes.keys())

        # 设置动作空间和观察空间
        self.observation_space = np.zeros((stack_frames, 84, 84), dtype=np.float32)
        self.action_space = np.zeros((action_dim,), dtype=np.float32)

    def preprocess(self, img):
        """
        对环境图像进行预处理：裁剪、灰度化、归一化。
        :param img: 原始图像
        :return: 预处理后的图像
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0  # 转换为灰度图并归一化
        return img

    def reset(self):
        """
        重置环境并初始化堆叠状态。
        :return: 堆叠状态
        """
        ma_obs_list = self.env.reset()
        state = ma_obs_list['UGVParkingAgent?team=0'][0][0][0]
        ray = ma_obs_list['UGVParkingAgent?team=0'][2][0][0]
        processed = self.preprocess(state)

        # 初始化堆叠状态
        self.frame_buffer = [processed] * self.stack_frames
        self.stacked_image = np.stack(self.frame_buffer, axis=0)

        return self.stacked_image, state, ray

    def step(self, d_action, c_action):
        """
        执行动作并与环境交互。每次执行相同的动作 4 次，并更新堆叠状态。
        :param action: 智能体的动作
        :return: 更新后的堆叠状态、奖励、done、info
        """
        total_reward = 0
        done = False
        ray = []
        position = []
        next_image = []

        # 执行动作四次（重复相同的动作 4 次）
        for _ in range(4):
            # 环境交互，获取新的状态、奖励和done标志
            ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask = self.env.step(d_action, c_action)

            next_image = ma_obs_list['UGVParkingAgent?team=0'][0][0][0]
            ray = ma_obs_list['UGVParkingAgent?team=0'][2][0][0]
            position = ma_obs_list['UGVParkingAgent?team=0'][6][0][0]
            reward = ma_reward['UGVParkingAgent?team=0'][0]
            done = ma_done['UGVParkingAgent?team=0'][0]
            total_reward += reward
            self.total_steps += 1

            if done:
                break

        position_x, position_z = position[4], position[5]
        next_image_no = next_image.copy()

        # 对新状态进行预处理并更新堆叠状态
        processed = self.preprocess(next_image)
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > self.stack_frames:
            self.frame_buffer.pop(0)

        self.stacked_image = np.stack(self.frame_buffer, axis=0)

        return self.stacked_image, next_image_no, ray, total_reward, done, position_x, position_z
