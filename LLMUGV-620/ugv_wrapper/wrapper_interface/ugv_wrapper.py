import logging

import numpy as np
from ugv_wrapper.base.unity_wrapper import UnityWrapper
from ugv_wrapper.stack.stack_wrapper import UGVEnvWrapper


class UGV:
    def __init__(self,
                 seed=42,
                 train_mode=True,
                 env_name=r'E:\UGVDEMO\UGVParking\RLEnvironments.exe',
                 # env_name=r'F:\UGV-CRNP\RLEnvironments.exe',
                 # env_name=r'/mnt/LLMUGV-620/unity/LINUX/Combine/Combine.x86_64',
                 n_envs=1,
                 group_aggregation=True,
                 scene='scene_1'):
        """
        初始化无人车环境
        """
        logging.basicConfig(level=logging.INFO)

        # 初始化 Unity 环境及其包装器
        self.unity_env = UnityWrapper(train_mode=train_mode,
                                      env_name=env_name,
                                      n_envs=n_envs,
                                      seed=seed,
                                      group_aggregation=group_aggregation,
                                      scene=scene)

        # 初始化后会返回观测名称、形状以及动作空间大小信息
        self.ma_obs_names, self.ma_obs_shapes, self.ma_d_action_sizes, self.ma_c_action_size = self.unity_env.init()

        self.ma_names = list(self.ma_obs_shapes.keys())

        # 使用 UGV 环境包装器进行进一步封装
        self.env = UGVEnvWrapper(self.unity_env)
        self.n_envs = n_envs

    def reset(self):
        """
        重置环境，并返回初始状态
        """
        return self.env.reset()

    def step(self, c_action):

        ma_d_action = {}
        ma_c_action = {}

        for n in self.ma_names:
            d_action, c_action = None, c_action
            if self.ma_d_action_sizes[n]:
                d_action_sizes = self.ma_d_action_sizes[n]
                d_action_list = [np.random.randint(0, d_action_size, size=self.n_envs)
                                 for d_action_size in d_action_sizes]
                d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                                 for d_action, d_action_size in zip(d_action_list, d_action_sizes)]
                d_action = np.concatenate(d_action_list, axis=-1)

        ma_d_action[n] = d_action
        ma_c_action[n] = c_action

        return self.env.step(ma_d_action, ma_c_action)

    def close(self):
        if self.unity_env is not None:
            self.unity_env.close()
