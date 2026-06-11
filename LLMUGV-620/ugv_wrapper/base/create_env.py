import logging

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .unity_wrapper import UnityWrapper

"""
this functions is to create the environments

"""


def create_unity_env(env_name):
    engine_configuration_channel = EngineConfigurationChannel()
    environment_parameters_channel = EnvironmentParametersChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel,
                                          environment_parameters_channel],
                           worker_id=9999)
    env.reset()
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)

    return env


def create_multiple_unity_envs(env_name, n_envs=10, train_mode=True,base_port=5005):
    logging.basicConfig(level=logging.INFO)
    GROUP_AGGREGATION = False
    env = UnityWrapper(train_mode=train_mode,
                       env_name=env_name,
                       n_envs=n_envs,
                       group_aggregation=GROUP_AGGREGATION,
                       base_port=base_port,
                      )

    return env

