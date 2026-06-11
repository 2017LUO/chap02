import importlib


def get_config(agent_name):
    """
    根据 agent 名称动态加载对应的配置模块。
    例如，agent 为 'sac' 时加载 config/sac_config.py
    """
    module_name = f"config.{agent_name}_config"
    return importlib.import_module(module_name)


def get_environment(env_name, seed, train_mode):
    """
    根据环境名称动态加载环境模块，并返回环境实例。
    例如：
    - `env_name="ugv"` → `from ugv_wrapper.wrapper_interface.ugv_wrapper import UGV`
    - `env_name="sim"` → `from unity.sim_env import SimEnv`
    """
    if env_name.lower() == "ugv":
        from ugv_wrapper.wrapper_interface.ugv_wrapper import UGV
        return UGV(seed=seed, train_mode=train_mode)

    else:
        module_name = f"unity.{env_name}_env"
        env_module = importlib.import_module(module_name)

        env_class_name = ''.join(word.capitalize() for word in env_name.split('_'))
        env_class = getattr(env_module, env_class_name, None)

        if env_class is None:
            raise ImportError(f"环境模块 {module_name} 中未找到类 {env_class_name}")

        return env_class()


def get_agent(agent_name, config):
    """
    根据 agent 名称动态加载对应的 agent 模块，并返回 agent 实例。
    """
    agent_module = importlib.import_module(f"agent.{agent_name}.{agent_name}")
    return agent_module.SAC(config)


def get_replay_buffer(agent_name, capacity):
    """
    根据 agent 名称返回对应的经验回放缓存（对于 PPO 可能不需要）。
    """
    if agent_name in ["sac", "td3"]:
        buffer_module = importlib.import_module(f"agent.{agent_name}.{agent_name}_memory")
        return buffer_module.ReplayBuffer(capacity)
    return None
