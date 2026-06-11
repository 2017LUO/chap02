import argparse
from datetime import datetime
import numpy as np
import torch
import os
import pandas as pd
import logging

from utils.loader import get_config, get_environment, get_agent, get_replay_buffer
from plot.plotter import save_training_plot
from utils.seed import set_random_seed


# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger()
logger.info("Logger Initialized")


def train(agent_name, env_name):

    config = get_config(agent_name)

    set_random_seed(config.seed)

    env = get_environment(env_name, config.seed, train_mode=True)

    agent = get_agent(agent_name, config)

    buffer = get_replay_buffer(agent_name, config.buffer_capacity)

    episode_rewards = []

    # 创建保存目录
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    csv_file = os.path.join(config.save_dir, f"{agent_name}_training_rewards.csv")

    total_steps = 0

    for episode in range(config.max_episodes):
        image, ray = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if total_steps < config.warmup_steps:
                action = np.random.randn(1, config.action_dim)
                action = np.tanh(action)
            else:
                action = agent.select_action(image, ray)

            next_image, next_ray, reward, done, position_x, position_z = env.step(action)

            action = action[0]

            buffer.push(image, ray, action, reward, next_image, next_ray, done)

            episode_reward += reward
            total_steps += 4
            image = next_image
            ray = next_ray
            if buffer and len(buffer) >= config.batch_size and total_steps >= config.warmup_steps:

                batch = buffer.sample(config.batch_size)
                agent.update([t.to(config.device) for t in batch], episode)

        # TensorBoard 记录
        agent.writer.add_scalar("Reward/Episode", episode_reward, episode)

        # 日志记录与 CSV 保存
        episode_rewards.append(episode_reward)
        log_entry = f"Episode {episode + 1}, Reward: {episode_reward:.1f}, Time: {datetime.now()}"
        logger.info(log_entry)  # 使用 logging 替代 print()

        pd.DataFrame({"Episode": range(1, len(episode_rewards) + 1), "Reward": episode_rewards}).to_csv(csv_file, index=False)

        # 每100回合保存模型和奖励曲线
        if (episode + 1) % 100 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(config.save_dir, f'{agent_name}_actor_{episode + 1}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(config.save_dir, f'{agent_name}_critic_{episode + 1}.pth'))
            save_training_plot(episode_rewards, config.window_size, config.save_dir, episode + 1)

        # 提前终止条件
        if len(episode_rewards) >= config.window_size and np.mean(episode_rewards[-config.window_size:]) >= config.target_reward:
            logger.info(f"Solved! {agent_name} achieved average reward >= {config.target_reward} over last {config.window_size} episodes")  # 使用 logging 替代 print()
            break

    # 最终模型保存
    torch.save(agent.actor.state_dict(), os.path.join(config.save_dir, f'{agent_name}_actor_final.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(config.save_dir, f'{agent_name}_critic_final.pth'))

    logger.info(f"训练完成！所有数据保存在 {config.save_dir}/")  # 使用 logging 替代 print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定 agent 和环境进行训练")
    parser.add_argument("--agent", type=str, choices=["sac", "ppo", "td3"], required=True, help="选择的 agent 类型")
    parser.add_argument("--env", type=str, required=True, help="实验场景（环境）名称")
    args = parser.parse_args()

    # python Linux_train_skills.py --agent sac --env ugv
    train(args.agent, args.env)
