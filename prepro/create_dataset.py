import os
from argparse import ArgumentParser, Namespace

import yaml
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO


def create_dataset(
        root_data: str,
        gym_env_id: str,
        gym_seed: int,
        sb3_ppo_weights: str,
        total_episodes: int,
        disc_act_space: bool,
        verbose: bool=True,
        **kwargs
) -> None:
    """
    Creates a very simple offline-RL dataset.
    The dataset is saved as .npz
    """
    assert os.path.isfile(sb3_ppo_weights), f"File {sb3_ppo_weights} does not exist."
    model = PPO.load(sb3_ppo_weights)
    
    env = gym.make(gym_env_id)

    dataset = []
    episode_rewards = np.zeros(total_episodes, dtype=np.float32) 
    for ep in range(total_episodes):
        state, _ = env.reset(seed=gym_seed + ep)

        states = [state]
        actions = []
        rewards = []
        terminated = []
        truncated = []

        while True:
            action, _ = model.predict(state, deterministic=True) 
            state, reward, term, trunc, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)

            if term or trunc:
                break

        if disc_act_space:
            actions = np.asarray(actions, dtype=np.int64)
        else:
            actions = np.asarray(actions, dtype=np.float32)

        states = np.asarray(states, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.dtype(bool))
        truncated = np.asarray(truncated, dtype=np.dtype(bool))

        episode = {
            "states": states, 
            "actions": actions, 
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated 
        }
        dataset.append(episode)
        episode_rewards[ep] = np.sum(rewards) 

    if verbose:
        mean_reward = np.mean(episode_rewards)
        print(f"Created dataset with mean episode reward of: {mean_reward}")

    np.savez_compressed(root_data, episodes=np.array(dataset, dtype=object))


def create_medium_expert_datasets(cfg: dict) -> None:
    dataset_names = [cfg["root_medium_dataset"], cfg["root_expert_dataset"]]  
    sb3_ppo_weights = [cfg["sb3_ppo_medium_weights"], cfg["sb3_ppo_expert_weights"]]
    num_episodes = [cfg["total_episodes_medium"], cfg["total_episodes_expert"]] 

    for dataset_name, num_eps, ppo_weights in zip(dataset_names, num_episodes, sb3_ppo_weights):
        create_dataset(root_data=dataset_name, sb3_ppo_weights=ppo_weights, total_episodes=num_eps, **cfg)


def load_config(config_yaml: str) -> dict:
    with open(config_yaml, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    create_medium_expert_datasets(cfg)