from argparse import ArgumentParser, Namespace

import yaml

import gymnasium as gym

from stable_baselines3 import PPO


def train_sb3(
        gym_env_id: str,
        gym_seed: int,
        sb3_cfg: dict,
        sb3_timesteps: int,
        sb3_ppo_weights: str,
        **kwargs
) -> None:
    """Trains SB3-PPO policy on a given gym environment."""
    env = gym.make(gym_env_id)
    env.reset(seed=gym_seed)
    model = PPO(env=env, **sb3_cfg)
    model.learn(total_timesteps=sb3_timesteps)
    model.save(sb3_ppo_weights)
    env.close()


def train_medium_expert_sb3(cfg: dict) -> None:
    train_timesteps = [cfg["sb3_medium_timesteps"], cfg["sb3_expert_timesteps"]] 
    weights = [cfg["sb3_ppo_medium_weights"], cfg["sb3_ppo_expert_weights"]]
    for steps, weight in zip(train_timesteps, weights): 
        train_sb3(sb3_timesteps=steps, sb3_ppo_weights=weight, sb3_cfg=cfg["sb3"], **cfg) 


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
    train_medium_expert_sb3(cfg)