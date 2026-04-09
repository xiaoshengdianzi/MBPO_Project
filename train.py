
import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from mbpo.sac import SAC
from mbpo.dynamics import EnsembleDynamicsModel, FakeEnv
from mbpo.buffer import ReplayBuffer
from mbpo.mbpo import MBPO

try:
    import gymnasium as gym
except ImportError:
    import gym

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, done, info
    next_obs, reward, done, info = out
    return next_obs, reward, done, info

def parse_args():
    parser = argparse.ArgumentParser(description="MBPO Modular Training")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    real_ratio = 0.5
    env_name = args.env_name
    env = gym.make(env_name)
    actor_lr = 5e-4
    critic_lr = 5e-3
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    target_entropy = -float(action_dim)
    model_alpha = 0.01
    rollout_batch_size = 1000
    rollout_length = 1
    model_pool_size = rollout_batch_size * rollout_length
    try:
        env.action_space.seed(args.seed)
    except Exception:
        pass
    agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    model = EnsembleDynamicsModel(state_dim, action_dim, model_alpha, device=device)
    fake_env = FakeEnv(model)
    env_pool = ReplayBuffer(buffer_size)
    model_pool = ReplayBuffer(model_pool_size)
    mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, rollout_length, rollout_batch_size, real_ratio, args.num_episodes)
    return_list = mbpo.train(reset_env, step_env)
    episodes_list = list(range(1, len(return_list) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"MBPO on {env_name}")
    plt.tight_layout()
    plt.savefig('mbpo_return.png')
    print('训练曲线已保存为 mbpo_return.png')

if __name__ == "__main__":
    main()
