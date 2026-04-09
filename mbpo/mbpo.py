import numpy as np
from .buffer import ReplayBuffer

class MBPO:
    def __init__(self, env, agent, fake_env, env_pool, model_pool, rollout_length, rollout_batch_size, real_ratio, num_episode):
        self.env = env
        self.agent = agent
        self.fake_env = fake_env
        self.env_pool = env_pool
        self.model_pool = model_pool
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.num_episode = num_episode

    def rollout_model(self):
        observations, _, _, _, _ = self.env_pool.sample(self.rollout_batch_size)
        if len(observations) == 0:
            return
        for obs in observations:
            curr_obs = obs
            for _ in range(self.rollout_length):
                action = self.agent.take_action(curr_obs)
                reward, next_obs = self.fake_env.step(curr_obs, action)
                self.model_pool.add(curr_obs, action, reward, next_obs, False)
                curr_obs = next_obs

    def update_agent(self, policy_train_batch_size=64):
        env_batch_size = int(policy_train_batch_size * self.real_ratio)
        model_batch_size = policy_train_batch_size - env_batch_size
        for _ in range(10):
            env_obs, env_action, env_reward, env_next_obs, env_done = self.env_pool.sample(env_batch_size)
            if len(env_obs) == 0:
                return
            if self.model_pool.size() > 0:
                model_obs, model_action, model_reward, model_next_obs, model_done = self.model_pool.sample(model_batch_size)
                obs = np.concatenate((env_obs, model_obs), axis=0)
                action = np.concatenate((env_action, model_action), axis=0)
                next_obs = np.concatenate((env_next_obs, model_next_obs), axis=0)
                reward = np.concatenate((env_reward, model_reward), axis=0)
                done = np.concatenate((env_done, model_done), axis=0)
            else:
                obs, action, next_obs, reward, done = env_obs, env_action, env_next_obs, env_reward, env_done
            transition_dict = {
                "states": obs,
                "actions": action,
                "next_states": next_obs,
                "rewards": reward,
                "dones": done,
            }
            self.agent.update(transition_dict)

    def train_model(self):
        obs, action, reward, next_obs, _ = self.env_pool.return_all_samples()
        if len(obs) == 0:
            return
        inputs = np.concatenate((obs, action), axis=-1)
        reward = np.array(reward, dtype=np.float32)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), next_obs - obs), axis=-1)
        self.fake_env.model.train(inputs, labels)

    def explore_episode(self, reset_env, step_env, random_explore=False):
        obs = reset_env(self.env)
        done = False
        episode_return = 0.0
        while not done:
            if random_explore:
                action = self.env.action_space.sample()
            else:
                action = self.agent.take_action(obs)
            next_obs, reward, done, _ = step_env(self.env, action)
            self.env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self, reset_env, step_env):
        return_list = []
        explore_return = self.explore_episode(reset_env, step_env, random_explore=True)
        print(f"episode: 1, return: {int(explore_return)}")
        return_list.append(explore_return)
        for i_episode in range(self.num_episode - 1):
            obs = reset_env(self.env)
            done = False
            episode_return = 0.0
            step = 0
            while not done:
                if step % 50 == 0 and self.env_pool.size() >= 200:
                    self.train_model()
                    self.rollout_model()
                action = self.agent.take_action(obs)
                next_obs, reward, done, _ = step_env(self.env, action)
                self.env_pool.add(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_return += reward
                self.update_agent()
                step += 1
            return_list.append(episode_return)
            print(f"episode: {i_episode + 2}, return: {int(episode_return)}")
        return return_list
