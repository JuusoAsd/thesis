"""
Tests if PPO can replicate a non-linear policy even with a smaller network
It can
"""

import gym
from gym import spaces
import numpy as np
import dotenv

# Imitation learning
import stable_baselines3 as sb3
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pandas as pd

dotenv.load_dotenv()


# Create the custom Gym environment
class SquaredFunctionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

    def step(self, action):
        observation = self.observation_space.sample()
        reward = -np.square(observation[0] - action[0])
        done = True
        return observation, reward, done, {}

    def reset(self):
        return self.observation_space.sample()


# Create the expert policy
def expert_policy(observation):
    return np.array([observation[0] ** 2])


def get_expert():
    return expert_policy


if __name__ == "__main__":
    # Generate expert demonstrations
    venv = DummyVecEnv([SquaredFunctionEnv])
    rng = np.random.default_rng(0)

    observations = []
    actions = []

    # Initialize model
    model = sb3.PPO("MlpPolicy", venv, verbose=1, policy_kwargs={"net_arch": [64, 64]})
    student_policy = model.policy
    rollouts = rollout.rollout(
        get_expert(),
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=1000),
        unwrap=False,
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    learner = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=student_policy,
        rng=rng,
    )

    # Train model
    learner.train(n_epochs=200)

    # Test model
    observation = venv.reset()

    data_dict = {
        "expert": [],
        "model": [],
        "observation": [],
    }
    # loop from -1 to 1 (inclusive) with 0.01 step
    for i in np.arange(-1, 1.01, 0.01):
        observation = np.array([i])
        action, _ = model.predict(observation, deterministic=True)
        expert_action = expert_policy(observation)
        # print(i, action, expert_action)

        data_dict["expert"].append(expert_action[0])
        data_dict["model"].append(action[0])
        data_dict["observation"].append(observation[0])

    df = pd.DataFrame(data_dict)
    df.to_csv(
        os.path.join(os.getenv("RESULT_PATH"), "random", "non_linear.csv"), index=False
    )
