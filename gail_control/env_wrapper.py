""" Adroit expert domain configuration

The purpose of this file is to set up the Adroit environment for
the expert, which entails the creation of a no-reward version of it.

@author: Rodrigo Pereira
"""
import gym
import numpy as np
from tianshou.env import SubprocVectorEnv


class NoRewardEnv(gym.RewardWrapper):
    """
    A no-reward wrapper for the environment. 
    """
    def __init__(self, env):
        super().__init__(env)


    def reward(self, reward):
        """
        Sets the reward to 0.
        """
        return np.zeros_like(reward)


def make_env(env_id, training_num=10, test_num=5):
    """
    A function that returns vectorized environments.
    """
    env = gym.make(env_id)

    # Creating the train and test envs.
    train_envs = SubprocVectorEnv(
        [lambda: NoRewardEnv(env) for _ in range(training_num)],
        #norm_obs=True
    )

    test_envs = SubprocVectorEnv(
        [lambda: env for _ in range(test_num)],
        #norm_obs=True,
        #obs_rms=train_envs.obs_rms,
        #update_obs_rms=False
    )

    return env, train_envs, test_envs
    