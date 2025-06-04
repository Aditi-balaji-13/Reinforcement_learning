# import time
import random
import numpy as np
# from IPython.display import clear_output
from config import *
import aicrowd_gym

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""
# Hyperparameters
alpha = 0.5
gamma = 0.8
epsilon = 0.8


class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]

        if self.env_name == "acrobot":
            self.Env = aicrowd_gym.make("Acrobot-v1")
        elif self.env_name == "taxi":
            self.Env = aicrowd_gym.make("Taxi-v3")
        elif self.env_name == "kbca":
            self.Env = aicrowd_gym.make("gym_bellman:kbc-a-v0")
        elif self.env_name == "kbcb":
            self.Env = aicrowd_gym.make("gym_bellman:kbc-b-v0")
        elif self.env_name == "kbcc":
            self.Env = aicrowd_gym.make("gym_bellman:kbc-c-v0")

        self.q_table = np.zeros([500, 6])
        self.tempstate = 0
        self.tempaction = 0
        pass

    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode.
        PARAMETERS  :
            - obs - raw 'observation' from environment
        RETURNS     :
            - action - discretized 'action' from raw 'observation'
        """

        if random.uniform(0, 1) < epsilon:
            action = self.Env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(self.q_table[obs])
        self.tempstate = obs
        self.tempaction = action

        # return 1
        # raise NotImplementedError
        return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        old_value = self.q_table[self.tempstate, self.tempaction]
        next_max = np.max(self.q_table[obs])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_table[self.tempstate, self.tempaction] = new_value

        if random.uniform(0, 1) < epsilon:
            action = self.Env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(self.q_table[obs])  # Exploit learned values

        self.tempstate = obs
        self.tempaction = action
        # return 1
        # raise NotImplementedError
        return action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        # print(self.q_table)
        action = np.argmax(self.q_table[obs])
        # return 1
        # raise NotImplementedError
        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        action = np.argmax(self.q_table[obs])
        # return 1
        # raise NotImplementedError
        return action
