import random
import numpy as np
from config import *

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


def _softmax(x):
    return np.exp(x - np.amax(x)) / np.sum(np.exp(x - np.amax(x)))


class Agent:
    def __init__(self, env):

        if env == "acrobot":
            self.env_name = env
            self.config = config[self.env_name]

            self.actions = self.config[1]
            self.action_dim = len(self.actions)
            self.state_dim = self.config[0]

            self.w = np.random.normal(0, 1 / np.sqrt(self.state_dim), (self.action_dim, self.state_dim))

            self.train_epoch = 0
            self.episode_reward = 0
            self.grad = np.zeros_like(self.w)

            self.prob_actions = 0
            self.tempaction = 0

            self.traj_sample = 5
            self.alpha = self.config[2]

            pass

        else:
            self.env_name = env
            self.config = config[self.env_name]
            self.q_table = np.zeros([self.config[0], self.config[1]])
            self.ntcame = np.zeros([self.config[0], self.config[1]])
            self.tempstate = 0
            self.tempaction = 1

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
        if self.env_name == "kbca" or self.env_name == "kbcb" or self.env_name == "kbcc":
            khali_index = obs.index("")
            action = 1
            self.tempstate = khali_index
            self.tempaction = action
            self.ntcame[self.tempstate, self.tempaction] += 1

            return action

        if self.env_name == "taxi":

            epsilon = 0.8
            if random.uniform(0, 1) < epsilon:
                action = random.choice([i for i in range(self.config[1])])  # Explore action space
            else:
                action = np.argmax(self.q_table[obs])
            self.tempstate = obs
            self.tempaction = action

            return action

        if self.env_name == "acrobot":
            if self.train_epoch % self.traj_sample == 0:
                self.episode_reward = 0
                self.grad = np.zeros_like(self.w)

            prob_actions = _softmax(np.matmul(self.w, obs.reshape(self.state_dim, 1)))
            action = np.random.choice(self.action_dim, p=prob_actions.flatten())

            self.prob_actions = prob_actions
            self.tempaction = action

            self.train_epoch += 1

            return action

        return 0

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
        if self.env_name == "kbca" or self.env_name == "kbcb" or self.env_name == "kbcc":

            if "" not in obs:
                khali_index = len(obs)
                next_max = 0
            else:
                khali_index = obs.index("")
                next_max = np.max(self.q_table[khali_index])

            old_value = self.q_table[self.tempstate, self.tempaction]

            if obs[khali_index - 1] == -1 or obs[khali_index - 1] == 0:
                self.q_table[self.tempstate, self.tempaction] = old_value + (
                            1 / self.ntcame[self.tempstate, self.tempaction]) * (reward - old_value)

            if obs[khali_index - 1] == 1:
                self.q_table[self.tempstate, self.tempaction] = old_value + (
                            1 / self.ntcame[self.tempstate, self.tempaction]) * (reward + next_max - old_value)

            if done:
                return 0

            if self.ntcame[khali_index, 0] == 0:
                action = 0
            else:
                action = 1

            self.tempstate = khali_index
            self.tempaction = action
            self.ntcame[self.tempstate, self.tempaction] += 1

            return action

        if self.env_name == "taxi":

            alpha = 0.5
            epsilon = 0.8

            old_value = self.q_table[self.tempstate, self.tempaction]
            next_max = np.max(self.q_table[obs])

            new_value = (1 - alpha) * old_value + alpha * (reward + next_max)
            self.q_table[self.tempstate, self.tempaction] = new_value

            if random.uniform(0, 1) < epsilon:
                action = random.choice([i for i in range(self.config[1])])
            else:
                action = np.argmax(self.q_table[obs])

            self.tempstate = obs
            self.tempaction = action

            return action

        if self.env_name == "acrobot":

            if self.train_epoch % self.traj_sample != 0 and done:
                return 0

            if self.train_epoch % self.traj_sample == 0 and done:
                self.w -= self.alpha * self.grad / self.traj_sample
                return 0

            if not done:
                self.episode_reward += reward
                self.prob_actions = -self.prob_actions
                self.prob_actions[self.tempaction] += 1
                self.grad += np.matmul((self.prob_actions * reward).reshape(self.action_dim, 1),
                                       obs.reshape(1, self.state_dim))

                prob_actions = _softmax(np.matmul(self.w, obs.reshape(self.state_dim, 1)))
                action = np.random.choice(self.action_dim, p=prob_actions.flatten())

                self.prob_actions = prob_actions
                self.tempaction = action

            return action

        return 0

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode.
        PARAMETERS  :
            - obs - raw 'observation' from environment
        RETURNS     :
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == "kbca" or self.env_name == "kbcb" or self.env_name == "kbcc":
            khali_index = obs.index("")
            action = np.argmax(self.q_table[khali_index])
            return action

        if self.env_name == "taxi":
            action = np.argmax(self.q_table[obs])
            return action

        if self.env_name == "acrobot":
            prob_actions = _softmax(np.matmul(self.w, obs))
            action = np.random.choice(self.action_dim, p=prob_actions)

            return action

        return 0

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
        if self.env_name == "kbca" or self.env_name == "kbcb" or self.env_name == "kbcc":

            if done:
                return 0
            khali_index = obs.index("")
            action = np.argmax(self.q_table[khali_index])
            return action

        if self.env_name == "taxi":
            action = np.argmax(self.q_table[obs])
            return action

        if self.env_name == "acrobot":
            if done:
                return 0

            prob_actions = _softmax(np.matmul(self.w, obs))
            action = np.random.choice(self.action_dim, p=prob_actions)

            return action

        return 0
