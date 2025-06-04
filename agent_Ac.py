from config import *
import time
import random
import numpy as np

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


class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.nparts = 10
        self.q_table = np.zeros([self.nparts**6, 3])
        self.ntcame = np.zeros([self.nparts**6, 3])      # no.of times action 'a' was taken in state , i.e , no.of samples of Q(i,a)
        # self.epsilon = self.config[2]
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

        # GET ACTUAL STATE FROM obs
        low = [-1 ,-1 ,-1 ,-1 ,-12.566371 ,-28.274334] 
        high = [1 ,1 ,1 ,1 ,12.566371 ,28.274334]
        bins = [self.nparts]*6
        state = []
        
        for i, lower_upper in enumerate(zip(low, high)):
            grid_column = np.linspace(lower_upper[0] ,lower_upper[1] , bins[i]+1)[1:-1]
            ans =np.digitize(obs[i], grid_column)    
            state.append(ans)
        state = np.array(state)/self.nparts
        
        state_int = 0
        for i in state:
            state_int += i
            state_int *= self.nparts
        state_int = int(state_int)
        eps = 0.8

        if np.random.randn(0, 1) < eps:
            action = random.choice([i for i in range(self.config[1])])  # Explore action space
        else:
            action = np.argmax(self.q_table[state_int])

        self.tempstate = state_int
        self.tempaction = action
        self.ntcame[self.tempstate, self.tempaction] += 1

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
        # GET ACTUAL STATE FROM obs
        low = [-1 ,-1 ,-1 ,-1 ,-12.566371 ,-28.274334] 
        high = [1 ,1 ,1 ,1 ,12.566371 ,28.274334]
        bins = [self.nparts]*6
        state = []
        
        for i, lower_upper in enumerate(zip(low, high)):
            grid_column = np.linspace(lower_upper[0] ,lower_upper[1] , bins[i]+1)[1:-1]
            ans =np.digitize(obs[i], grid_column)    
            state.append(ans)
        state = np.array(state)/self.nparts

        state_int = 0
        for i in state:
            state_int += i
            state_int *= self.nparts
        state_int = int(state_int)

        alpha = 1 / self.ntcame[self.tempstate, self.tempaction]
        #alpha = 0.3
        eps = 0.3

        old_value = self.q_table[self.tempstate, self.tempaction]
        next_max = np.max(self.q_table[state_int])

        self.q_table[self.tempstate, self.tempaction] = (1 - alpha) * old_value + alpha * (reward + next_max)

        if random.uniform(0, 1) < eps:
            action = random.choice([i for i in range(self.config[1])])
        else:
            action = np.argmax(self.q_table[state_int])

        self.tempstate = state_int
        self.tempaction = action
        self.ntcame[self.tempstate, self.tempaction] += 1
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
        # GET ACTUAL STATE FROM obs
        low = [-1 ,-1 ,-1 ,-1 ,-12.566371 ,-28.274334] 
        high = [1 ,1 ,1 ,1 ,12.566371 ,28.274334]
        bins = [self.nparts]*6
        state = []
        
        for i, lower_upper in enumerate(zip(low, high)):
            grid_column = np.linspace(lower_upper[0] ,lower_upper[1] , bins[i]+1)[1:-1]
            ans =np.digitize(obs[i], grid_column)    
            state.append(ans)
        state = np.array(state)/self.nparts

        state_int = 0
        for i in state:
            state_int += i
            state_int *= self.nparts
        state_int = int(state_int)

        action = np.argmax(self.q_table[state_int])

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
        # GET ACTUAL STATE FROM obs
        low = [-1 ,-1 ,-1 ,-1 ,-12.566371 ,-28.274334] 
        high = [1 ,1 ,1 ,1 ,12.566371 ,28.274334]
        bins = [self.nparts]*6
        state = []
        
        for i, lower_upper in enumerate(zip(low, high)):
            grid_column = np.linspace(lower_upper[0] ,lower_upper[1] , bins[i]+1)[1:-1]
            ans =np.digitize(obs[i], grid_column)    
            state.append(ans)
        state = np.array(state)/self.nparts

        state_int = 0
        for i in state:
            state_int += i
            state_int *= self.nparts
        state_int = int(state_int)

        if done:
            return 0

        action = np.argmax(self.q_table[state_int])

        return action
