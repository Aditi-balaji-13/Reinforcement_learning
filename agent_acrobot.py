from config import *
import time
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
        self.num_actions = 3
        self.dim = 6
        self.discrt = 4
        self.num_rbf = self.discrt * np.ones(self.dim).astype(int)
        self.width = 1. / (self.num_rbf - 1.)
        self.rbf_sigma = self.width[0] / 2.
        self.epsilon = 0.1
        self.epsilon_final = 0.1
        self.Lambda = 0.5
        self.alpha = 0.012
        self.gamma = 0.99

        self.num_ind = np.prod(self.num_rbf)
        self.activations = np.zeros(self.num_ind)
        self.new_activations = np.zeros(self.num_ind)
        self.theta = np.zeros((self.num_ind, self.num_actions))
        self.rbf_den = 2 * self.rbf_sigma ** 2
        
        self.c = np.zeros((self.num_ind, self.dim))
        for i in range(self.num_ind):
            if i == 0:
                self.pad_num = self.dim
            else:
                self.pad_num = self.dim - int(np.log(i) / np.log(self.discrt)) - 1
            self.ind = np.base_repr(i, base=self.discrt, padding=self.pad_num)
            self.ind = np.asarray([float(j) for j in list(self.ind)])
            self.c[i, :] = self.width * self.ind
        self.low = [-1,-1,-1,-1,-12.566371,-28.274334]
        self.high = [1,1,1,1,12.566371,28.274334]
        self.q = None
        self.q_new = None
        self.temp_act = 0
        self.temp_state = []*6
        self.e = np.zeros((self.num_ind, self.num_actions))
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
        #normalise
        state = []
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(state - self.c[_k, :]) ** 2 / self.rbf_den)
        self.activations = _phi
        vals = np.dot(self.theta.T, self.activations)
        
        #epsilon greedy
        if np.random.randn(0,1) <self.epsilon:
            action = random.choice([i for i in range(self.config[1])])
        else:
            action = vals.argmax()
        
        self.temp_state = state
        self.temp_act = action
        

        #return 1
        #raise NotImplementedError
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
        #normalise
        state = []
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(state - self.c[_k, :]) ** 2 / self.rbf_den)
        self.new_activations = _phi
        
        new_vals = np.dot(self.theta.T, self.new_activations)
        
        #epsilon greedy
        if np.random.randn(0,1) <self.epsilon:
            action = random.choice([i for i in range(self.config[1])])
        else:
            action = new_vals.argmax()
        
        self.q = np.dot(self.theta[:, self.temp_act], self.activations)
        self.q_new = np.dot(self.theta[:, action], self.new_activations)
        
        if done:
            self.target = reward - self.q
        else:
            self.target = reward + self.gamma * self.q_new - self.q
            
        self.e[:, self.temp_act] = self.activations
        for k in range(self.num_ind):
            for a in range(self.num_actions):
                self.theta[k, a] += self.alpha * self.target * self.e[k, a]
                
        self.e *= self.gamma * self.Lambda
        
        self.temp_state = state
        self.temp_act = action
        self.activations = self.new_activations.copy()
        #return 1
        #raise NotImplementedError
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
        #normalise
        state = []
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(state - self.c[_k, :]) ** 2 / self.rbf_den)
        self.activations = _phi
        vals = np.dot(self.theta.T, self.activations)
        
        action = vals.argmax()
            
        
        #return 1
        #raise NotImplementedError
        
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
            
        #normalise
        state = []
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        for i in range(len(obs)):
            state.append((obs[i] - self.low[i])/(self.high[i]- self.low[i]))
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(state - self.c[_k, :]) ** 2 / self.rbf_den)
        self.activations = _phi
        vals = np.dot(self.theta.T, self.activations)
        
        action = vals.argmax()

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        #return 1
        #raise NotImplementedError
        return action
