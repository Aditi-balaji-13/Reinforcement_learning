import numpy as np
from config import *
import random
import math


def create_tiling(feat_range, bins, offset):

    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def create_tilings(feature_ranges, number_tilings, bins, offsets):

    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)


# feature_ranges = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-12.566371, 12.566371], [-28.274334, 28.274334]]  # 6 features
# number_tilings = 3
# bins = [[5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5]]  # each tiling has a 10*10 grid
# offsets = [[0.2, 0.2, 0.2, 0.2, 2, 4], [-0.2, -0.2, -0.2, -0.2, -2, -4], [0, 0, 0, 0, 0, 0]]

# tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

# print(tilings.shape)  # # of tilings X features X bins


def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)


# feature = [1, 0, 1, 0, 0, 0]

# coding = get_tile_coding(feature, tilings)
# print(coding)

# array([[5, 1],
#       [4, 0],
#       [3, 0]])


class QValueFunction:

    def __init__(self, tilings, actions, lr):
        self.tilings = tilings
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr  # / self.num_tilings  # learning rate equally assigned to each tiling
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]  # [(10, 10), (10, 10), (10, 10)]
        self.q_tables = [np.zeros(shape=(state_size + (len(self.actions),))) for state_size in self.state_sizes]

    def value(self, state, action):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        value = 0
        for coding, q_table in zip(state_codings, self.q_tables):
            value += q_table[tuple(coding) + (action_idx,)]
        return value / self.num_tilings

    def update(self, state, action, target):
        state_codings = get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = target - q_table[tuple(coding) + (action_idx,)]
            q_table[tuple(coding) + (action_idx,)] += self.lr * delta


# QTables = QValueFunction(tilings, [-1, 0, 1], 0.1)
# print(QTables.q_tables)


class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]

        self.bins = [[6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 6]]
        self.ntiles = 3
        self.feature_ranges = self.config[2]
        self.offsets = self.config[3]
        self.tilings = create_tilings(self.feature_ranges, self.ntiles, self.bins, self.offsets)
        self.actions = self.config[1]

        self.QTables = QValueFunction(self.tilings, self.actions, 0.05)
        # self.NTcame = QValueFunction(self.tilings, self.actions, 0.1)  # no.of times action 'a' was taken in state , i.e , no.of samples of Q(i,a)
        # self.epsilon = self.config[2]
        self.tempstate = 0
        self.tempaction = 0
        self.train_epoch = 0

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

        eps = 0.8
        # eps = max(0.1, min(1.0, 1.0 - math.log10((n + 1) / 25)))

        if random.uniform(0, 1) < eps:
            action = random.choice(self.actions)  # Explore action space
        else:
            action = np.argmax([self.QTables.value(obs, -1), self.QTables.value(obs, 0), self.QTables.value(obs, 1)]) - 1

        self.tempstate = obs
        self.tempaction = action
        # self.NTcame.update(self.tempstate, self.tempaction, 1)
        self.train_epoch += 1

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

        #eps = max(0.1, min(1.0, 1.0 - math.log10((self.train_epoch + 1) / 25)))
        eps = 0.2
        target = reward + np.max([self.QTables.value(obs, -1), self.QTables.value(obs, 0), self.QTables.value(obs, 1)])
        self.QTables.update(self.tempstate, self.tempaction, target)

        if random.uniform(0, 1) < eps:
            action = random.choice(self.actions)
        else:
            action = np.argmax([self.QTables.value(obs, -1), self.QTables.value(obs, 0), self.QTables.value(obs, 1)]) - 1

        # if self.train_epoch <= 50:
            # action = -1 if obs[4] > 0 else 1
            # if (obs[1] > 0 and obs[4] > 0) or (obs[1]  0 and obs[4] > 0)

        self.tempstate = obs
        self.tempaction = action
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

        action = np.argmax([self.QTables.value(obs, -1), self.QTables.value(obs, 0), self.QTables.value(obs, 1)]) - 1

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

        if done:
            return 0

        action = np.argmax([self.QTables.value(obs, -1), self.QTables.value(obs, 0), self.QTables.value(obs, 1)]) - 1

        return action
