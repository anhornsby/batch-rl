# coding: utf-8
"""
Batch Reinforcement Learning agents
Adam Hornsby
"""

import numpy as np


class RandomAgent(object):
    """Randomly deciding action based on coin flip"""

    def __init__(self, action_space=list()):
        super(RandomAgent, self).__init__()
        self.action_space = action_space

    def determine_action(self, *kwargs):
        """Take state and make action"""

        action = np.random.choice(self.action_space)

        return action


class SingleFittedQAgent(object):
    """
    Make decisions by greedily selecting actions with the highest Q values

    Appropriate for a Fitted Q Iteration agent that has actions
    one-hot encoded as features (as in the original FQI paper)

    """

    def __init__(self, action_space, clfs):
        super(SingleFittedQAgent, self).__init__()
        self.action_space = action_space
        self.clfs = clfs

        # assert that there are multiple agents
        if isinstance(clfs, list):
            assert (len(action_space) == len(clfs))

    def _predict(self, state):
        """Make predictions from a single model"""

        Q_next_preds = list()
        for a in self.action_space:
            # make predictions for each different action type
            test_X_action = np.zeros(len(self.action_space))
            test_X_action[a] = 1.  # set 1s everywhere for action a

            # concatenate and predict
            test_X = np.column_stack([state, test_X_action.reshape(1, -1)])
            Q_next_preds.append(self.clfs.predict(test_X))

        Q_max = np.column_stack(Q_next_preds)

        return Q_max

    def determine_action(self, state):
        """Greedily determine the best action to be taken"""

        # use the two models to make predictions about the Q values
        Q_vals = self._predict(state)

        # greedily select the argmax action
        Q_arg_max = np.argmax(Q_vals, axis=1)

        return Q_arg_max[0]


def max_per_multiple_action(estimators, state):
    """
    For an agent with multiple models per action,
    calculate the max over all arms

    :return:
    """

    # calculate Q_next (i.e. Q values over the subsequent state)
    Q_next_preds = list()
    for est in estimators:
        Q_next_preds.append(est.predict(state))

    # column concat the predictions
    Q_next = np.column_stack(Q_next_preds)

    return Q_next


class MultiFittedQAgent(object):
    """
    Make decisions by greedily selecting actions with the highest Q values

    Appropriate for a Fitted Q Iteration agent that has an estimator
    per action (as in Ernst, Geurts & Wehenkel, 2005)
    """

    def __init__(self, action_space, clfs):
        super(MultiFittedQAgent, self).__init__()
        self.action_space = action_space
        self.clfs = clfs

        # assert that there are multiple agents
        if isinstance(clfs, list):
            assert (len(action_space) == len(clfs))

    def _predict(self, state):
        """Make predictions from multiple models"""

        Q_max = max_per_multiple_action(self.clfs, state)

        return Q_max

    def determine_action(self, state):
        """Greedily determine the best action to be taken"""

        # use the two models to make predictions about the Q values
        Q_vals = self._predict(state)

        # greedily select the argmax action
        Q_arg_max = np.argmax(Q_vals, axis=1)

        return Q_arg_max[0]


class NeuralFittedQAgent(object):
    """
    Neural Fitted Q Iteration agent
    """

    def __init__(self, action_space, clf):
        super(NeuralFittedQAgent, self).__init__()
        self.action_space = action_space
        self.clf = clf

    def _predict(self, state):
        """Make predictions from multiple models"""

        Q_max = self.clf.predict(state)

        return Q_max

    def determine_action(self, state):
        """Greedily determine the best action to be taken"""

        # use the two models to make predictions about the Q values
        Q_vals = self._predict(state)

        # greedily select the argmax action
        Q_arg_max = np.argmax(Q_vals, axis=1)

        return Q_arg_max[0]
