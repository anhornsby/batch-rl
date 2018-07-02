# coding: utf-8
"""
Batch Reinforcement Learning training algorithms
Adam Hornsby
"""

import pandas as pd
import numpy as np

from agent import max_per_multiple_action
from sklearn.metrics import mean_squared_error


def separate_actions(state, action, n_actions):
    # separate actions for model training
    Xs = list()
    for a in range(n_actions):
        Xs.append(state[action == a])

    return Xs


def multiple_fitted_q_iteration(estimators, state, action, target, not_terminal, discount=0.99, iterations=10):
    """
    Perform fitted Q iteration using a model per action. Return trained models.
    """

    n_actions = int(len(np.unique(action)))

    # FQI training
    # now train two models: one on action=0 and another on action=1
    Xs = separate_actions(state, action, n_actions)
    ys = separate_actions(target, action, n_actions)
    nts = separate_actions(not_terminal, action, n_actions)

    # initialise Qs with 0s everywhere (train on reward in first instance)
    Q_next = [np.zeros((x.shape[0], n_actions)) for x in Xs]

    losses = list()
    for i in range(iterations):
        print('Performing fitted Q iteration {0:d}'.format(i))
        for j in np.unique(action):
            j = int(j)

            # if first state then q_primes are set to 0
            q_prime = Q_next[j]

            # calculate target variable
            # bootstrap on non-terminal states
            # just set target to reward on terminal states
            y = ys[j] + nts[j].astype(bool) * (discount * q_prime.max(1))

            # train the model
            estimators[j].fit(Xs[j], y)

            loss = mean_squared_error(estimators[j].predict(Xs[j]) ** 2., y)
            losses.append(loss)

        Q_next = max_per_multiple_action(estimators, state)

        # shift upwards so that predictions denote subsequent state
        Q_next = np.vstack([Q_next[1::, :], [0] * n_actions])

        # split them out as with above
        Q_next = separate_actions(Q_next, action, n_actions)

    mean_loss = np.mean(losses)

    return estimators, mean_loss


def neural_fitted_q_iteration(clf, state, action, target, not_terminal, discount=0.99, iterations=10):
    """
    Perform neural fitted Q iteration using a model per action. Return trained models.
    """

    n_actions = int(len(np.unique(action)))

    # initialise Qs with 0s everywhere (train on reward in first instance)
    Q_next = [target for x in range(n_actions)]  # np.zeros((state.shape[0], n_actions))
    Q_next = np.column_stack(Q_next)

    losses = list()
    for i in range(iterations):
        print('Performing fitted Q iteration {0:d}'.format(i))

        # calculate target variable
        y = target + not_terminal * (discount * Q_next.max(1))
        Q_next[:, action.astype(int)] = y

        # train the model
        clf.fit(state, Q_next)

        # calculate Q_next
        Q_next = clf.predict(state)

        # shift upwards so that predictions denote subsequent state
        Q_next = np.vstack([Q_next[1::, :], [0] * n_actions])

        losses.append(0.5 * np.sum(y - clf.predict(state) ** 2.))  #  TODO: check this

    mean_loss = np.mean(losses)

    return clf, mean_loss


def single_fitted_q_iteration(clf, state, action, target, not_terminal, discount=0.99, iterations=10):
    """
    Perform fitted Q iteration using a single model acoss all actions. Actions are
    one-hot-encoded as features.

    Return trained models.
    """

    n_actions = int(len(np.unique(action)))

    actions = pd.get_dummies(action).values

    # initialise Qs with 0s everywhere (train on reward in first instance)
    Q_next = np.zeros((state.shape[0], n_actions))

    losses = list()
    for i in range(iterations):
        print('Performing fitted Q iteration {0:d}'.format(i))

        # calculate target variable
        y = target + not_terminal * (discount * Q_next.max(
            1))

        # calculate x by concatenating state with dummy coded actions
        X = np.column_stack([state, actions])

        # train the model
        clf.fit(X, y)

        # calculate Q_next
        Q_next_preds = list()
        for a in range(n_actions):
            # make predictions for each different action type
            test_X_action = np.zeros((X.shape[0], n_actions))
            test_X_action[:, a] = 1.  # set 1s everywhere for action a

            # concatenate and predict
            test_X = np.column_stack([state, test_X_action])
            Q_next_preds.append(clf.predict(test_X))

            losses.append(0.5 * np.sum(y - clf.predict(test_X) ** 2.))

            # column concat the predictions
        Q_next = np.column_stack(Q_next_preds)

        # shift upwards so that predictions denote subsequent state
        Q_next = np.vstack([Q_next[1::, :], [0] * n_actions])

    mean_loss = np.mean(losses)

    return clf, mean_loss
