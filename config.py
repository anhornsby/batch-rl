# coding: utf-8
"""
Batch Reinforcement Learning configuration
Adam Hornsby
"""

import copy
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

CONFIG = {

    # open AI
    'environment': 'CartPole-v0',  # set to run a maximum of 300 steps

    # save path
    'plot_save_path': 'plots/',

    # algorithm tyoe
    'algorithm': 'multiple',  # single, multiple or neural
    'grow_batch': True,  # collected episodes over iterations

    # episodes
    'n_episodes': 200,
    'max_steps': 200,
    'n_simulations': 10,

    # hyperparams
    'lambda': 0.99,  # discount factor
    'n_iterations': 50,  # number of fitted q iterations

    # rewards"
    'success': 0,
    'failure': -1,
    'end_reward': 0,

    # experience
    'replay_sample': None,

    # model
    'model_param_grid': {'alpha': [0.00001]},
    'model': LinearRegression()
# RandomForestRegressor() #DecisionTreeRegressor() # XGBRegressor() # MLPRegressor([100], 'logistic', solver='sgd', learning_rate_init=0.001)
}

# CONFIG.update({'model' : GridSearchCV(CONFIG['model'], CONFIG['model_param_grid'], n_jobs=1)})
CONFIG.update({'model': [CONFIG['model'], copy.deepcopy(CONFIG['model'])]})
