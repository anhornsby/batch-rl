# coding: utf-8
"""
Batch Reinforcement Learning using Open AI gym
Adam Hornsby
"""
from __future__ import print_function

import sys
import gym
import pandas as pd
import numpy as np

# own functions
from config import CONFIG
from agent import RandomAgent, NeuralFittedQAgent, MultiFittedQAgent, SingleFittedQAgent
from algorithms import multiple_fitted_q_iteration, single_fitted_q_iteration, neural_fitted_q_iteration
from visualise import plot_fqi_results

np.random.seed(42)


# Here we use Fitted Q Iteration (a batch RL algorithm) as described in:
# https://www.researchgate.net/publication/220320978_Tree-Based_Batch_Mode_Reinforcement_Learning
# (note that here they often maintain a model per discrete action, like here)
#
# To do this, we take the following steps:
# 1. Set N = 0
# 2. Initialise our Q functions (one per action)
# 3. Repeat until stopping criterion reached:
# 4. Set $i^{l} = (Xt, ut)$ where X is the state and u is the action
#   For each (Q function, action):
# 5. Set $o^{l} = r_t + \gamma * Q_{n-1} (X_{t+1}, u)$ where action == a
# 7. Set each observation in your training set $TS = (i^{l}, o^{l})$ such that input=i, target=o
# 8. Train each regression model TS on tuples where a particular action has been taken
#
# So to speak in English, the target defined in step 5 $o^{l}$ is the reward at the current time step
# plus the discount factor multiplied by the previous regression model's estimate of what the value
# of the next state would be.
#
# In episodic tasks, we don't bootstrap future values in terminal states. However - given this is
# a non-episodic task - we will simply ignore the last state (as there is no subsequent state). Thus
# we'll always bootstrap. We just have to make sure that gamma is high (to ensure long term value) but not 1
# (to ensure the action values are finite).


def cartpole_shape_reward(done, success=0, failure=-1):
    """Shape the reward"""

    if done:
        return failure
    else:
        return success


def simulate_environment(env, model, n_episodes, max_steps, success, failure, episode_end_reward=1):
    """Simulate an environment for n_episodes for max_steps and return the results of that"""

    mean_timesteps = list()

    # simulate episodes
    training_sims = list()
    for i in range(n_episodes):

        observation_i = env.reset()

        # step through this single episode
        for j in range(max_steps):
            action = model.determine_action(observation_i.reshape(1, -1))
            observation_i, reward_i, done_i, _ = env.step(action)

            # shape the reward
            target_i = cartpole_shape_reward(done_i, success=success, failure=failure)

            if j == (max_steps - 1):
                target_i = episode_end_reward  # if max_steps then give big positive reward

            # collate those in a training set
            # columns: episode, step, state_0, state_1, state_2, state_3, reward
            training_sims_i = np.hstack([[i], [j], observation_i, [action, reward_i, target_i, not done_i]])
            training_sims.append(training_sims_i)

            if done_i:
                print("Episode {0:d} finished after {1:d} timesteps  \r".format(i, j), end="")
                sys.stdout.flush()
                mean_timesteps.append(j)
                break

    # calculate mean timesteps
    mean_timesteps = np.mean(mean_timesteps)

    # row stack those
    training_sims = np.vstack(training_sims)

    return training_sims, mean_timesteps


def make_environment(env_name, seed=42):
    """Make the OpenAI environment"""

    # initialise the gym environment
    env = gym.make(env_name)
    env.seed(seed)

    return env


def generate_batch(env, q_model, n_episodes, max_steps, success_reward, failure_reward, end_reward):
    """Generate a batch of data using an agent in q_model"""

    # simulate the environment with a random agent
    t_X, q_mean_timesteps = simulate_environment(env,
                                                 q_model,
                                                 n_episodes,
                                                 max_steps,
                                                 success_reward,
                                                 failure_reward,
                                                 episode_end_reward=end_reward)

    # restructure as DataFrame
    # TODO: dynamically name the state space
    training_x = pd.DataFrame(t_X, columns=['episode', 'step', 'state_0',
                                            'state_1', 'state_2', 'state_3',
                                            'action', 'reward', 'target', 'not_terminal'])

    return training_x, q_mean_timesteps


def determine_algorithm(algorithm_name):
    """Determine the algorithm to be used"""

    if algorithm_name == 'multiple':
        # one model per action
        algorithm = multiple_fitted_q_iteration
        agent = MultiFittedQAgent

    elif algorithm_name == 'single':
        algorithm = single_fitted_q_iteration
        agent = SingleFittedQAgent

    elif algorithm_name == 'neural':
        algorithm = neural_fitted_q_iteration
        agent = NeuralFittedQAgent

    else:
        raise NotImplementedError

    return algorithm, agent


def iteratatively_train_agent(env, model, training_algorithm, n_simulations, n_episodes, max_steps,
                              success_reward, failure_reward, end_reward, lambda_val=0.99, n_iterations=1,
                              grow_batch=False, save_path=None):
    """Repeatedly simulate batches and train batch RL agent"""

    # initialise the stochastic agent
    q_model = RandomAgent(action_space=np.arange(env.action_space.n))

    # initialise data collection
    mean_timesteps = list()
    losses = list()

    # determine training algorithm
    algorithm, agent = determine_algorithm(training_algorithm)

    for sim in range(0, n_simulations):

        # generate a random batch of data
        t_X, q_mean_timesteps = generate_batch(env, q_model, n_episodes, max_steps,
                                               success_reward, failure_reward, end_reward)

        # grow the batch if necessary
        if sim > 0:
            # for i>0, grow the training set (including simulations from previous iterations too)
            if grow_batch:
                training_X = pd.concat([training_X, t_X], axis=0)
            else:
                training_X = t_X

            mean_timesteps.append(q_mean_timesteps)
        else:
            training_X = t_X
            timesteps_baseline = q_mean_timesteps

        # train the agent on the batch
        # TODO: dynamically name state space
        models, mean_loss = algorithm(model,
                                      training_X[['state_0', 'state_1', 'state_2', 'state_3']].values,
                                      training_X['action'].values,
                                      training_X['target'].values,
                                      training_X['not_terminal'].values,
                                      discount=lambda_val,
                                      iterations=n_iterations)

        # run the simulation with a trained model
        q_model = agent(np.arange(env.action_space.n), models)

        losses.append(mean_loss)

    # generate a final test batch
    _, q_mean_timesteps = generate_batch(env, q_model, n_episodes, max_steps,
                                         success_reward, failure_reward, end_reward)
    mean_timesteps.append(q_mean_timesteps)

    # generate a plot of the results
    plot_fqi_results(mean_timesteps, timesteps_baseline, losses, save_path)

    return q_model


def main(config):
    """Run the simulation and train the agent"""

    # initialise the environment
    env = make_environment(config['environment'], seed=42)

    # train the model using fitted q iteration
    iteratatively_train_agent(env, config['model'], config['algorithm'], config['n_simulations'],
                              config['n_episodes'], config['max_steps'],
                              config['success'], config['failure'], config['end_reward'], lambda_val=config['lambda'],
                              n_iterations=config['n_iterations'],
                              grow_batch=config['grow_batch'], save_path=config['plot_save_path'])


if __name__ == '__main__':
    main(CONFIG)
