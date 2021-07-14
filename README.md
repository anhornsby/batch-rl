# Batch Reinforcement Learning in Python

Experimenting with batch Q Learning (Reinforcement Learning) in OpenAI gym. This is intended as a very basic starter code.

The aim of batch reinforcement learning is to learn the optimal policy using offline data, which is useful in contexts where continuous online training may be very costly/impossible.

Fitted Q Iteration learns the optimal policy by continuously retraining agents on batches of observations collected from the environment. The agent is initialised with a random policy, acts in the environment for `n_episodes`, and a Q learning agent is subsequently re-trained on those episodes. This process
repeated for `n_iterations`. 

## Running

The model training process can be started by running:

```shell
python run.py
```

## Configuring the run

The training process can be configured in the `config.py`. The key parameters to configure are:

* `n_episodes` - How many episodes do you wish to simulate for each iteration? Each timestep will be collected into a batch, which will be used to train the agent in subsequent iterations.
* `grow_batch` - If `True`, then we will re-use episodes from all iterations. If `False`, then we will only use episodes from the latest simulation.
* `n_iterations` - The number of fitted Q iterations (i.e., how many times the agent will be simulated and subsequently re-trained).
* `algorithm` - `single` trains an agent with a single function approximator, but outputs different Q estimates for each action using one-hot encoding to represent each action. `multiple` trains multiple estimators, one for each action. `neural` fits a multi-headed neural network, predicting Q values for each action.

The others probably don't need to be changed.

## Results

Plots of the bellman loss for each timestep and the mean number of timesteps for each simulation will be saved to `plots/`.

You will probably see that the model is quite unstable, in that it will frequently learn and then forget the optimal solution over subsequent iterations.
