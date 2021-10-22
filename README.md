# RL_project: Comparison of Q-Learning and Double Q-Learning
Het grote versterkende leren project

For the plots see the results.ipynb notebook. 

# Requirements
Python >= 3.8.5 with:
* numpy >= 1.19.2
* matplotlib >= 3.3.4
* gym >= 0.19.0
* tqdm >= 4.36.0

# Overview of files
## Environments (codebase/environments/)
Contains the implementations of the Gym environments, with the following files:
* codebase/environments/stochasticEnv.py: The base stochastic Gridworld Gym environment implementation
* codebase/environments/grid_world.py: The deterministic, stochastic and left-right Gridworld implementations used for the paper
* codebase/environments/left_right.py: The Left-Right problem implementatation

## Models (codebase/models/)
Contains the implementation of Q-Learning and Double Q-Learning (codebase/models/q_learning.py) and the behaviour policies (codebase/models/policies.py)

## Experiments  (codebase/experiments.py)
Finally, codebase/experiments.py contains the code for the different experiments we ran to obtain the results in the paper


