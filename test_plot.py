# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 03:38:58 2020

@author: yyimi
"""

# Testing Algorithm 
import environment.env_maze as env
import environment.maze_sim as mazesim
import numpy as np
import algorithm.qlearning_funct_app as QL
import algorithm.sarsa_func_app as SA
from plot.graph import test_parameters
from plot.graph import get_trace
from plot.graph import plot_weights
from plot.graph import get_determinant_policy
from plot.graph import plot_reward
from plot.graph import val_path
from plot.graph import plot_step

#      General Information
#---------------------------------

maze =[[2,1,1,3],
      [0,0,1,0],
      [1,1,1,0],
       [0,0,1,2]]

maze =[[2,1,0,1,1],
       [0,1,1,1,0],
       [3,0,1,0,0],
       [1,1,1,1,2]]

maze = [[2,1,1,1,1],
        [0,0,1,0,1],
        [3,1,1,0,1],
        [0,0,1,1,1],
        [0,0,1,1,2]]

"""
maze = [[2,1,1,0,1,0,0,0],
        [1,0,1,0,1,1,1,3],
        [1,0,1,1,1,0,0,1],
        [1,0,1,0,1,0,0,1],
        [1,1,1,0,1,0,0,1],
        [1,0,1,1,1,1,1,1],
        [1,0,0,0,0,0,1,0],
        [0,0,1,1,1,1,1,2]]
"""

action = ["up","right","down","left"]

maze_grid = env.maze_grid(maze,action)
feature_maze_sim = mazesim.MazeSim_Features(maze_grid,feature = "onehot_tiling")

num_episodes = 300

#       Analysis for Sarsa
#---------------------------------------
sasw,sasw_key,sasw_door,sasreward,sasteps = SA.sarsa_value_function_approx(feature_maze_sim,
                                                                          0.9,0.1,0.01,num_episodes)
#determinant policy
determinant_policy = get_determinant_policy(feature_maze_sim, sasw)
print(determinant_policy)

#1. Trace
get_trace(determinant_policy)
val_path(feature_maze_sim,determinant_policy)

#2. Weights
plot_weights(num_episodes,sasw_key)
plot_weights(num_episodes,sasw_door)

#3. Rewards
plot_reward(num_episodes,sasreward)

#4. Steps
plot_step(num_episodes,sasteps)




#       Analysis for Q-Learning
#----------------------------------------
qw,qw_key,qw_door,qreward,qsteps = QL.q_learning_value_function_approx(feature_maze_sim,
                                                                       0.9,0.1,0.01,num_episodes)

#determinant policy
determinant_policy = get_determinant_policy(feature_maze_sim, qw)
print(determinant_policy)

#1. Trace
get_trace(determinant_policy)
val_path(feature_maze_sim,determinant_policy)

#2. Weights
plot_weights(num_episodes,qw_key)
plot_weights(num_episodes,qw_door)

#3. Rewards
plot_reward(num_episodes,qreward)

#4. Steps
plot_step(num_episodes,qsteps)



#------------------------------------------------
#         Compare two algorithms
#------------------------------------------------

import matplotlib.pyplot as plt
episodes = [j+1 for j in range(num_episodes)]

#1. Reward
plt.plot(episodes, sasreward,label = "Sarsa")
plt.plot(episodes, qreward, label = "Q-Learning")
plt.legend()
plt.ylabel("Total Reward")
plt.xlabel("Episodes")
plt.title("Compare Two Algorithms")


#2. Steps 
plt.plot(episodes, sasteps,label = "Sarsa")
plt.plot(episodes, qsteps, label = "Q-Learning")
plt.legend()
plt.ylabel("Total Steps")
plt.xlabel("Episodes")
plt.title("Compare Two Algorithms")






plt.plot(episodes, qsteps, label = "Q-learning")
plt.plot(episodes, a[-1], label = "Q-learning with TDC")
plt.ylabel("Total Steps")
plt.xlabel("Episodes")
plt.legend()
plt.title("Compare Two Algorithms")









