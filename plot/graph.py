# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 03:39:12 2020

@author: yyimi
"""

import numpy as np
import algorithm.qlearning_funct_app as QL
import algorithm.sarsa_func_app as SA
import matplotlib.pyplot as plt


def get_determinant_policy(env, qweights):
    """
    give the determiant policy for each state
    1. env is the featuring environment
    2. qweights is the weightes we estimated from different algorthms 
    ------------------------
    return the determiant policy
    
    """
    num_actions = env.num_actions
    determinant_policy = {"without_key":{},"with_key":{}}
    is_finding_key = [True, False]
    key_status = ["without_key","with_key"]
    feature_mapping = env.rep_function # this is a function that returns feature rep for each state
    total_states = env.states
    for key, finding in zip(key_status, is_finding_key):
        for state in total_states:
            rep = feature_mapping(state, finding)
            value_action = np.zeros(num_actions)
            for i in range(num_actions):
                value_action[i] = np.dot(qweights[i],rep.T)
                
            best_action = np.argmax(value_action)
            determinant_policy[key][state] = best_action

    return determinant_policy

def correct(determinant_policy):
    # for the following 4x4 maze the correct policy the agent should learn is:
    # maze =[[2,1,1,3],
    #       [0,0,1,0],
    #       [1,1,1,0],
    #       [0,0,1,2]]
    correct_answers_without_key = [1,1,1,0,2,0,1,1,0,3]
    correct_key = []
    i = 0        
    for key in determinant_policy['without_key']:
        if determinant_policy['without_key'][key] ==correct_answers_without_key[i]:
            correct_key.append(1)
        else:
            correct_key.append(0)
        i += 1

    correct_answers_with_key = [1,1,2,3,2,1,1,2,1,2]
    correct_door = []
    j=0
    for key in determinant_policy['with_key']:
        if determinant_policy['with_key'][key] ==correct_answers_with_key[j]:
            correct_door.append(1)
        else:
            correct_door.append(0)
        j += 1
    
    return (correct_key, correct_door)
    
# print(correct(determinant_policy))

            

def test_parameters (env,parameter):
    """
    Input parameters:
    1. env is the featuring environment
    2. paramter is one of the parameters gamma, epsilon and alpha
       for which we want to iterate through to estimate learning 
       efficiency as this parameter varies. 
    ------------------------
    return 
    
    """
    
    parameter_v = []
    if parameter  == 'alpha':
         
        parameter_v = np.arange(0.01,1,0.01)
        
        for i in parameter_v:

            qweights = QL.q_learning_value_function_approx(env,0.9,i,0.01,100)[0]
            determinant_policy = get_determinant_policy(env, qweights)

        a,b = correct(determinant_policy)
            
    if parameter  == 'epsilon':
        
        parameter_v = np.arange(0.01,1,0.01)  
        
        for i in parameter_v:

            qweights = QL.q_learning_value_function_approx(env,0.9,0.05,i,100)[0]
            determinant_policy = get_determinant_policy(env, qweights)

        a,b = correct(determinant_policy)

    if parameter  == 'gamma':

        parameter_v = np.linspace(0.8,0.9,num = 100)  

        for i in parameter_v:

            qweights = QL.q_learning_value_function_approx(env,i,0.05,0.01,100)[0]
            determinant_policy = get_determinant_policy(env, qweights)

        a,b = correct(determinant_policy)

    correct_total = sum(b)
    plt.scatter(parameter_v,correct_total)
    plt.xlabel(parameter)
    plt.ylabel ('Number of correct answers found')
    plt.show()
#     plt.savefig('Correct answers vs epsilon')


def plot_weights(num_episodes, qweights_one_state):
    
    episodes = np.arange(1,num_episodes+1,1)
    plt.figure(figsize=(10, 8))
    plt.plot(episodes,qweights_one_state[1],label = 'action 1')
    plt.plot(episodes,qweights_one_state[2],label = 'action 2')
    plt.plot(episodes,qweights_one_state[3],label = 'action 3')
    plt.plot(episodes,qweights_one_state[4],label = 'action 4')
    plt.xlabel('Number of episodes')
    plt.ylabel('Determaint Values')
    plt.title("Analysis for determiant value against episodes")
    plt.legend()
    plt.show()

#     plt.savefig('Weight covergence for single state_2')


def get_trace(determinant_policy):
    actions = {0:'Up', 1:'Right',2:'Down',3: 'Left'}
    print('Final trace of actions chosen by determinant policy to get to the key:')
    for i in determinant_policy['without_key']:
        print(i, actions[determinant_policy['without_key'][i]])
    # print('-------------') 
    print('Final trace of actions chosen by determinant policy to get to the door:')   
    for i in determinant_policy['with_key']:
        print(i, actions[determinant_policy['with_key'][i]])

# test_parameters (xy_maze_sim,'epsilon')

def plot_reward(num_episodes,reward):
    """
    num_episodes: the number of episodes.
    reward: a list contains the reward after each episode.
    
    """
    episodes = np.arange(1,num_episodes+1,1)
    plt.plot(episodes,reward)
    plt.ylabel("Total Reward")
    plt.xlabel("Episodes")
    plt.title("Analysis for reward")
    plt.show()
    
def val_path(env, determinant_policy):
    """
    env(maze_grid) is the grid-world environment (not simulation)
    
    policy is the moving policy
    """    
    two_path_sets=[[(env.initial_state[0]+0.5,
                     env.initial_state[1]+0.5)],[env.key]]

    choose_action = [[action for _,action in determinant_policy["without_key"].items()],
                      [action for _,action in determinant_policy["with_key"].items()]]
    terminal = [env.key, env.door]
    start_state = [env.initial_state,env.key]
    for actions, end, start, path_set in zip(choose_action,
                                             terminal,start_state,two_path_sets):
        a_s_dict = {state:action for state,action in zip(env.states,actions)}
        next_state = start
        
        while next_state != end :
            action = a_s_dict[next_state]
            
            i, j = next_state
            if action==0 :
                next_state = (i-1,j)
            elif action==1: 
                next_state = (i,j+1)
            elif action==2:
                next_state = (i+1,j)
            elif action==3:
                next_state = (i,j-1)
            
            
            loc_mapping = (next_state[1]+0.5,next_state[0]+0.5)
            path_set.append(loc_mapping)
            
              
             
    path_sets = two_path_sets[0]+two_path_sets[1]
    path_sets.remove(env.key)
    plt.pcolormesh(env.maze) 
    xs, ys = zip(*path_sets)
    plt.plot(xs, ys, 'x--', lw=2, color='red', ms=10)
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().invert_yaxis()#invert the y-axis so the first row of data is at the top
    plt.show()
    

def plot_step(num_episodes, step):
    """
    num_episodes: the number of episodes.
    step: a list contains the total steps of each episode.
    
    """
    episodes = np.arange(1,num_episodes+1,1)
    plt.plot(episodes, step)
    plt.ylabel("Total Steps")
    plt.xlabel("Episodes")
    plt.title("Analysis for Steps")
    plt.show()
    
    


