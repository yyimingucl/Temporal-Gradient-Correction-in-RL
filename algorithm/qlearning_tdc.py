# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:41:02 2020

@author: yyimi
"""

import numpy as np

def qtdc_value_function_approx(
        env, gamma, alpha, epsilon, beta, num_episodes, max_steps = np.inf,
        qweights_init=None):
    """
      Estimates optimal policy by interacting with an environment using
      a q-learning approach

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      qweights - weights for our linear value function approximation

    """
    num_actions = env.num_actions
    actions = env.action
    qweights_one_action_key = {1:[],2:[],3:[],4:[]}
    qweights_one_action_door = {1:[],2:[],3:[],4:[]}
    reward = []
    num_step = []
    
    #For adapting the key-door problem
    # look at MazeSim_Features to see how env.initial is initialised
    features = len(env.initial)
    if qweights_init is None:
        # we generate a weight for each state and each actions
        # the dimension of qweights is 4 * length_of_state_rep
        qweights = np.zeros(num_actions*features).reshape((num_actions, features))
    else:
        qweights = qweights_init
    lms = np.zeros(features)
    for _ in range(num_episodes):
        # initialise state
        s_rep = env.reset()
        steps = 0
        done = False
        while not env.is_terminal() and steps < max_steps:
            # choose the action
            a = sample_from_epsilon_greedy(env, s_rep, qweights, epsilon)
            next_s_rep, r = env.next(a)
            # update the weights
            qweights,lms = q_learning_update(actions, qweights, s_rep, a, r,
                                             next_s_rep, lms, gamma, alpha, beta)
            # set next state and action to current state and action
            
            s_rep = next_s_rep
#            
            # increment the number of steps
            steps += 1
            
        for i in range(env.num_actions):
            state = (1,2)
            qweights_one_action_key[i+1].append(np.dot(env.rep_function(state,True),
                               qweights[i]))
            qweights_one_action_door[i+1].append(np.dot(env.rep_function(state,False),
                               qweights[i]))
        reward.append(env.reward)
        num_step.append(steps)
        
        
#         print(qweights) 
    # return the policy
    return qweights,qweights_one_action_key,qweights_one_action_door,reward,num_step




def state_action_value_function_approx(s_rep, a, qweights):
    """
    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action so for us it is 0, 1, 2 or 3
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action

    returns
    -------
    the q_value approximation for the state and action input to put in the qlearning code
    from the fomlads
    """
    qweights_a = qweights[a]
    return np.dot(s_rep, qweights_a)

def q_learning_update(actions, qweights, s_rep, a, r, next_s_rep, lms,
                      gamma, alpha, beta):
    """
    A method that updates the qweights following the qlearning method for
    function approximation.     

    returns
    -------
    the updated weights for our function approximation
    """
    q_current = state_action_value_function_approx(s_rep, a, qweights)
    q_all = []
    for i in actions:
        q_all.append(state_action_value_function_approx(next_s_rep, i, qweights))
    q_next  = np.max(q_all)
    Delta = r +gamma*q_next - q_current
    lms += beta*(Delta - lms.T*s_rep)*s_rep
    
    qweights[a] += alpha*(Delta*s_rep-gamma*next_s_rep*s_rep.T*lms)
    
    return qweights,lms

def sample_from_epsilon_greedy(env,s_rep, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    Action that maximises the value of the Q function
    """

    qvalues = []
    for i in range(qweights.shape[0]):
        qvalues.append(state_action_value_function_approx(s_rep, i, qweights))
    qvalues  = np.array(qvalues)
    if np.random.random() > epsilon:
      return np.argmax(qvalues)
    return np.random.randint(qweights.shape[0])