# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 04:44:33 2020

@author: yyimi
"""
import numpy as np

def sarsaGTD2_value_function_approx(
        env, gamma, alpha, epsilon, num_episodes, max_steps=np.inf,
        qweights_init=None):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be reset and interacted with via step
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the output policy
    """
    
    num_actions = env.num_actions
    features = len(env.initial)
    qweights_one_action_key = {1:[],2:[],3:[],4:[]}
    qweights_one_action_door = {1:[],2:[],3:[],4:[]}
    reward = []
    num_step = []
    if qweights_init is None:
        # we genearte a weight for each state and each actions
        # the dimension of qweights is 4 * length_of_state_rep
        qweights = np.random.uniform(0,1,num_actions*features).reshape((num_actions, features))
    else:
        qweights = qweights_init
    # policy = get_epsilon_greedy_policy(epsilon, Q, env.absorbing)
    for _ in range(num_episodes):
        # reset state
        s_rep = env.reset() # it will return state_representation
        # we estimate the q function using the weights we have initialised and 
        # the state representation for the selected state
        # choose initial action
        a = sample_from_epsilon_greedy(s_rep, qweights, epsilon)
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            next_s_rep, r = env.next(a)
            # choose the next action
            next_a = sample_from_epsilon_greedy(next_s_rep, qweights, epsilon)
            """
            # get td_error (called delta on slides)
            Q = state_action_value_function_approx(s_rep, a, qweights)
            Q_next = state_action_value_function_approx(next_s_rep, a, qweights)
            td_error = r + gamma*Q_next - Q
            # update the Q function estimate
            Q += alpha*td_error
            """
            # update the weights
            qweights = sarsa_update_GTD2(qweights, s_rep, a, r, next_s_rep, 
                                         next_a, gamma, alpha)
            # set next state and action to current state and action
            s_rep = next_s_rep
            a = next_a
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
    return (qweights,qweights_one_action_key,qweights_one_action_door,reward,num_step)
  

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
    the q_value approximation for the state and action input to put in the sarsa code
    from the fomlads
    USE THIS TO APPROXIMATE THE VALUE OF Q USING THE UPDATED WEIGHT 
    """
    qweights_a = qweights[a]
    return np.dot(s_rep, qweights_a)

def sarsa_update_GTD2(qweights, s_rep, a, r, next_s_rep, next_a, gamma, alpha):
    """
    A method that updates the qweights following the sarsa method for
    function approximation. You will need to integrate this with the full
    sarsa algorithm
    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    q_current = state_action_value_function_approx(s_rep, a, qweights)
    q_next = state_action_value_function_approx(next_s_rep, next_a, qweights) # NEXT STATE REP 
    delta = r +gamma*q_next - q_current
    DeltaW = alpha*(delta-np.dot(s_rep,qweights[a]))*s_rep
    qweights[a] += DeltaW
    return qweights
    

def sample_from_epsilon_greedy(s_rep, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    qvalues = []
    for i in range(qweights.shape[0]):
        qvalues.append(state_action_value_function_approx(s_rep, i, qweights))
    qvalues  = np.array(qvalues)
    if np.random.random() > epsilon:
      return np.argmax(qvalues)
    return np.random.randint(qweights.shape[0])
