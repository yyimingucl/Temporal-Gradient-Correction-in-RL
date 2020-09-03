# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 03:56:57 2020

@author: yyimi
"""
import numpy as np

def n_step_sarsa(env, gamma, alpha,epsilon, n,
                 num_episodes, max_steps=np.inf, qweights_init=None):
    # Initialization
    num_actions = env.num_actions
    features = len(env.initial)   
    if qweights_init is None:
        qweights = np.random.uniform(0,1,num_actions*features).reshape((num_actions, features))
    else:
        qweights = qweights_init

    for _ in range(num_episodes):
        # initializations
        T = np.inf
        tau = 0
        t = -1
        stored_actions = dict([(i, 0)for i in range(n+1)])
        stored_rewards = dict([(i, 0)for i in range(n)])
        stored_states = dict([(i, 0)for i in range(n+1)])
        s_rep = env.reset() 
        
        a = sample_from_epsilon_greedy(s_rep, qweights, epsilon)
        # With prob epsilon, pick a random action
        stored_states[0] = s_rep
        stored_actions[0] = a
        
        while tau != T-1:
            t+=1
            next_s_rep, r =env.next(a)
            stored_rewards[t%n] = r
            stored_states[(t+1)%n] = next_s_rep
            
            if env.is_terminal():
                T = t+1
            else:
                next_a = sample_from_epsilon_greedy(s_rep,qweights,epsilon)
                stored_actions[(t+1)%n] = next_a
                a = next_a
            tau = t-n+1
            if t%n == 0 and t!=0:
                G = np.sum([(gamma**(i-tau-1))*stored_rewards[i%n] 
                    for i in range(tau+1,min(tau+n,T)+1)])
                if tau+n < T:
                    qweights = sarsa_update(qweights, stored_states[0],stored_actions[0],
                                       G,stored_states[n],stored_actions[n],gamma,alpha,n)
                elif tau+n >= T:
                    qweights = sarsa_update(qweights,stored_states)
    return qweights
            
            
            
        


    
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





def sarsa_update(qweights, s_rep, a, r, next_s_rep, next_a, gamma, alpha,n):
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
    DeltaW = alpha*(r +(gamma**n)*q_next - q_current)*s_rep
    qweights[a] += DeltaW
    return qweights



