# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 03:05:33 2020

@author: yyimi
"""

import numpy as np

def nstep_sarsa_function_approx(env,gamma,alpha,epsilon, n,
                 num_episodes, max_steps=10000, qweights_init=None):
    """
    this algorithm is for n-step sarsa function approximation
    parameters:
        1. env - the environment
        2. gamma - the decay coefficient 
        3. alpha - the learning rate
        4. epsilon - the epsilon to use with epsilon greedy policies 
        5. n - the step size n
        6. num_episodes - number of episode to run
        7. max_steps (optional) - maximum number of steps per trace (to avoid very
           long episodes)
    ---------------------------------------------------------------------    
    returns: 
        1. the estimated qweights
        2. list of rewards
        3. list of number of steps
        .....
    """
    num_actions = env.num_actions
    features = len(env.initial)   
    if qweights_init is None:
        # we genearte a weight for each state and each actions
        # the dimension of qweights is 4 * length_of_state_rep
        qweights = np.random.uniform(0,1,num_actions*features).reshape((num_actions, features))
    else:
        qweights = qweights_init
    # policy = get_epsilon_greedy_policy(epsilon, Q, env.absorbing)
    for _ in range(num_episodes):
        stored_actions = {}
        stored_rewards = {}
        stored_states = {}
        # reset state
        s_rep = env.reset() # it will return state_representation
        # we estimate the q function using the weights we have initialised and 
        # the state representation for the selected state
        # choose initial action
        a = sample_from_epsilon_greedy(s_rep, qweights, epsilon)
        T = np.inf
        t = -1
        tau = 0
        stored_actions[0] = a
        stored_states[0] = s_rep
        end = False
        
        while end:
            t += 1
            if t < T and env.is_terminal():
                
                next_s_rep, r = env.next(a)
                stored_rewards[(t+1)%n] = r        
                stored_states[t%n] = next_s_rep 
                
                s_rep = next_s_rep
                
                if env.is_terminal():
                    T = t+1
                else:
                    next_a = sample_from_epsilon_greedy(next_s_rep, qweights, epsilon)
                    stored_actions[t%n] = next_a
                    a = next_a
               
                
            tau = (t%n)+1-n
            if tau >= 0:
                qweights = n_step_sarsa_update(qweights,stored_actions,stored_states,
                                               stored_rewards,tau,n,T,gamma,alpha)
                    
                    
                    
                
                
    return qweights
                    


def n_step_sarsa_update(qweights,stored_actions,stored_states,stored_rewards,tau,
                        n,T,gamma,alpha):
    """
    This function is used the reward-sequence from timepoint tau to tau+n
    to update the qweights
    parameters:
        1.qweights - current qweights
        2.action - the sequence of action from tau to tau+n
        3.trace - the sequence of states
        4.R - the sequence of reward
    -------------------------------------------------------
    return:
        qweights - the updated qweights
    """
    G = np.sum([gamma**(i-tau-1) * stored_rewards[i%n] 
                for i in range(tau+1, min(tau+n, T)+1)])
    if tau+n < T:
        Q_n_further = state_action_value_function_approx(stored_states[n-1],
                                                    stored_actions[n-1],qweights)
        G += (gamma**n)*Q_n_further
        Q_tau = state_action_value_function_approx(stored_states[0],
                                                   stored_actions[0],qweights)
        
        qweights[stored_actions[tau%n]] += alpha*(G-Q_tau)*stored_states[tau%n]
    
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



def sarsa_update(qweights, s_rep, a, r, next_s_rep, next_a, gamma, alpha):
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
    DeltaW = alpha*(r +gamma*q_next - q_current)*s_rep
    qweights[a] += DeltaW
    return qweights