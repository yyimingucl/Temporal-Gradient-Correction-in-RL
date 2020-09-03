# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 07:34:13 2020

@author: yyimi
"""
import numpy as np

def mc_evaluation_keydoor(
        env, gamma, policy, num_episodes, max_steps=None, default_value=0):
    """
      Estimates V (state value) function from interacting with an environment
      using the batch method.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      policy - (num_states x num_actions)-matrix of policy probabilities
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      V - an estimate for V
      
    """
   
    num_states = env.num_states
    num_actions = env.num_actions
    # initialise returns lists as dictionary of empty lists (one per state)
    
    # indexed by that state
    returns_lists = { s:[] for s in range(num_states)}
    for _ in range(num_episodes):
        # get a trace by interacting with the environment
        trace = env.run(policy, max_steps=max_steps)
        # iterate over each unique state in the trace and store the return
        # following the first such visit in the corresponding return list
        for s, ret  in trace.first_visit_state_returns(gamma):
            returns_lists[s].append(ret)
            
    # once all experience is gathered, we take the sample average return from
    # each state as the expected return
    V = default_value*np.ones(num_states)
    for s, returns_list in returns_lists.items():
        # if there are any returns for that state then take the average
        if len(returns_list) > 0:
            V[s] = np.mean(returns_list)
        # otherwise the default value will be used
       
    # return the value estimates
        
    return V




def mc_qevaluation_keydoor(
        env_key, env_door, gamma, policy, num_episodes, max_steps=None, default_value=0):
    """
      Estimates Q (state-action value) function by interacting with an
      environment.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      policy - (num_states x num_actions)-matrix of policy probabilities
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      Q - an estimate for Q
      
    """
    both_Q=[]
    for env in [env_key,env_door]:
        num_states = env.num_states
        num_actions = env.num_actions
        # initialise returns lists as dictionary of empty lists (one per
        # state-action pair indexed by that pair)
        returns_lists = {
            (s,a):[] for s in range(num_states) for a in range(num_actions)}
        for _ in range(num_episodes):
            # get a trace by interacting with the environment
            trace = env.run(policy, max_steps=max_steps)
            # iterate over unique state-action pairs in the trace and store the
            # return following the first visit in the corresponding return list
            for (s,a), ret  in trace.first_visit_state_action_returns(gamma):
                returns_lists[(s, a)].append(ret)
        # once all experience is gathered, we take the sample average return from
        # each state as the expected return
        Q = default_value * np.ones((num_states, num_actions))
        for (s, a), returns_list in returns_lists.items():
            # if there are any returns for that state-action pair then take the
            # average
            if len(returns_list) > 0:
                Q[s,a] = np.mean(returns_list)
            # otherwise the default value will be used
        # return the value estimates
        both_Q.append(Q)
        
    return both_Q   