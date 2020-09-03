import numpy as np
def choose_from_policy(policy, state):
    num_actions = policy.shape[1]
    return np.random.choice(num_actions, p=policy[state,:])

class Trace(object):
    """
    A trace object stores a sequence of states, actions and rewards for
    an RL style agent. Can calculate returns for the given trace.
    """

    def __init__(self, initial_state, state_names=None, action_names=None):
        """
        Construct the trace with the initial state, and state/action names
        for nice output.

        parameters
        ----------
        """
        self.states = [initial_state]
        self.actions = []
        self.rewards = []
        # if names aren't provided then the output will use the index
        self.state_names = state_names
        self.action_names = action_names

    def record(self, action, reward, state):
        """
        Record the chosen action and the subsequent reward and state.
        """
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)

    def trace_return(self, gamma, t=0):
        """
        Gets the geometrically discounted return from a given time index
        """ 
        return np.sum(gamma**k * r for k, r in enumerate(self.rewards[t:]))

    def first_visit_state_returns(self, gamma):
        """
        Given a geometric discount gets a state indexed return for
        each unique state, corresponding to the first visit return.
        """
        # a dictionary stores the returns
        first_visit_returns = {}
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, r) in enumerate(zip(self.states, self.rewards)):
            # check whether state has been seen already
            if not s in first_visit_returns:
                 first_visit_returns[s] = self.trace_return(gamma,t)
        return list(first_visit_returns.items())

    def every_visit_state_returns(self, gamma):
        """
        Given a geometric discount gets a state indexed return for
        each state appearing in the trace.
        """
        every_visit_returns = []
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, r) in enumerate(zip(self.states, self.rewards)):
             every_visit_returns.append((s, self.trace_return(gamma,t)))
        return every_visit_returns

    def first_visit_state_action_returns(self, gamma):
        """
        Given a geometric discount gets a (state, action) indexed return for
        each unique (state, action), corresponding to the first visit return.
        """
        # a dictionary stores the returns to keep track of first visits
        first_visit_returns = {}  
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, a, r) in enumerate(
                zip(self.states, self.actions, self.rewards)):
            # check whether state has been seen already
            if not (s, a) in first_visit_returns:
                 first_visit_returns[(s,a)] = self.trace_return(gamma,t)
        return list(first_visit_returns.items())

    def every_visit_state_action_returns(self, gamma):
        """
        Given a geometric discount gets a (state, action) indexed return for
        each (state, action) appearing in the trace.
        """
        every_visit_returns = []
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, a, r) in enumerate(
                zip(self.states, self.actions, self.rewards)):
            every_visit_returns.append( ((s, a), self.trace_return(gamma,t)))
        return every_visit_returns

    def __str__(self):
        """
        Convert the trace to a readable string
        """
        # for readability give states, actions and rewards short names
        states = self.states
        actions = self.actions
        rewards = self.rewards
        # if we don't have state names then use string version of index
        if self.state_names is None:
            state_names = {s:str(s) for s in np.unique(states)}
        else:
            state_names = self.state_names
        # if we don't have action names then use string version of index
        if self.action_names is None:
            action_names = {a:str(a) for a in np.unique(actions)}
        else:
            action_names = self.action_names
        # trace output
        out = str(state_names[states[0]])
        for a,r,s in zip(actions, rewards, states[1:]):
            out += ", " + str(action_names[a]) + ", " + str(r) + ", " 
            out += str(state_names[s])
        return  out

class Simulation(object):
    """
    A general simulation class for discrete state and discrete actions,
    any inheriting class must define reset(), next(action),
    is_terminal() and step(action). See MDPSimulation for examples of these.
    """
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        

    def run(self, policy, max_steps=None):
        """
        parameters
        ----------
        max_steps - maximum number of steps per epsiode, max_steps;
        policy - control policy, a (num_states x num_actions) matrix, where 
            each row is a probability vector over actions, given the
            corresponding state

        returns
        -------
        trace - a simulated trace/episode in the sim following the policy
           a tuple of (trace_states, trace_actions, trace_rewards)
        """
        if max_steps is None:
            max_steps = np.inf
        step = 0
        # get the initial state
        state = self.reset()
        # store the initial state
        trace = Trace(state, self.state_names, self.action_names)
        while not self.is_terminal():
            step += 1
            # if the trace has not terminated then choose another action
            action = choose_from_policy(policy, state)
            # the posterior state and reward are drawn from next()
            next_state, reward = self.next(action)
            # store the action, state and reward
            trace.record(action, reward, next_state)
            state = next_state
            if step >= max_steps:
                break
        #
        return trace

    def reset_counts(self):
        """
        The default reset behaviour, you will need to extend this with state
        initialisation for any inheriting class.

        In particular, this helps to monitor performance, by providing a record
        of  total reward and total steps per episode. Note that total reward
        is not discounted.
        """
        self.reward_this_episode = 0
        self.steps_this_episode = 0
        return None

    def increment_counts(self, reward):
        """
        The default behaviour for a step, you will need to extend this with 
        state transitions for any inheriting class
        """
        self.reward_this_episode += reward
        self.steps_this_episode += 1


    def step(self, action):
        """
        The step function mimics the environment step function from the OpenAI
        Gym interface. Note that this returns a 4th value, but for our purposes
        this will always be None

        returns
        -------
        next_state - the next observed state
        reward - the reward for the transition
        done - if the environment is terminal
        None - a blank return value (you can safely ignore this)
        """
        next_state, reward = self.next(action)
        done = self.is_terminal()
        return next_state, reward, done, None


class MDPSimulation(Simulation):
    """
    A class to simulate an environment based on an mdp. Inherits the run()
    function from the Simulation class.
    """
    def __init__(self, model):
        super(MDPSimulation, self).__init__(
            model.num_states, model.num_actions, model.state_names,
            model.action_names) 
        self.action_names = model.action_names
        # boolean vector over state ids indicating which states are absorbing
        self.absorbing = model.absorbing
        # distribution over initial states
        self.initial = model.initial
        # the transition function
        self.t = model.t
        # the reward function
        self.r = model.r 

    def reset(self):
        self.reset_counts()
        # Initialises state from initial distribution and then outputs state
        self.state = np.random.choice(self.num_states, p=self.initial)
        return self.state
    
    def next(self,action):
        ## takes:
        ##      action choice, action
        ## returns:
        ##      next state and reward setting self.state to new state
        #  probability weights over next state
        dist = self.t(self.state, action)
        # next state drawn from transition function distribution
        next_state = np.random.choice(self.num_states, p=dist) 
        # reward is r(s,a,s')
        reward = self.r(self.state, action, next_state) 
        # current state is updated to hold the new state
        self.state = next_state
        self.increment_counts(reward)
        # the reward and next_state is returned
        return next_state, reward 

    def is_terminal(self):
        return self.absorbing[self.state]

