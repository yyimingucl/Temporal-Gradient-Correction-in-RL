from environment.simulation import Simulation


class MazeSim(Simulation):
    """
    A class to simulate an environment based on an Maze. Inherits the run()
    function from the Simulation class.
    """
    def __init__(self, model): 
        
        super().__init__(
                model.state_names, model.action_names)
        self.initial_state = model.initial
    
        self.num_states = model.num_states*2 # as we have 2 situations each cell 
                                             # (withkey and withoukey)
        self.shape = model.shape
        self.num_actions = model.num_actions
        self.maze = model.maze
        self.action = model.DIRECTIONS
        self.finish = False
        self.walls = model.walls    
        self.door = model.door
        self.key = model.key 
        self.states = model.states
        self.neighbours = model.neighbours
        self.is_finding_key = True
        self.state = self.reset()
        

    def reset(self):
        # Initialises state from initial distribution and then outputs state 
        self.state = self.initial_state
        self.reward = 0
        self.finish = False
        self.is_finding_key = True
        
        return self.state
        
    
    def next(self,action):
        current = self.states.index(self.state)
        explorer = self.states[self.neighbours[current, action]]
        if self.fall_outside(self.state,action):
            self.state = self.initial_state
            self.reward -= 0.1
            
            
        elif explorer == self.state:
             self.reward -= 0.1
             self.state = explorer
            
        else:       
            if self.is_finding_key:                
                if explorer == self.key: 
                   self.reward += 1.
                   self.is_finding_key = False
                else:
                    self.reward -= 0.1 
    
            if not self.is_finding_key: 
                if explorer == self.door:
                    self.reward += 1.
                    self.finish = True
                else:
                    self.reward -= 0.1 
            
            self.state = explorer
                
            
        
        return self.state, self.reward  #first return is next state

    def is_terminal(self):
        return self.finish
    
    def fall_outside(self,loc,action):
        row,col = self.shape
        i,j = loc
        if action == 0:
            i-=1
        elif action == 1:
            j+=1
        elif action == 2:
            i+=1
        elif action == 3:
            j-=1
        
        if i<0 or j<0 or i>row or j>col or (i,j) in self.walls:
            return True
        return False


class MazeSim_Features(MazeSim):
    def __init__(self, model, feature=None):
        """
        parameters
        ----------
        1.model is the maze_grid which stores the basic information about the maze
        2.valid_action: indicate valid actions for each state 
        """
        # initilise via the superclass
        if feature == "onehot":
            self.rep_function = model.get_feature_mapping_onehot()
            
        elif feature == "xy":
            self.rep_function = model.get_feature_mapping_xy()
            
        elif feature == "tiling":
            self.rep_function = model.get_feature_mapping_tiling()
        elif feature == "onehot_tiling":
            self.rep_function = model.get_feature_onehot_tiling()
        else:
            raise TypeError("unknown type of feature mapping")
            
        super().__init__(model)
        
        self.initial = self.reset()
        
                
    
    def reset(self):
        # initialise the state
        _ = super().reset()
        # but instead of returning the state, we return the representation
        return self.rep_function(self.state, self.is_finding_key)
    

    def next(self, action):
        # use the superclass next function to evolve the system
        next_state, reward = super().next(action)
        
        # states are now vectors of features
        features = self.rep_function(next_state, self.is_finding_key)
        return features, reward