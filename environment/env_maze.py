import numpy as np
from environment.simulation import Simulation
import math

##########################
#### MAZE_BASE CLASS #####
##########################

# This class takes a list of lists which contain O, 1, 2 and 3 and 
# a list which contains all possible actions.
# We start with a matrix with of 0s, 1s, 2s and a 3:
#     0 = wall
#     1 = blank
#     2 = door/start point
#     3 = key
# The possible actions are 4: up,  right, down and left.
# Instances of this class will have some attributes for the maze:
#     num_actions 
#     self.initial = coordinate of initial position
#     self.door = coordinate of door
#     self.key = coordinate of key
#     self.walls = list of coordinates of walls
# list_of_names method gives names to each cell:
#     (0,0) = s0, (0,1) = s1, (0,2) = s2
#     useful if we need to visualize the grid
#

class maze_base(object):
    """
    define the important components of maze
    """
    def __init__(
            self, maze, action):

        self.action = action
        self.maze = np.array(maze)
        self.rows, self.cols = self.maze.shape
        
        self.state_names = self.list_of_names('s', self.rows*self.cols)
        self.num_actions = len(action)
        self.action_names = action
        sd_row,sd_col = np.where(self.maze == 2)
        
        # coordinates of door and key
        self.initial = (sd_row[0],sd_col[0])
        self.door = (sd_row[1], sd_col[1])
        self.key = (np.where(self.maze ==3)[0][0],
                    np.where(self.maze ==3)[1][0])
        
        # list of coordinates of the wall cells
        wall_row,wall_col = np.where(self.maze == 0)
        self.walls = [(i,j) for i,j in zip(wall_row,wall_col)]
    
    
    # this generates names for the coordinates:
    # EX. (0,0) will have name s0, (0,1) --> s1, (0,2) --> s2
    def list_of_names(self, base_name, n):
        """
        A helper method that converts a base name and number of elements n
        into a list of names
        """
        # number of padded zeros
        num_digits = np.ceil(np.log10(n-1))
        fmt = base_name + '%0'+str(num_digits)+'d'
        return [ fmt % i for i in range(n) ]


##########################
#### MAZE_GRID CLASS #####
##########################

# This class takes the list representing the maze as described above and 
# a list which contains all possible actions and inherits all attributes 
# and methods from maze_base
# New instances of this class will have some attributes for the maze:
# Importan ones are:
#     self.locs = coordinates of all admissable locations for the agent,
#                 so everywhere except where the walls are 
#     self.neighbours = matrix that has as many rows as available cells. Each column
#                       correspond to an action and if entry j,i is k this means 
#                       that from the jth available cell you will move to kth available
#                       cell if you take action i.
# To obtain the 2 above methods get_topology, valid_location, get_neighbour_list and
# get_neighbour are used


class maze_grid(maze_base):
    """
    Creates objects that represent grid-worlds
    """
    # the cardinal points of the compass encoded as integers
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    DIRECTIONS = [ NORTH, EAST, SOUTH, WEST ]
    def __init__(self, maze, action):
        """
        The constructor for a grid-world. It is recommended to use the 
        class method build to build grid worlds.
        """
        super(maze_grid,self).__init__(maze, action)
        # the shape (height, width) of the grid-world
        self.shape = (self.rows, self.cols)
        locs, neighbours = self.get_topology(self.shape,self.walls)
        # the mapping from states to locations (i,j,k) k indicates to the key
        # 0 means without the key and 1 means with the key     
        self.states = locs
        # mapping from each state to the 4 neighbouring states (aka topology)
        self.neighbours = neighbours
        self.num_states = len(self.states)

        
        
    @classmethod
    def get_topology(cls, shape, walls):
        """ 
        shape - (width, height) of grid 
        walls - a list of (i,j) pairs corresponding to grid positions that
            are not valid states
        """
        height, width = shape
        locs = []
        all_neighbour_locs = []
        index = 1
        for i in range(height):
            for j in range(width):
                loc = (i,j)
                # for every potential grid-position we only add it to the list
                # of locs if it is a valid location
                if cls.valid_location(loc, shape, walls):
                    locs.append(loc)
                    all_neighbour_locs.append(
                        cls.get_neighbour_list(loc, shape, walls))
        # translate neighbour lists from locations to states
        num_states = len(locs)
        num_actions = len(cls.DIRECTIONS)
        neighbours = np.empty((num_states,4), dtype=int)
        for s, these_neighbours in enumerate(all_neighbour_locs):
            for dirn in range(num_actions):
                neighbour_loc = these_neighbours[dirn]
                # find index of neighbour location
                # to turn location into a state number
                neighbour_state = locs.index(neighbour_loc)
                # insert into neighbour matrix
                neighbours[s,dirn] = neighbour_state
        #
        return locs, neighbours #only contain neithbours of valid loc
    @classmethod
    def valid_location(cls, loc, shape, walls):
      """
      Evaluates whether a grid location is valid for this grid world

      parameters
      ----------
      loc - pair of x, y grid coordinates
      shape - (width, height) of grid world
      obstacles - a list of grid coordinates that are forbidden locations
      """
      x, y = loc
      max_x, max_y = shape
      if x<0 or y<0 or x>=max_x or y >= max_y :
        return False
      elif loc in walls:
        return False
      return True
  
    @classmethod
    def get_neighbour_list(cls, loc, shape, walls):
        """
        Given a location in our grid-world, this determines the ordered list of
        neighbouring locations (one per direction). 
    
        parameters
        ----------
        loc - (i,j) grid location
        shape - shape of the grid-world
        walls - list of grid-position that are not valid due to obstacles
    
        returns
        -------
        neighbour_list - neighbouring grid-positions, or loc when neighbour is
            invalid
        """
        neighbour_list = []
        for dirn in cls.DIRECTIONS:
            this_neighbour = cls.get_neighbour(loc, dirn, shape, walls)
            neighbour_list.append(this_neighbour)
        return neighbour_list
    
    @classmethod
    def get_neighbour(cls, loc, dirn, shape, walls):  
      """
      Given a location in our grid-world and a direction, this determines the
      neighbouring location if there is one, if not it returns the input
      location.

      Used to determine where agent ends up if moving in a given direction
      from a given location.

      parameters
      ----------
      loc - (i,j) grid location
      dirn - NORTH, EAST, SOUTH or WEST (cardinal DIRECTIONS)
      shape - shape of the grid-world
      walls - list of grid-position that are not valid due to walls

      returns
      -------
      neighbour - neighbouring grid-position in the direction specified or loc
          if the direction is invalid
      """
      i, j = loc
      if dirn == cls.NORTH:
          target = (i-1,j)
      elif dirn == cls.EAST:
          target = (i,j+1)
      elif dirn == cls.SOUTH:
          target = (i+1,j)
      elif dirn == cls.WEST:
          target = (i,j-1)
      else:
        raise ValueError("Unrecognised direction %d" % dirn)
      # target refers to the grid position that the agent would move to if it
      # were a valid location, if it is not valid then return the present
      # location
      if cls.valid_location(target, shape, walls):
          neighbour = target
      else:
          neighbour = loc
      return neighbour
  
##### DISCLAIMER
##### I know that the tiling example in the slide had 3 tilings (red, blue and green) and not all cells where coverd by
##### any of them (for example cell (0,11)) but in reality we want a set if tilings that altogether covers the whole grid
##### so if we consider a 11x11 grid and 3 tilings with tile size 3x3 and starting positions (0,0),(2,1),(1,2) we will need
##### 16 tiles for each tiling and not 9 (or at least some tiling arrangment that guarantess that all 
##### cells in the grid are covered), so we should have tilings that cover even more than what is needed.
##### Depending on the starting point though some cells will never be covered by some tiles.
# CONSIDER THE FOLLOWING MAZE:
# maze =[[2,1,1,3],
#        [0,0,1,0],
#        [1,1,1,0],
#        [0,0,1,2]]
# When we initialise the 3 tiles starting at [0,0] (red) ,[2,1] (green), [1,2] (blue)
# we will need 4 3x3 tiles in the red tile, 1 3x3 tile in the green tile, and 1 3x3 tile in the blue tile
# Green and blue tiles do not cover everything because of their starting point but still they cover as much as they can from there.

    def get_feature_mapping_tiling(
            self,tile_size=3, starting_positions=[[0,0], [2,1], [1,2]]):
        locs = np.array(self.states)
        # In the following we round up result of width of grid divided by tile_size, it considers the fact that 
        # our tiling has to completely cover the grid, that is why we round up to the next integer.
        # For the slide example, tile_cols = math.ceil(4/3) = 2.
        tile_cols = math.ceil(self.shape[0]/tile_size)
        tile_rows = math.ceil(self.shape[1]/tile_size)
        state_indices = np.zeros(
            (self.num_states, len(starting_positions)), dtype=int)
        xs = np.array(locs)[:,0] # selects all x-coordinates from the list of available states
        ys = np.array(locs)[:,1]
        for tiling, (xshift, yshift) in enumerate(starting_positions):
            # selects all y-coordinates from the list of available states
            y_shifted = (ys-yshift)
            x_shifted = (xs-xshift)

            # print('x_shifted for tiling {} are {}'.format(tiling, x_shifted))
            # print('y_shifted for tiling {} are {}\n'.format(tiling, y_shifted))
            
            indices = []
            for i in range(len(y_shifted)):
                # this first condition is to make sure that cells not covered by tiles will have 
                # representation 0 for that tiling 
                if (y_shifted[i] < 0)  | (x_shifted[i] < 0):
                    indices.append(-tile_cols*tile_rows*len(starting_positions)-10)
                else:
                    indices.append(tile_cols * ((ys[i]-yshift)/tile_size).astype(int) \
                    + ((xs[i]-xshift)/tile_size).astype(int))
                    # print('index in y is {} for tile {}'.format(tile_cols * ((ys[i]-yshift)/tile_size).astype(int), tiling))
                    # print('index in x is {} for tile {}'.format(((xs[i]-xshift)/tile_size).astype(int), tiling))
                    # print('final index is {} for tile {}\n'.format(tile_cols * ((ys[i]-yshift)/tile_size).astype(int) \
                    # + ((xs[i]-xshift)/tile_size).astype(int), tiling))
            # print('indices for tiling {} are {} \n'.format(tiling, indices))
            state_indices[:,tiling] = indices
        # print(state_indices)
        
        subfeature_lengths = np.max(state_indices, axis=0) + 1 #聽tells us how many tiles we actually need for each of the 3 tilings
                                                               #聽in our case np.max(state_indices, axis=0) = [3,0,0] + 1 --> [4,1,1]
        # print('number of tiles per tiling that we really need to cover available locs {}'.format(subfeature_lengths))
        features = np.zeros(
            (self.num_states, 1+np.sum(subfeature_lengths)),dtype=int)
        # the zeroth feature is the constant term, that is why there is the 1+ in the line above 
        features[:,0] = 1
        # start counting tiles at 1
        min_index = 1
        for tiling, sub_len in enumerate(subfeature_lengths):
            # for every state get the local index and add the min_index
            these_indices = min_index+state_indices[:,tiling]
            # print('index for tiling independent of the previous {}'.format(state_indices[:,tiling]))
            # print('index for tiling including the previous {}\n'.format(these_indices))
            these_indices[these_indices<0] = 0
            features[np.arange(self.num_states), these_indices] = 1
            # print(features)
            min_index += sub_len
        key_zeros = features.shape[1] - 1
        # print(key_zeros)
        #print(features[:,1:])
    

        # without_key_vector = np.array([[1,0] for i in range(len(self.states))])
        # with_key_vector = np.array([[0,1] for i in range(len(self.states))])
        key_vector = np.array([[0]*key_zeros for i in range(len(self.states))])
        const_vector = np.array(features[:,0]).reshape((len(self.states), 1))
        feature_key_with = np.hstack((const_vector, key_vector, features[:,1:]))
        feature_key_without = np.hstack((const_vector, features[:,1:], key_vector))
        # feature_key_without = np.hstack((features, without_key_vector))
        # feature_key_with = np.hstack((features, with_key_vector))
        # This just creates matrices for the state representation.
        # In the end feature_key is a dictionary that contains 2 keys: without_key and with_key
        # each of the these keys has a dictionary that has the cooridnates of the available
        # states as keys.
        # each of the state keys has a list that contains 0 and 1s for the feature representation
        # EX. feature_key["without_key"][(0,0)] returns the feature rep when the agent is at the origin
        # and has't yet found the key
        
        feature_key = {"without_key":{},
                            "with_key":{}}
        
        for name, rep_type in zip(["without_key","with_key"],
                             [feature_key_without,feature_key_with]):
            for i, state in enumerate(self.states):
                feature_key[name][state] = rep_type[i]
        # print(feature_key)
        # function to get the tiling rep for each state (key dependent)
        def tiling_mapping(state, is_finding_key = True):
            if is_finding_key:
                return feature_key["without_key"][state]
            else:
                return feature_key["with_key"][state]
        #print(feature_key)
        return tiling_mapping 
        #1. is a function 2. is a table
        
    def get_feature_mapping_xy(self):
        
        feature_key = {"without_key":{},"with_key":{}}
        without_key_vector = np.array([[0,1] for i in range(len(self.states))])
        with_key_vector = np.array([[1,0] for i in range(len(self.states))])
        
        rep=[]
        for state in self.states:
            x,y = state
            rep.append([1,x,y]) #1 is the constant term
            
        feature_key_without = np.hstack((rep,without_key_vector))
        feature_key_with = np.hstack((rep,with_key_vector))
        
        for name, rep_type in zip(["without_key","with_key"],
                             [feature_key_without,feature_key_with]):
            for i, state in enumerate(self.states):
                feature_key[name][state] = rep_type[i]
        
        def xy_mapping(state, is_finding_key = True):
            if is_finding_key:
                return feature_key["without_key"][state]
            else:
                return feature_key["with_key"][state]
            
        return xy_mapping
    
    
    def get_feature_mapping_onehot(self):
        
        feature_key = {"without_key":{},"with_key":{}}
        without_key_vector = np.array([[0,1] for i in range(len(self.states))])
        with_key_vector = np.array([[1,0] for i in range(len(self.states))])

        rep=[]
        for index in range(len(self.states)):
            feature = np.zeros(len(self.states))
            feature[index] = 1.
            rep.append(feature) #1 is the constant term
        feature_key_without = np.hstack((rep,without_key_vector))
        feature_key_with = np.hstack((rep,with_key_vector))

        for name, rep_type in zip(["without_key","with_key"],
                             [feature_key_without,feature_key_with]):
            for i, state in enumerate(self.states):
                feature_key[name][state] = rep_type[i]
                
        def onehot_mapping(state, is_finding_key = True):
            if is_finding_key:
                return feature_key["without_key"][state]
            else:
                return feature_key["with_key"][state]
            
        return onehot_mapping
    
    
    def get_feature_mapping_neighbours_binary(self, check=True):
        states = np.array(self.states)
        # access grid is a boolean matrix of the grid world with a margin
        # width 1 all the way around corresponding to the inaccessible 
        # region round the edge (all False). Within the grid proper
        # accessible cells are True and inaccessible (obstacles) are False
        # access_grid for 4x4 maze is:
        #  111111
        #  100001
        #  100001
        #  100001
        #  100001
        #  111111
        access_grid = np.ones((self.shape[0]+2,self.shape[1]+2), dtype=int)
        access_grid[1:-1,1:-1] = np.zeros(self.shape, dtype=int)
        for i,j in self.walls:
            access_grid[i+1, j+1] = True # +1 are because we need to consider the surrounding 1s in access_grid
        # need mask to remove the middle square of the flattened feature
        # (which is accesible by definition)
        mask = np.ones(9,dtype=bool)
        mask[4] = False
        # for each available state the row of features will contain 8 entries all 0s or 1s, based on wether
        # there are obstacles in the 8 adjacent locations (lecture 7 slide 45)
        features = np.empty((self.num_states, 9), dtype=int)
        features[:,0] = 1
        for s, (i, j) in enumerate(states): 
            # 3x3 grid of accessibility for cell i,j
            neighbours = access_grid[i:i+3, j:j+3] # this is a list of length 9 of 0s or True boolen variables
            features[s,1:] = neighbours.flatten()[mask] # the mask filter is there to only select all states except state s 
        if check:  
            print(features)
       
        feature_key = {"without_key":{},"with_key":{}}
        without_key_vector = np.array([[1,0] for i in range(len(self.states))])
        with_key_vector = np.array([[0,1] for i in range(len(self.states))])
        
        feature_key_without = np.hstack((features, without_key_vector))
        feature_key_with = np.hstack((features, with_key_vector))
        # This just creates matrices for the state representation.
        # In the end feature_key is a dictionary that contains 2 keys: without_key and with_key
        # each of the these keys has a dictionary that has the cooridnates of the available
        # states as keys.
        # each of the state keys has a list that contains 0 and 1s for the feature representation
        # EX. feature_key["without_key"][(0,0)] returns the feature rep when the agent is at the origin
        # and has't yet found the key
        
        feature_key = {"without_key":{},
                            "with_key":{}}
        
        for name, rep_type in zip(["without_key","with_key"],
                             [feature_key_without,feature_key_with]):
            for i, state in enumerate(self.states):
                feature_key[name][state] = rep_type[i]
        # print(feature_key)
        # function to get the tiling rep for each state (key dependent)
        def neighbours_binary_mapping(state, is_finding_key = True):
            if is_finding_key:
                return feature_key["without_key"][state]
            else:
                return feature_key["with_key"][state]
            
        return neighbours_binary_mapping 
    
    def get_feature_onehot_tiling(
            self,tile_size=3, starting_positions=[[0,0], [2,1], [1,2]]):
        locs = np.array(self.states)
        tile_cols = math.ceil(self.shape[0]/tile_size)
        tile_rows = math.ceil(self.shape[1]/tile_size)
        state_indices = np.zeros(
            (self.num_states, len(starting_positions)), dtype=int)
        xs = np.array(locs)[:,0] # selects all x-coordinates from the list of available states
        ys = np.array(locs)[:,1]
        for tiling, (xshift, yshift) in enumerate(starting_positions):
            y_shifted = (ys-yshift)
            x_shifted = (xs-xshift)

            indices = []
            for i in range(len(y_shifted)):
                if (y_shifted[i] < 0)  | (x_shifted[i] < 0):
                    indices.append(-tile_cols*tile_rows*len(starting_positions)-10)
                else:
                    indices.append(tile_cols * ((ys[i]-yshift)/tile_size).astype(int) \
                    + ((xs[i]-xshift)/tile_size).astype(int))
                    
            state_indices[:,tiling] = indices
        
        subfeature_lengths = np.max(state_indices, axis=0) + 1 #聽tells us how many tiles we actually need for each of the 3 tilings
                                                               #聽in our case np.max(state_indices, axis=0) = [3,0,0] + 1 --> [4,1,1]
        features = np.zeros(
            (self.num_states, 1+np.sum(subfeature_lengths)),dtype=int)
        features[:,0] = 1
        # start counting tiles at 1
        min_index = 1
        for tiling, sub_len in enumerate(subfeature_lengths):
            these_indices = min_index+state_indices[:,tiling]
            these_indices[these_indices<0] = 0
            features[np.arange(self.num_states), these_indices] = 1
            min_index += sub_len
        
        # onehot representation
        rep=[]
        for index in range(len(self.states)):
            feature = np.zeros(len(self.states))
            feature[[index]] = 1.
            rep.append(feature) #1 is the constant term
        key_zeros = features.shape[1]+len(rep)
        key_vector = np.array([[0]*key_zeros for i in range(len(self.states))])
        feature_key_with = np.hstack((key_vector, features,rep))
        feature_key_without = np.hstack((features,rep, key_vector))
        
        feature_key = {"without_key":{},
                            "with_key":{}}
        
        for name, rep_type in zip(["without_key","with_key"],
                             [feature_key_without,feature_key_with]):
            for i, state in enumerate(self.states):
                feature_key[name][state] = rep_type[i]

        # function to get the rep for each state (key dependent)
        def onehot_tiling(state, is_finding_key = True):
            if is_finding_key:
                return feature_key["without_key"][state]
            else:
                return feature_key["with_key"][state]
        print('This is what the representation now looks like:')
        print(feature_key)
        return onehot_tiling


        
    