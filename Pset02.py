# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib as mpl

class BigBang:


    def __init__(self,W,L,pe):
        '''
        Creates gridded world/environment in which agent will operate.
        
        Inputs:
            W - INT indicating number of rows in grid (width)
            L - INT indicating number of columns in grid (length)
        
        NOTE: policy is held in two 1-D arrays (one for orientation, one for location).
        All other info is in matrices.
        '''
        assert 0 <= pe <= 1, 'Probability of rotation error (pe) must be: 0 <= pe <= 1'
        self.width = W
        self.length = L
        self.pe = pe
        
        # rewardspace will not include orientation. It only holds a reward for physical location.
        self.rewardspace = np.zeros(W*L)
        
        # Create set of possible actions. Note: actions 0 and 2 are only 
        # permissible if the robot is on the perimeter. All others may occur on
        # the interior and exterior. However, agent is not allowed to move off
        # the grid.
        # Key:
        #   position 0 (translation) - stay(0),up(0),right(3),down(6),left(9)
        #   position 1 (rotation) - Counter-Clockwise(-1),stay(0),Clockwise(1)
        self.actionspace = np.array([[-10,-1],
                                     [-10, 0],
                                     [-10, 1],
                                     [  0,-1],
                                     [  0, 0],
                                     [  0, 1],
                                     [  3,-1],
                                     [  3, 0],
                                     [  3, 1],
                                     [  6,-1],
                                     [  6, 0],
                                     [  6, 1],
                                     [  9,-1],
                                     [  9, 0],
                                     [  9, 1]])
        
        self.validdirections = list(set(self.actionspace[:,0]))
       
        # Initialize an array of matrices, one for each possible translation command
        N_directions = len(self.validdirections)   # Number of travel directions(stay,up,right,down,right)
        self.adjacencytensor = np.zeros([W*L,W*L,N_directions],dtype=int)
        
        # Create offset arrays(stay,up,left,down,right)
        directions = [np.array([0,0]),np.array([-1,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1])]
        N_states = self.length*self.width
        for ID in range(N_states):
            
            coords0 = self.stateid_2_coords(ID)
            coords0 = np.array(coords0[0])
            
            # Determine neighbor state based on translation direction
            idx = 0
            for offset in directions:
                coords_neighbor = coords0 + offset     # Get location of neighbor
                
                # Try indexing the rewardspace using the neighbor's coordinates.
                # If successful, change a 0 to a 1 in the adjacency matrix.
                try:
                    ID_neighbor = self.coords_2_stateid(coords_neighbor)
                    self.adjacencytensor[ID,ID_neighbor,idx] = 1
                except IndexError:
                    pass
                
                idx += 1
        
        # Check that adjacency matrix is symmetric since graph is undirected
        assert np.sum(self.Symmetricity(True)) == 0

        print('Adjacency matrix successfully populated!')
        
        # Populate state IDs on grid
        self.statespace = np.zeros([W,L])
       
        # Create grid with each position containing its ID
        count = 0
        for row,col in zip(range(self.width),range(self.length)):
           self.statespace[row,col] = count
           count += 1
          
        self.policy = np.zeros([12,W*L,2]) # Initial policy: face 0 and stand still

    def coords_2_stateid(self,coords,rewards=[]):
        '''
        Converts a list of nested lists, each containing grid row & col coordinates,
        to a list state IDs. The result is two nested lists, one with state IDs 
        and the other list containing the reward for the state ID in the 
        corresponding position in the first list (state IDs). The output is 
        compatible as an input to GenStateSpace
       
        Inputs:
            coords - LIST of length 2 lists each containing the row & col coordinates
                    of each state that will be assigned a reward
            rewards - LIST (optional) of reward values for each rol-col combination
           
        Outputs:
            output - LIST of two lists. First one containing each state ID, second
                    one containing the reward for each state in the first list.
                    If no rewards are provided, second list is empty.
        '''
        # Determine whether a list of rewards was received
        if len(rewards) == 0:
            norewards = True
        else:
            norewards = False
            assert len(coords) == len(rewards)
        
        # If a non-nested list of length 2 was provided, make it a nested list.
        if len(coords) == 2:
            coords = [coords]
        
        # Determine the state IDs
        indices = []
        for i in range(len(coords)):
           position = coords[i]
           assert len(position) == 2
           
           row = position[0]
           col = position[1]
           
           idx = self.length*row + col%self.width
           indices.append(idx)
       
        # If rewards weren't provided, 2nd list is empty.
        if norewards:
            output = [indices,[]]
        else:
            output = [indices,rewards]
        return output


    def stateid_2_coords(self,IDlist):
        '''
        Converts list of state IDs to lists coordinates for each state ID.
       
        Inputs:
           IDlist - LIST of state IDs
           
        Outputs:
           output - LIST (nested) with lists, each containing the row & col
                    coordinates of each state in IDlist
        '''
        output = []
        if type(IDlist) == int:
            IDlist = [IDlist]
        for i in range(len(IDlist)):
            ID = IDlist[i]
            
            row = ID//self.length
            col = ID%self.length
            assert (row < self.width and col < self.length), "Invalid state ID"
#            assert self.statespace[row,col] == ID   # Double check that the state at the coordinates agrees with the input iD
            output.append([row,col])
        return output


    def AssignRewards(self,state_reward):
        '''
        Assigns reward values to states in grid.
        
        Inputs:
            state_reward - LIST (nested). First list is IDs to which rewards will
                           be assigned. Second is corresponding reward vaules
        
        Outputs:
            None (modifies self.rewardspace)
        '''
        states = state_reward[0]
        rewards = state_reward[1]
        for idx,reward in zip(states,rewards):
           self.rewardspace[idx] = reward

    def ReadPolicy(self,state):
        '''
        Return the optimal policy based on the current state.
        '''
        theta = state[0]
        ID = state[1]
        action = self.policy[theta,ID,:]
        return action

    def GetReward(self,ID):
        return self.rewardspace[ID]

    def IsInterior(self,ID):
        '''
        Determines whether a state lies on the perimeter of the grid.
        
        Inputs:
            ID - INT of state ID to check
            
        Outputs:
            result - BOOL indicating whether state is on perimeter
        '''
        N_neighbors = np.sum(self.adjacencytensor[ID,:])
        
        if N_neighbors == 4:
            result = True
        else:
            result = False
        return result
    
    def IsPerimeter(self,ID):
        '''
        Determines whether a state lies on the perimeter of the grid.
        
        Inputs:
            ID - INT of state ID to check
            
        Outputs:
            result - BOOL indicating whether state is in a corner
        '''
        N_neighbors = np.sum(self.adjacencytensor[ID,:])
        
        if N_neighbors <= 3:
            result = True
        else:
            result = False
        return result

    def Adjacency(self,ID0,ID1,direction):
        '''
        Determines whether two grid squares are adjacent.
        
        Inputs:
            ID0 - INT of current state ID (s)
            ID1 - INT of desired state ID (s')
            direction - INT indicating direction of travel
        
        Outputs:
            result - BOOL indicating true or false for adjacency
        '''
        if not(direction in self.validdirections):
            raise ValueError("Invalid action provided as input! Must be in the set {-10,0,3,6,9}")

        # Determine whether states are adjacent
        idx = self.validdirections.index(direction)
        matrixentry = self.adjacencytensor[ID0,ID1,idx]
        result = bool(matrixentry)
        
        return result

    def Symmetricity(self,hiddenmode=False):
        '''
        Calculates the difference between the adjacency matrix and its transpose.
        Can choose whether to display a figure or return the difference.
        Inputs:
            hiddenmode - BOOL; if False, a figure will be displayed showing the difference image.
                               if True, no figure will be displayed and matrix will be returned.
        Outputs:
            difference - ARRAY; if hiddenmode == True
        '''
        if not(hiddenmode):
            assert np.sum(self.adjacencytensor) != 0, "Adjacency tensor has not been populated yet."
        
        image = np.sum(self.adjacencytensor,axis=2)
        difference = image - np.transpose(image)
        
        if not(hiddenmode):
            mpl.pyplot.subplot(1,2,1)
            mpl.pyplot.title('Adjacency Matrix')
            mpl.pyplot.imshow(image)
            
            mpl.pyplot.subplot(1,2,2)
            mpl.pyplot.title('Original Minus Transpose')
            mpl.pyplot.imshow(difference)
        else:
            return difference
    
    def GetNextState(self,state0,action):
        '''
        Given the current state [theta,ID] and a translation action, determine
        the state that will follow. This does not incorporate rotation error.
        
        Input:
            state0 - LIST (state vector)
            action - LIST (action vector)
        '''
        theta0 = state0[0]
        ID0 = state0[1]
        
        a_rot = action[1]
        
        a_trans = action[0]
        orientation = self.RoundTheta(theta0)        
        # Either flip heading, stay in place, or maintain heading based on command
        if a_trans == 1:
            heading = orientation
        elif a_trans == 0:
            heading = -10
        elif a_trans == -1:
            heading = (orientation - 6)%12
        
        if heading == -10:
            adjacent = self.adjacencytensor[ID0,:,0]
        elif heading == 0:
            adjacent = self.adjacencytensor[ID0,:,1]
        elif heading == 3:
            adjacent = self.adjacencytensor[ID0,:,2]
        elif heading == 6:
            adjacent = self.adjacencytensor[ID0,:,3]
        elif heading == 9:
            adjacent = self.adjacencytensor[ID0,:,4]
        
        if np.sum(adjacent) == 0 and self.IsPerimeter(ID0):
            ID1 = ID0
            theta1 = (theta0 + a_rot)%12
        
        elif np.sum(adjacent) == 0 and not(self.IsPerimeter(ID0)):
            raise ValueError('No adjacent cell found, but agent is not on the perimeter. Something is wrong!')
        
        elif np.sum(adjacent) == 1:
            ID1 = np.where(adjacent == 1)[0][0]
            theta1 = (theta0 + a_rot)%12
        
        else:
            errormsg = ' '.join(['Invalid number of adjacent states found for cell',str(ID0),'direction',str(heading)])
            raise ValueError(errormsg)
        
        state1 = [theta1,ID1]
        return state1
        
    
    def Probability(self,state0,action,state1,debug=False):
        
        theta0 = state0[0]
        ID0 = state0[1]
        
        P_path = [self.pe,1-2*self.pe,self.pe]
        rotations = [-1,0,1]
        theta_error = [(theta0 + d)%12 for d in rotations]
        theta_tilde = [self.RoundTheta(theta) for theta in theta_error] # Call this heading later
        # a_trans is squared as a quick fix. Algorithm was getting confused on directionality.
        ID_tilde = [self.GetNextState([t,ID0],action)[1] for t in theta_tilde]
        
        # Get a list of all possible states reachable from current state using
        # the chosen action.
        allstates = []
        for ID in ID_tilde:
            allstates.append([])
            for rot in rotations:
                a = (theta0 + rot)%12
                b = ID
                allstates[-1].append([a,b])
        
        # Calculate total probability
        TotalProbability = 0
        for i in range(len(allstates)):
            group = allstates[i]
            if state1 in group:
                TotalProbability += P_path[i]
        
        if debug:
            return TotalProbability,allstates
        else:
            return TotalProbability
        
    def RoundTheta(self,theta):
        '''
        Round current orientation to the nearest cardinal direction
        '''
        if theta in [11,0,1]:
            orientation = 0
        elif theta in [2,3,4]:
            orientation = 3
        elif theta in [5,6,7]:
            orientation = 6
        elif theta in [8,9,10]:
            orientation = 9
        
        else:
            errormsg = ' '.join(['Invalid theta received as input:',str(theta)])
            raise ValueError(errormsg)
        
        return orientation
    
class Agent(BigBang):
    def __init__(self,world,state=[0,0],patience=18):
        self.state = state
        self.statehist = [self.state]
        self.errorhist = []
        self.patience = patience
        self.world = world
        self.foundcake = False
        self.stuck = False
        initmsg = ' '.join(['Agent initialized with state',str(self.state)])
        print(initmsg)
    
    def UpdateHist(self,state=[],error=[]):
        if error in [-1,0,1] and state == []:
            self.errorhist.append(error)
            assert (len(self.statehist) == len(self.errorhist)),'When updating the error history, it must be as long as the state history.'

        
        elif len(state) == 2 and error == []:
            self.statehist.append(state)
        
        elif len(state) == 2 and error in [-1,0,1]:
            errormsg = 'Cannot update state history and error history simultaneously since they occur in sequence.'
            raise ValueError(errormsg)
        
        else:
            errormsg = ' '.join(['Invalid state and/or error update! State:',str(state),'Error:',str(error)])
            raise ValueError(errormsg)
        
    def CheckStatus(self):
        if self.world.GetReward(self.state[1]) == np.max(self.world.rewardspace):
            self.foundcake = True
            
            print('Agent found the cake!')
        
        if len(self.statehist) >= self.patience:
            result = True
            N = min(self.patience,len(self.statehist))
            for n in range(N-1,0,-1):
                result = result and (self.statehist[n] == self.statehist[n-1])
            
            if result:
                self.stuck = True
                print('Agent got stuck and gave up!')
        
    def Translate(self,action):
        '''
        Takes an action as an input and updates the agent's state
        
        Inputs:
            action - LIST [{-1,0,1},{-1,0,1}] (state vector)
        '''
        theta0 = self.state[0]
        ID0 = self.state[1]
        
        heading = self.RoundTheta(theta0)
        
        self.state = self.world.GetNextState([heading,ID0],action)
        
        # No state update here because
    
    def Rotate(self,action):
        delta = action[1]
        self.state[0] = int((self.state[0] + delta)%12)
        
        self.UpdateHist(state=self.state,error=[])
        
#    def RotateToHighestReward(self):
#        if self.IsPerimeter(self.state):
#            ID0 = self.state[1]
#            adjacency_neighbors = np.sum(self.adjacencytensor[ID0,:,1:],2)
#            IDs_neighbors = [ID for ID,adj in enumerate(adjacency_neighbors) if adj == 1]
#            Rewards_neighbors = [self.GetReward(ID) for ID in IDs_neighbors]
#        else:
#            raise ValueError('Pure rotation command encountered in a non-perimeter state.')
        
        
    def RotationError(self):
        '''
        Determines whether a rotation error is incurred and then updates the agent's error history.
        
        Inputs:
            None
        
        Outputs:
            None
        '''
        error_eff = 0
        if np.random.rand() <= self.pe:
            self.Rotate(1)
            error_eff += 1
        
        if np.random.rand() <= self.pe:
            self.Rotate(-1)
            error_eff -= 1
        
        self.UpdateHist(state=[],error=error_eff)
    
    def Act(self):
        action_ideal = self.world.ReadPolicy(self.state)
        
        # Agent's state is modified directly by each of these steps
        self.RotationError  # Possibly incur rotation error
        self.Translate(action_ideal) # Move in commanded direction
        self.Rotate(action_ideal)   # Post-translation rotation
        
    def Navigate(self):
        action_proposed = self.world.ReadPolicy(self.state)
        
        while not(self.foundcake) and not(self.stuck):
            state_proposed = self.world.GetNextState(self.state,action_proposed)
            
            if self.world.Probability(self.state,action_proposed,state_proposed) > 0:
                self.Act()
            
            self.CheckStatus()
        
World = BigBang(6,6,0.5)

IDs_red = [0,1,2,3,4,5,6,11,12,17,18,23,24,29,30,31,32,33,34,35]
rewards_red = [-100 for i in range(len(IDs_red))]

IDs_yellow = [8,10,14,16,20,22]
rewards_yellow = [-1 for i in range(len(IDs_yellow))]

IDs_green = [9]
rewards_green = [1]

World.AssignRewards([IDs_red,rewards_red])
World.AssignRewards([IDs_yellow,rewards_yellow])
World.AssignRewards([IDs_green,rewards_green])

agent0 = Agent(World,state=[6,7],patience=18)
agent0.Navigate()
