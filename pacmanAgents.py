# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action)
                      for action in legal]
        scored = [(self.evaluationFunction(state), action)
                  for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


def scoreEvaluation(state):
    return state.getScore()
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<Original File<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>DQN Agent>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""Cite: https://github.com/tychovdo/PacmanDQN
    Tensorflow V1 was used in tychovdo's project.
    It is an obsolete NN platform, so I rewrite it using Pytorch
    tychovdo's setup of the network inspired this project."""
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import torch
import numpy as np
from collections import deque

import torch.nn as nn
import torch.nn.functional as F


"""Change this to T after training"""
model_trained = False

# -----------DQN Utils---------------------
class util_DQN(game.Agent):

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST
			
    def observationFunction(self, state):
        self.terminal = False
        self.observation_step(state)
        return state
		
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 4))
        for i in range(len(actions)):                                        
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    matrix[-1-int(pos[1])][int(pos[0])] = 1
            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        matrix[-1-int(pos[1])][int(pos[0])] = 1
            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        matrix[-1-int(pos[1])][int(pos[0])] = 1
            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    matrix[-1-i][j] = 1 if grid[j][i] else 0
            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1
            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

    def registerInitialState(self, state): # inspects the starting state
        # Reset
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_reward = 0
        self.last_state = None
        self.current_state = self.getStateMatrices(state)
        self.last_action = None
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0
        self.frame = 0

        self.episode_number += 1

    def getAction(self, state):
        move = self.getMove(state)
        legal = state.getLegalActions(0)
        act = Directions.STOP if move not in legal else move
        return act

# set up DQN. 2 CNN(ReLu) + 2 FC
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # inPutNodes = 1024 
        inPutNodes = (9-2) * (18-2) * 64
        self.fc3 = nn.Linear(inPutNodes, 2048)
        self.fc4 = nn.Linear(2048, 4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        return self.fc4(x)

"""----------------------Start DQN--------------------"""
GAMMA = 0.95
batch_size = 32
memory_size = 50000
start_training = 300
TARGET_REPLACE_ITER = 100
epsilon_final = 0.05
epsilon_step = 10000

class PacmanDQN(util_DQN):
    def __init__(self, args):        
        # train using gpu or cpu
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

		# trained or not trained.
        if(model_trained == True):
            print("Model has been trained")
            self.policy_net = torch.load('DQN_policy.pt').to(self.device)
            self.target_net = torch.load('DQN_target.pt').to(self.device)
            self.epsilon = 0.0 # epsilon init value
        else:
            print("Training model")
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)
            self.epsilon = 0.5 # epsilon init value
		
        self.policy_net.double()
        self.target_net.double()
        
        # init optim
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.001, alpha=0.95, eps=0.01)
        
        # init parameters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0

        self.width = args['width']
        self.height = args['height']
        self.num_training = args['numTraining']
        
        self.episode_number = 0
        self.last_score = 0
        self.last_reward = 0.
        
		# memory replay and score databases
        self.replay_mem = deque()
        
		# init Q(s,a)
        self.Q_global = []
		

    def getMove(self, state):
        random_value = np.random.rand() 
         
        if random_value > self.epsilon: # trained 0; training 0.5 
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)
            
			# get Q
            self.Q_found = self.policy_net(temp_current_state)        
            self.Q_found =  self.Q_found.detach().cpu()
            self.Q_found = self.Q_found.numpy()[0]
            self.Q_global.append(max(self.Q_found)) # store max Q
			
			# get best_action - value between 0 and 3
            best_action = np.argwhere(self.Q_found == np.amax(self.Q_found))          
			
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = self.get_direction(best_action[random_value][0])
            else:
                move = self.get_direction(best_action[0][0])
        
        else:
            random_value = np.random.randint(0, 4)
            move = self.get_direction(random_value)

        self.last_action = self.get_value(move)
        return move
           
    def observation_step(self, state):
        if self.last_action is not None:
            # get state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # get reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # ate a ghost 
            elif reward > 0:
                self.last_reward = 10.    # ate food 
            elif reward < -10:
                self.last_reward = -500.  # was eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # didn't eat

            if(self.terminal and self.won):
                self.last_reward = 100.
                self.win_counter += 1
            self.episode_reward += self.last_reward

            # store transition
            transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(transition)
            if len(self.replay_mem) > memory_size:
                self.replay_mem.popleft()
            
            self.train()

        # update counter
        self.local_cnt += 1
        self.frame += 1
		
		# update epsilon
        if(model_trained == False):
            self.epsilon = max(epsilon_final, 1.00 - float(self.episode_number) / float(epsilon_step))

    def final(self, state):
        self.episode_reward += self.last_reward

        self.terminal = True
        self.observation_step(state)

        disp_state = 'Win' if self.won else 'lose'
        disp_qval = max(self.Q_global, default=float('-999'))
        print(f'Episode No. {self.episode_number}\t| {disp_state} | ',end='')
        print(f'Q(s,a) = {disp_qval: .4f}\t| reward = {self.episode_reward:.1f}\t| ',end='')
        print(f'epsilon = {self.epsilon:.4f}')
		
        self.counter += 1
        if(self.counter % 500 == 0) or (self.episode_number % TARGET_REPLACE_ITER == 0):
            torch.save(self.policy_net, 'DQN_policy.pt')
            torch.save(self.target_net, 'DQN_target.pt')
            print("------------------------------------Network Saved--------------------------------------")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        if (self.local_cnt > start_training):
            batch = random.sample(self.replay_mem, batch_size)
            batch_s, batch_r, batch_a, batch_n, batch_t = zip(*batch)
            
            batch_s = torch.from_numpy(np.stack(batch_s))
            batch_s = batch_s.to(self.device)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
            batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
            batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(self.device)
            
            state_action_values = self.policy_net(batch_s).gather(1, batch_a)  # Q(s,a)
            next_state_values = self.target_net(batch_n)  # V(s')
            # expected Q values
            next_state_values = next_state_values.detach().max(1)[0]
            next_state_values = next_state_values.unsqueeze(1)
            expected_state_action_values = (next_state_values * GAMMA) + batch_r
			# calculate loss
            loss_function = torch.nn.SmoothL1Loss()
            self.loss = loss_function(state_action_values, expected_state_action_values)
			#update weights
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()