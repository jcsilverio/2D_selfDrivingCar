#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:36:20 2018

@author: jcsilverio
"""

import numpy as np
import random
import os
import torch #pyTorch
import torch.nn as nn #module to implement neural networks
import torch.nn.functional as F #shortcut
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable #take variable class to make some conversions from tensor to variable that contains a gradientf

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):  # input_size = no. of input neurons (3)
        super(Network, self).__init__()  # use all the tools of nn.Module
        self.input_size = input_size
        self.nb_action = nb_action 
        self.fc1 = nn.Linear(input_size, 30)  # Full Connection - all the neurons of the input layer will be connected to tall the neurons of the hidden layer
        self.fc2 = nn.Linear(30, nb_action) #  all the neurons of the hidden layer will be connected to the neurons of the output layer
        
    def forward(self, state): # Function that will perform forward propagation
        x = F.relu(self.fc1(state))  #represents the hidden neurons
        q_values = self.fc2(x)        #return the output neurons (Q-values)
        return q_values #will return Q values for each possible action (go left, go forward, go right)
    
# Implementing Experience Replay
        
   
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity  # maximum number of transitions we want to have available in memory of events
        self.memory = []  # contains the last 100 transitions
        
    def push(self, event):   # used to append a new event(transition) into memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]  # deleting the oldest transition if over capacity
            

    def sample(self, batch_size):
        # if list = ((1,2,3). (4,5,6)), then zip(*list) = ((1,4), (2,5), (3,6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implement Deep Q Learning
        
class Dqn(): # "deep q network"
      #input_size is the number of dimensions in the vectors that are encoding your input state
      # nb_actions are number of actions the car can make (left/straight/right)
      # gamma is the delay coefficient
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        # the reward window  - sliding window of the revolving mean of the last 100 rewards, used to evaluate the evolution of the AI performance
        self.reward_window = []        
        #neural network for the deep Q learning model
        self.model = Network(input_size, nb_action) 
        self.memory = ReplayMemory(100000) #  taking 100,000 transitions into memory, on which the model will learn
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001 ) #connect Adam optimizer to the neural network; lr is the learning rate
        # variables composing the transition events
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    #action comes from the output of the neural network which depends on the input state
    def select_action(self, state):
        #step 10 ~ 8min mark
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T = 7 Increasing the temperature (T) increases the extremes of probability; T = 0 deactivates the AI
        action = probs.multinomial()   #gives a random draw
        return action.data[0,0]
    
    def learn(self,batch_state, batch_next_state, batch_reward, batch_action):
        #get the outputs of the batch state
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)  #td = temporal difference
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) #back propagate it into the network
        self.optimizer.step() #uses the optimizer to update the weights
    # memory update after reaching a new state    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) #100 is the number of transitions we want our AI to learn from
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) # learning will happen from all these random batches
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: # assuring this element only gets 1000 means of the last 100 rewards
            del self.reward_window[0]
        return action # returns the action that was just played when reaching the new state
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) #assuring the denominator will never be zero which would crash the system
    
    #saving in a python dictionary
    def save(self):
        torch.save({ 'state_dict': self.model.state_dict(), #saves the parameters of the model in this first key, state_dict
                     'optimizer': self.optimizer.state_dict(),
                     }, 'last_brain.pth') 
    
    def load(self):
        if os.path.isfile('last_brain.pth'): # has save file been created?
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) # updates the weights of the model
            self.optimizer.load_state_dict(checkpoint['optimizer']) # update the parameters of the optimizer
            print("done !")
        else:
            print("no checkpoint found...")
        
        
        
        
        
        
        
        