#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:51:36 2018

@author: markus

Everything around the state: reward update, etc.
"""

import environment as env
import numpy as np
import gym
import mxnet as mx

class cartpole_env(env.environment):
    """
    Contains the environment state.
    Most important variables:
        self.state (numpy array) with all the information about the current state
        self.rewards (numpy array) the potential rewards for all actions
        self.isDone ## True if all seats are filled and the game is over.
        
    """
    
    def __init__(self, episodeLength = 1, seqLength = 1, useSeqLength = False):
        """ 
        set up the basic environment
        Args:
            episodeLength (int): how many games constitute one episode?
            seqLength(int): sequence length.
            useSeqLength (bool): whether to return full sequence or only single state.
        """
        self.env = gym.make('CartPole-v0')
        self.validActions = np.array([True, True])
        self.state = 0
        self.netState = [None] * seqLength
        self.useSeqLength = useSeqLength
        self.episodeLength = episodeLength
        self.reset()
        
    def getRawState(self):    
        """
        Returns the state as numpy array
        """
        return self.state
        
    def raw2singleNetState(self):
        """
        converts numpy array to apropriate mxnet nd array
        """
        data = mx.nd.array(self.state)
        data = data.expand_dims(axis = 0)
        data = data.flatten()
        return(data)
    
    def getNetState(self):
        """
        Returns the state as required as input for the a3cNet
        """
        if not self.useSeqLength:
            return self.netState[-1]
        out = mx.nd.zeros(shape = (len(self.netState),) + self.netState[0].shape)
        for i in range(len(self.netState)):
            out[i,:,:] = self.netState[i]
        return out
        
    def getValidActions(self):
        """ 
        Returns a vector with the indices of the valid actions
        """
        return self.validActions
    
    def reset(self):
        """
        resets the environment to starting conditions, i.e. starts a new game
        """
        self.state = self.env.reset()
        self.netState = [self.raw2singleNetState()] * len(self.netState)
        self.is_done = False
        self.is_partDone = False
        self.gameCounter = 0
        self.score = 0
        self.lastReward = -np.Inf
        
    
    def update(self, action):
        """ 
        Updates the environment accoring to an action.
        Stores relevant returns
        args: 
            action (float): the id of the action to be chosen
        """
        
        action = int(action) ## for using as array index
        
        ## throw error on invalid action
        if action >= len(self.validActions):
            raise   ValueError("invalid action: " + str(action))
        if not self.validActions[action]:
            raise   ValueError("invalid action: " + str(action))
        
        tmp = self.env.step(action)
        self.is_partDone = tmp[2]
        if self.is_partDone: self.gameCounter +=1
        if self.is_partDone:
            self.state = self.env.reset()
            self.netState = [self.raw2singleNetState()] * len(self.netState)
        else:
            self.state = tmp[0]
            self.netState = self.netState[:-1] + [self.raw2singleNetState()]
        self.lastReward = tmp[1]
        if self.gameCounter == self.episodeLength:
            self.is_done = True
        self.score += self.lastReward
            
    def isDone(self):
        return self.is_done
    
    def isPartDone(self):
        return self.is_partDone
    
    def getLastReward(self):
        """
        Returns the reward of the last action that was taken
        """
        return self.lastReward   
        
    def getScore(self):
        """
        returns the total score (sum of rewards of all actions taken)
        """
        return self.score / self.episodeLength
        
    