#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:51:36 2018

@author: markus

Everything around the state: reward update, etc.
"""
sys.path.insert(0,'/home/markus/Documents/Nerding/python/dinnerTest/src')

import environment as env
import numpy as np
import mxnet as mx
from state import state
from assignDinnerCourses import assignDinnerCourses
from randomDinnerGenerator import randomDinnerGenerator

from datetime import datetime

class dinner_env(env.environment):
    """
    Contains the environment state.
    Most important variables:
        self.state (numpy array) with all the information about the current state
        self.rewards (numpy array) the potential rewards for all actions
        self.isDone ## True if all seats are filled and the game is over.
        
    """
    def __init__(self, seqLength, useSeqLength = False, nTeams = 20):
        """ 
        set up the basic environment
        actions are reduced to 3: move up, move down and wait.
        Args:
            nBallsEpisode (int): after how many balls does an episode end?
            seqLength(int): sequence length. 0 if no rnn is present.
            useSeqLength(bool): if True, getNetState returns a list, if False, a single state.
        """
        self.env = gym.make('PongDeterministic-v0')
        self.validActions = np.ones(shape = (3))
        self.state = 0
        self.useSeqLength = useSeqLength
        self.netState = [None] * seqLength
        self.nBallsEpisode = nBallsEpisode
        self.reset()
        
    def getRawState(self):    
        """
        Returns the state as numpy array
        """
        return self.state
        
#    def raw2singleNetState(self):
#        """
#        converts numpy array to apropriate mxnet nd array
#        """
#        state = self.state[35:209,0:159] ## crop unnecessary parts
#        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) ## convert to greyscale
#        state = cv2.resize(src = state, dsize = (48,48), interpolation = cv2.INTER_AREA) ## resize
#        state = np.expand_dims(state, 0)
#        state = np.expand_dims(state, 0) ## additional axes for batch and channels
#        state = mx.nd.array(state)
#        return(state)
        
    def raw2singleNetState(self):
        """
        converts numpy array to apropriate mxnet nd array
        """
        state = self.state[34:193,0:159] ## crop unnecessary parts
        origState = state
        ## convert to black and white, assuming constant background color
        bg = (144,72,17)
        state = np.ones_like(origState[:,:,0]) * 255
        state[(origState[:,:,0] == bg[0]) & (origState[:,:,1] == bg[1]) & (origState[:,:,2] == bg[2])] = 0
        state = cv2.resize(src = state, dsize = (80,80), interpolation = cv2.INTER_NEAREST) ## resize
        state = np.expand_dims(state, 0)
        state = np.expand_dims(state, 0) ## additional axes for batch and channels
        state = mx.nd.array(state)
        return(state)
    
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
        self.score = 0
        self.lastReward = -np.Inf
        self.ballsPlayed = 0
        
    
    def update(self, action):
        """ 
        Updates the environment accoring to an action.
        Stores relevant returns
        ATTENTION: game is artificially ended
        args: 
            action (float): the id of the action to be chosen
        """
        
        action = int(action)
        if action >= len(self.validActions):
            raise   ValueError("invalid action: " + str(action))
        if not self.validActions[action]:
            raise   ValueError("invalid action: " + str(action))
        
        ## convert 3 actions to 6 actions of env.
        ## 0, 2 and 3 are all the moves
        if action == 1:
            action = 3
        tmp = self.env.step(action)
        self.state = tmp[0]
        self.netState = self.netState[:-1] + [self.raw2singleNetState()]
        self.lastReward = tmp[1]
        ## whenever reward is -1 or 1, a ball is lost
        if self.lastReward != 0: 
            self.ballsPlayed +=1
            self.is_partDone = True
        else: self.is_partDone = False
        if tmp[2] or self.ballsPlayed == self.nBallsEpisode:
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
        return self.score
        
    