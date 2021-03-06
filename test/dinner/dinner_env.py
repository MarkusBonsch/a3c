#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:51:36 2018

@author: markus

Everything around the state: reward update, etc.
"""
import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/dinnerTest/src')

import environment as env
import pdb
import numpy as np
import mxnet as mx
from state import state
from assignDinnerCourses import assignDinnerCourses
from randomDinnerGenerator import randomDinnerGenerator
from datetime import datetime

from datetime import datetime

class dinner_env(env.environment):
    """
    Contains the environment state.
    Most important variables:
        self.state (numpy array) with all the information about the current state
        self.rewards (numpy array) the potential rewards for all actions
        self.isDone ## True if all seats are filled and the game is over.
        
    """
    def __init__(self, seqLength, useSeqLength = False, 
                 nTeams = 10
                 ,padSize = 10
                 ,shuffleTeams = False
                 ,restrictValidActions = True
                 ,centerAddress={'lat':53.551086, 'lng':9.993682}
                 ,radiusInMeter=5000
                 ,dinnerTime = datetime(2020, 7, 1, 20)
                 ,wishStarterProbability=0.3
                 ,wishMainCourseProbability=0.4
                 ,wishDessertProbability=0.3
                 ,rescueTableProbability=0.5
                 ,meatIntolerantProbability=0
                 ,animalProductsIntolerantProbability=0
                 ,lactoseIntolerantProbability=0
                 ,fishIntolerantProbability=0
                 ,seafoodIntolerantProbability=0
                 ,dogsIntolerantProbability=0
                 ,catsIntolerantProbability=0
                 ,dogFreeProbability=0
                 ,catFreeProbability=0
                 ,travelMode = 'simple'):
        """ 
        set up the basic environment
        """
        self.travelMode = travelMode
        self.dinnerTime = dinnerTime
        self.padSize = padSize
        self.shuffleTeams = shuffleTeams
        self.restrictValidActions = restrictValidActions
        self.rawGen = randomDinnerGenerator(numberOfTeams=nTeams
                                    ,centerAddress=centerAddress
                                    ,radiusInMeter=radiusInMeter
                                    ,wishStarterProbability=wishStarterProbability
                                    ,wishMainCourseProbability=wishMainCourseProbability
                                    ,wishDessertProbability=wishDessertProbability
                                    ,rescueTableProbability=rescueTableProbability
                                    ,meatIntolerantProbability=meatIntolerantProbability
                                    ,animalProductsIntolerantProbability=animalProductsIntolerantProbability
                                    ,lactoseIntolerantProbability=lactoseIntolerantProbability
                                    ,fishIntolerantProbability=fishIntolerantProbability
                                    ,seafoodIntolerantProbability=seafoodIntolerantProbability
                                    ,dogsIntolerantProbability=dogsIntolerantProbability
                                    ,catsIntolerantProbability=catsIntolerantProbability
                                    ,dogFreeProbability=dogFreeProbability
                                    ,catFreeProbability=catFreeProbability,
                                    checkValidity = False)
        self.validActions = np.zeros(shape = (self.padSize))
        self.state = 0
        self.useSeqLength = useSeqLength
        self.netState = [None] * seqLength
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
        state = mx.nd.array(self.state).reshape(-1) ## 1d array
        state = mx.nd.expand_dims(state, 0) ## dummz axes for batch and channel
        state = mx.nd.expand_dims(state, 0)
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
    
    def reset(self, initState = None):
        """
        resets the environment to starting conditions, i.e. starts a new game
 
        """
        if initState is None:
            rawData = self.rawGen.generateDinner()
            assigner = assignDinnerCourses(rawData[0], rawData[1])
            dinnerAssigned = assigner.assignDinnerCourses(random = False)
            self.env = state(data = dinnerAssigned, 
                             dinnerTime = self.dinnerTime, 
                             travelMode = self.travelMode, 
                             padSize = self.padSize,
                             shuffleTeams = self.shuffleTeams)
            self.env.initNormalState()
        else:
            self.env = initState
        self.validActions = 0 * self.validActions
        self.validActions[self.env.getValidActions()] = 1
        if not self.restrictValidActions:
            self.validActions[:] = 1
        self.state = self.env.getState()
        self.netState = [self.raw2singleNetState()] * len(self.netState)
        self.is_done = False
        self.is_partDone = False # is set to True if all non-rescue teams have been assigned.
        self.score = 0
        self.lastReward = -np.Inf
    
    def update(self, action):
        """ 
        Updates the environment according to an action.
        Stores relevant returns
        args: 
            action (float): the id of the action to be chosen
        """
        
        action = int(action)
        isValidAction = action in self.env.getValidActions()
        if not isValidAction:
            if self.restrictValidActions:
                raise   ValueError("invalid action: " + str(action))
            else:
                self.is_partDone = True
                self.is_done = True
                self.lastReward = - self.env.alphaMeet * self.env.getMissingTeamScore()[1]
                self.score = self.env.getScore()
                return None
        self.lastReward = self.env.getRewards()[action] ## important before update
        self.env.update(action)
        self.score = self.env.getScore()
        if self.env.isDone():
            ## check if it is fully done or rescue Tables need to be assigned.
            if self.isPartDone(): ## We were already in rescue mode
                self.is_done = True
            else: ## we still have to do the rescue mode
                self.env.initRescueState()
                self.is_partDone = True
                self.is_done = self.env.isDone()
        if self.restrictValidActions:
            self.validActions *= 0
            self.validActions[self.env.getValidActions()] = 1
        self.state = self.env.getState()
        self.netState = self.netState[:-1] + [self.raw2singleNetState()]
            
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
        
    