"""
Created on Tue Mar 27 17:51:36 2018

@author: markus

Everything around the state: reward update, etc.
"""
import sys
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/dinnerTest/src')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')

import environment as env
import pdb
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
    def __init__(self, seqLength, useSeqLength = False, 
                 # each dinner is generated with a random number of teams nTeams
                 # between nMinTeams and nMaxTeams. Only nTeams divisible by 3 are allowed
                 # to make sure everything resolves easily
                 # padSize automatically corresponds to nMaxTeams
                 nMinTeams = 24 ## 
                 ,nMaxTeams = 24
                 ,padSize = 24 # only relevant if nMinTeams = nmaxTeams, i.e. if a certain teamSize is required
                 ,shuffleTeams = False
                 ,restrictValidActions = False
                 ,centerAddress={'lat':53.551086, 'lng':9.993682}
                 ,radiusInMeter=5000
                 ,dinnerTime = datetime(2020, 7, 1, 20)
                 ,wishStarterProbability=1/3
                 ,wishMainCourseProbability=1/3
                 ,wishDessertProbability=1/3
                 ,rescueTableProbability=1
                 ,meatIntolerantProbability=0
                 ,animalProductsIntolerantProbability=0
                 ,lactoseIntolerantProbability=0
                 ,fishIntolerantProbability=0
                 ,seafoodIntolerantProbability=0
                 ,dogsIntolerantProbability=0
                 ,catsIntolerantProbability=0
                 ,dogFreeProbability=1
                 ,catFreeProbability=1
                 ,travelMode = 'simple'):
        """ 
        set up the basic environment
        """
        self.travelMode = travelMode
        self.dinnerTime = dinnerTime
        if nMinTeams > nMaxTeams: raise ValueError("nMinTeams is larger than nMaxTeams")
        if nMaxTeams > padSize: raise ValueError("nMaxTeams is larger than padSize")
        self.padSize = padSize
        self.shuffleTeams = shuffleTeams
        self.restrictValidActions = restrictValidActions
        # now we have to generate randomDinnerGenerators for all possible teamSizes
        possibleTeamSizes = np.arange(nMinTeams, nMaxTeams+3, 3)
        self.rawGen = []
        for teamSize in possibleTeamSizes:
            self.rawGen.append(randomDinnerGenerator(numberOfTeams=teamSize                                    
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
                                    checkValidity = False))
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
    
    def getVariableIndices(self):
        """Returns the index positions of important variables in the raw state (dict)
        """        
        return self.env.stateIndices
        
    def raw2singleNetState(self):
        """
        converts numpy array to apropriate mxnet nd array
        """
        state = mx.nd.array(self.state).reshape(-1) ## 1d array
        ## add active course information in the end
        activeCourseInfo = mx.nd.zeros(3)
        if self.env.isDone(): # if it is done, set the activeCourse to 3 for desert.
            activeCourseInfo[2] = 1
        else:
            activeCourseInfo[self.env.activeCourse - 1] = 1
        state = mx.nd.concat(state, activeCourseInfo, dim = 0)
        state = mx.nd.expand_dims(state, 0) ## dummy axes for batch and channel
        state = mx.nd.expand_dims(state, 0)
        return(state)
        
    def raw2singleNetStateValid(self):
        """
        test with only valid actions as state
        """
        state = mx.nd.array(self.validActions) * 0
        if(len(self.env.getValidActions()) == 0):
            return state
        state[self.env.getValidActions()] = 1
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
            # random sample a teamSize for the current dinner
            # i.e. pick a randomDinnerGenerator from the list
            thisSize = np.random.randint(len(self.rawGen))
            rawData = self.rawGen[thisSize].generateDinner()
            assigner = assignDinnerCourses(dinnerTable = rawData[0])
            dinnerAssigned = assigner.assignDinnerCourses(random = False)
            self.env = state(data = dinnerAssigned,
                             finalPartyLocation = rawData[1],
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
        self.stateIndices = self.env.stateIndices ## Lookup table for variable positions in the input
        self.stateIndices["activeCourse"] = [x + self.netState[0].shape[2] - 3 for x in [0,1,2]] # add activeCourse Info

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
                self.lastReward = -1
                self.score += self.lastReward
                self.is_done = True
                self.is_partDone = True
                self.env.done = True
                return None
        ## if we reach here, we had a valid action!
        self.lastReward = 1

#        # add number of new teams met to reward
#        self.lastReward += self.env.getNewPersonsMet()[action]
        self.env.update(action)
        ## additional reward if the game is over
        if self.env.isDone():
            self.lastReward += 4
        self.score += self.lastReward
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
        
    