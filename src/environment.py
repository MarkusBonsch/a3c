# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:03:38 2018

@author: markus
"""

from abc import ABCMeta, abstractmethod

class environment:
    """ 
    abstract class that defines the interface for the a3c algorithm.
    Any real environment needs to inherit from here.
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getRawState(self):    
        """
        Returns the unprocessed state as numpy array
        """
        raise NotImplementedError("Please overload getRawState method when implementing environment.")    

    @abstractmethod
    def getNetState(self):    
        """
        Returns the state as required as input for the a3cNet
        """
        raise NotImplementedError("Please overload getNetState method when implementing environment.")    

    @abstractmethod
    def getValidActions(self):    
        """
        Returns a vector with indices of the valid actions
        """
        raise NotImplementedError("Please overload getValidActions method when implementing environment.")    

    @abstractmethod
    def reset(self):
        """
        resets the environment to starting conditions, i.e. starts a new game
        """
        raise NotImplementedError("Please overload reset method when implementing environment.")
        
    @abstractmethod 
    def update(self, action):
        """
        Updates the environment according to a chosen action
        """
        raise NotImplementedError("Please overload update method when implementing environment.")    
    
    @abstractmethod
    def isDone(self):
        """
        Returns bool that indicates whether the game is over
        """
        raise NotImplementedError("Please overload isDone method when implementing environment.")   
        
    @abstractmethod
    def isPartDone(self):
        """
        Returns bool that indicates whether a part of the game is done,
        e.g. one ball in Pong.
        """
        raise NotImplementedError("Please overload isPartDone method when implementing environment.")   
        
    @abstractmethod
    def getLastReward(self):    
        """
        Returns the reward of the last executed action
        """
        raise NotImplementedError("Please overload getRewards method when implementing environment.")    
    
    @abstractmethod
    def getScore(self):
        """
        Returns the total score of a finished game
        """
        raise NotImplementedError("Please overload getScore method when implementing environment.")    