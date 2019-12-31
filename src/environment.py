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
    def getState(self):    
        """
        Returns the state as numpy array
        """
        raise NotImplementedError("Please overload getState method when implementing environment.")    

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
        Receives bool that indicates whether the game is over
        """
        raise NotImplementedError("Please overload isDone method when implementing environment.")   
    
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