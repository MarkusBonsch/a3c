# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
import threading
from mxnetTools import a3cModule
import copy

class worker(threading.Thread):
    """
    Setup for worker instances that perform the training.
    They synchronize their parameters with a server, play the game,
    compute gradients and push them back to the server for updates.
    """    
    def __init__(self, mainThread):
        """
        Args:
            mainThread (object of class paramServer): the parameter server to connect to.
            
        """
        threading.Thread.__init__(self)
        self.mainThread = mainThread        
        ## determine worker id from length of mainThread log
        self.id = len(self.mainThread.log) - 1
        ## setup the local module. Optimizer doesn't matter since updates are performed by mainThread
        self.module = a3cModule(symbol   = self.mainThread.symbol, 
                                inputDim = self.mainThread.inputDim)
        ## own copy of game environment
        self.environment = copy.deepcopy(self.mainThread.environment)
        self.updateFrequency = self.mainThread.cfg['updateFrequency']
        self.nGames = self.mainThread.cfg['nGames']
    
    def run(self):
        """
        Do the actual training
        """
                
        
        
        
    