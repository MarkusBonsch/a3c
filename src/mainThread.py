# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
from mxnetTools import mxnetTools as mxT
from mxnetTools import a3cModule
from worker import worker
import yaml
import pandas as pd
import os
import plotly.graph_objs as go
import plotlyInterface as pi

class mainThread:
    """
    This is the main thread that controls training. 
    It sets up the model, the workers, and the optimizer.
    It contains:
    
    A module (identical to worker modules)
    that is used to do the updates. Workflow is as follows:
    - Worker gets parameters from paramServer Module
    - Worker computes gradients and pushes them to parameter server module
    - Parameter server module performs update
    - repeat
    
    A log, essentially a list where the workers can enter train metrics
    """    
    def __init__(self, symbol, environment, configFile, verbose = False):
        """
        Sets up a parameter server accorfing to a config.
        Args:
            symbol (mx.symbol): the neural network symbol without the output layer. 
                                The a3c output layer is added here.                                
                                Input layer must be named 'data'
            environment(of class environment): the game environment.
            configFile(string): path to the config file
        """
        self.log = []
        self.readConfig(configFile)
        ## get the number of policy options in the game
        self.environment = environment
        self.environment.reset() ## initialize to start
        self.nPolicy = self.environment.getRewards().size
        
        ## get the input dimension of the game
        self.inputDim = self.environment.state.size
        
        ## construct the final neural network symbol.     
        self.symbol = mxT.a3cOutput(symbol, self.nPolicy)
        
        ## bind the module 
        self.module = a3cModule(self.symbol, 
                                inputDim      = self.inputDim,
                                optimizer     = self.cfg['optimizer'],
                                optimizerArgs = self.cfg['optimizerArgs'])
        self.verbose = verbose
        
    def readConfig(self, configFile):
        """
        reads the config, does some necessary transformations and stores it
        Args:
            configFile(str): path to the config file
        """
        with open(configFile, "r") as f:
            self.cfg = yaml.load(f)
            
        ## convert optimizer args to the correct structure
        self.cfg['optimizerArgs'] = tuple(self.cfg['optimizerArgs'].iteritems())
        
    def run(self):
        """
        Do the actual training.
        """
        
        ## clear the log
        self.log = []
        ## initialize workers
        workers = []
        for wId in xrange(self.cfg['nWorkers']):
            thisWorker = worker(self)
            workers.append(thisWorker)
            thisWorker.start()
        
        for x in workers:
            x.join()
            
        ## transform the log into nice pandas data frames
        self.log = map(pd.DataFrame,self.log)
        
    def getPerformancePlots(self, dirname = 'mainThreadPerformance'):
        """
        Produces some plots with scores
            Args: 
                dirname (str): the output folder        
        """
        os.mkdir(dirname)
        
        data = []
        for wData in self.log:
            data.append(go.Scatter(
                        x = wData['gamesFinished'],
                        y = wData['score'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wData.loc[0,'workerId'])))      
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'scores.html'))
    