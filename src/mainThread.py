# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
from worker import worker
import yaml
import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
import plotlyInterface as pi
import pdb

class mainThread:
    """
    This is the main thread that controls training. 
    It sets up the model, the workers, and the optimizer.
    It contains:
    
    A network (identical to worker modules)
    that is used to do the updates. Workflow is as follows:
    - Worker gets parameters from paramServer Network
    - Worker computes gradients and pushes them to parameter server Network
    - Parameter server Network performs update
    - repeat
    
    A log, essentially a list where the workers can enter train metrics
    """    
    def __init__(self, netMaker, envMaker, configFile, outputDir = None, saveInterval = 200, verbose = False):
        """
        Sets up a parameter server accorfing to a config.
        Args:
            netMaker (function): a function returning the neural network
                                 with initialized parameters. Must be of type a3cHybridSequential.
                                 Last block must be a3cOutput
            envMAker(function): creates the game environment.
            configFile(string): path to the config file
            outputDir (string): if None, no output is saved. Otherwise, the model and log are saved every 200 games.
            sevaInterval (int): model will be saved after saveInterval episodes.
        """
        self.log = pd.DataFrame(columns = ['workerId','step','updateSteps','gamesFinished',
                                           'loss','lossPolicy','lossValue','lossEntropy',
                                           'score','rewards','actionDst'])
        self.extendedLog = []
        self.gradients = []
        self.gameCounter = 0
        self.outputDir = outputDir
        self.saveInterval =saveInterval
        self.readConfig(configFile)
        
        self.envMaker = envMaker
        self.environment = self.envMaker()
        self.environment.reset() ## initialize to start
        
        self.netMaker = netMaker
        self.net = netMaker()
        self.net.hybridize()
        ## do one forward pass to really initialize the parameters
        self.net(self.environment.getNetState())
        ## the trainer updates the parameters
        self.net.initTrainer(optimizer     = self.cfg['optimizer'],
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
        
    def run(self):
        """
        Do the actual training.
        """
    
        ## initialize workers
        workers = []
        for wId in xrange(self.cfg['nWorkers']):
            thisWorker = worker(self, id = wId)
            workers.append(thisWorker)
            thisWorker.start()
        
        for x in workers:
            x.join()

        self.extendedLog = map(pd.DataFrame,self.extendedLog)
        self.extendedLog = pd.concat(self.extendedLog)
        self.save("{0}_{1}".format(self.outputDir, self.gameCounter), savePlots = True, overwrite = True)
        
    def getPerformancePlots(self, dirname = 'mainThreadPerformance', overwrite = False):
        """
        Produces some plots with scores
            Args: 
                dirname (str): the output folder        
                overwrite (Bool): whether to overwrite an existing folder
        """
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        elif not overwrite:
            raise IOError("Directory {0} already exists!".format(dirname))  
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData.sort_values(['gamesFinished'])
            thisData.drop(thisData.index[0])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['loss'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'losses.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData.sort_values(['gamesFinished'])
            thisData.drop(thisData.index[0])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossValue'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'lossesValue.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData.sort_values(['gamesFinished'])
            thisData.drop(thisData.index[0])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossPolicy'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'lossesPolicy.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData.sort_values(['gamesFinished'])
            thisData.drop(thisData.index[0])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossEntropy'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'lossesEntropy.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData.sort_values(['gamesFinished'])
            thisData.drop(thisData.index[0])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['score'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))        
        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'scores.html'))
        
    def save(self, outFolder, savePlots = True, overwrite = False):
        """ 
        saves the model and the config and plots if required
        Args:
            outFolder (str): path to the target directory where files are saved.
                           Will be created if it doesn't exist
            savePlots (bool): if True, performancePlots are created and saved.
            overwrite(bool): whether to overwrite dir if it exists.
        """
        if not os.path.exists(outFolder):
                os.makedirs(outFolder)
        elif not overwrite:
                raise  IOError('Folder already exists: ' + outFolder + '. Specify overwrite = True if needed')
        ## save model
        self.net.save(outFolder, overwrite = True)
        ## save log
        self.log.to_pickle(os.path.join(outFolder, 'log.pck'))
        ## save config
        with open(os.path.join(outFolder, 'config.cfg'), 'w') as outfile:
            yaml.dump(self.cfg, outfile, default_flow_style=False)
        if savePlots:
            self.getPerformancePlots(os.path.join(outFolder, "plots"), overwrite = True)
            
    