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
    def __init__(self, netMaker, envMaker, configFile):
        """
        Sets up a parameter server accorfing to a config.
        Args:
            netMaker (function): a function returning the neural network
                                 with initialized parameters. Must be of type a3cHybridSequential.
                                 Last block must be a3cOutput
            envMaker(function): creates the game environment.
            configFile(string): path to the config file
        """
        self.log = pd.DataFrame(columns = ['workerId','step','updateSteps','gamesFinished',
                                           'loss','lossPolicy','lossValue','lossEntropy',
                                           'score','rewards','actionDst'])
        self.extendedLog  = pd.DataFrame(columns = ['workerId','gamesFinished','gradMean',
                                                     'gradSd','actions','initialState','paramMean'])
        self.gameCounter = 0
        self.readConfig(configFile)
        self.outputDir = self.cfg['outputDir']
        self.saveInterval =self.cfg['saveInterval']
        
        
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
        if self.cfg['trainerFile'] is not None:
            self.net.trainer.load_states(self.cfg['trainerFile'])
        self.verbose = self.cfg['verbose']
        
    def readConfig(self, configFile):
        """
        reads the config, does some necessary transformations and stores it
        Args:
            configFile(str): path to the config file
        """
        with open(configFile, "r") as f:
            self.cfg = yaml.safe_load(f)
        
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
        self.save(os.path.join(self.outputDir, "final"), savePlots = True, overwrite = True)
        
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
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'].astype(int),
                        y = thisData['loss'].astype(float),
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))  
        maxY = thisData['loss'].quantile(0.9)
        minY = thisData['loss'].min()
        out = pi.plotlyInterface(data)
#        out.fig.update_layout(
#                yaxis = go.layout.YAxis(
#                            range = [minY, maxY]))
        out.plotToFile(os.path.join(dirname, 'losses.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossValue'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        out = pi.plotlyInterface(data)
        maxY = thisData['lossValue'].quantile(0.9)
        minY = thisData['lossValue'].min()
        out = pi.plotlyInterface(data)
#        out.fig.update_layout(
#                yaxis = go.layout.YAxis(
#                            range = [minY, maxY]))
        out.plotToFile(os.path.join(dirname, 'lossesValue.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossPolicy'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      
        
        maxY = thisData['lossPolicy'].quantile(0.9)
        minY = thisData['lossPolicy'].min()
        out = pi.plotlyInterface(data)
#        out.fig.update_layout(
#                yaxis = go.layout.YAxis(
#                            range = [minY, maxY]))
        out.plotToFile(os.path.join(dirname, 'lossesPolicy.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['lossEntropy'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))      

        maxY = thisData['lossEntropy'].quantile(0.9)
        minY = thisData['lossEntropy'].min()
        out = pi.plotlyInterface(data)
#        out.fig.update_layout(
#                yaxis = go.layout.YAxis(
#                            range = [minY, maxY]))
        out.plotToFile(os.path.join(dirname, 'lossesEntropy.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['score'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'scores.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['step'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'episodeLengths.html'))
        
        data = []
        wId = np.unique(self.log['workerId'])[0]
        thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
        thisData = thisData.sort_values(['step'])
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['expTime'],
                mode = 'markers',
                name = "expTime worker {0}".format(wId)))
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['gradTime'],
                mode = 'markers',
                name = "gradTime worker {0}".format(wId)))
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['updateTime'],
                mode = 'markers',
                name = "updateTime worker {0}".format(wId)))            
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['totalTime'],
                mode = 'markers',
                name = "totalTime worker {0}".format(wId)))                        
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['totalTime'] - thisData['expTime'] - thisData['updateTime'] - thisData['gradTime'],
                mode = 'markers',
                name = "residualTime worker {0}".format(wId)))          
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['logTime'],
                mode = 'markers',
                name = "logTime worker {0}".format(wId)))  
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['discountTime'],
                mode = 'markers',
                name = "discountTime worker {0}".format(wId)))  
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['rewardTime'],
                mode = 'markers',
                name = "rewardTime worker {0}".format(wId))) 
        data.append(go.Scatter(
                x = thisData['step'],
                y = thisData['advantageTime'],
                mode = 'markers',
                name = "advantageTime worker {0}".format(wId))) 
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'timing.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['totalTime'],
                        mode = 'lines+markers',
                        name = "total worker {0}".format(wId)))        
            data.append(go.Scatter(
                        x = thisData['gamesFinished'],
                        y = thisData['gradTime'],
                        mode = 'lines+markers',
                        name = "grad worker {0}".format(wId)))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'timeOverEpisode.html'))
        
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
        if len(self.extendedLog) > 0:
            self.extendedLog.to_pickle(os.path.join(outFolder, 'extendedLog.pck'))
        ## save config
        with open(os.path.join(outFolder, 'config.cfg'), 'w') as outfile:
            yaml.dump(self.cfg, outfile, default_flow_style=False)
        if savePlots:
            self.getPerformancePlots(os.path.join(outFolder, "plots"), overwrite = True)
            
    