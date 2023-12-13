# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
from worker import worker
import yaml
import pandas as pd
import copy
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
        self.log = pd.DataFrame(columns = ['workerId','step','gameStep', 'updateSteps','gamesFinished',
                                           'loss','lossPolicy','lossValue','lossEntropy',
                                           'score','normalizedRewards', 'normalizedAdvantages', 'actionDst',
                                           'expTime', 'gradTime', 'rewardTime', 'advantageTime',
                                           'logTime', 'updateTime', 'totalTime'])
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
        for wId in range(self.cfg['nWorkers']):
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
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'].astype(int),
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
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'].astype(int),
                        y = thisData['lossValue'].astype(float),
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
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'].astype(int),
                        y = thisData['lossPolicy'].astype(float),
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
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'].astype(int),
                        y = thisData['lossEntropy'].astype(float),
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
                        x = thisData['gamesFinished'].astype(int),
                        y = thisData['score'].astype(float),
                        mode = 'lines+markers',
                        visible = 'legendonly',
                        name = "worker {0}".format(wId)))        
        # get average across workers and then moving average over 5 games
        meanData = copy.deepcopy(self.log[self.log['score'] != -999])
        meanData = meanData.sort_values(['gamesFinished', 'workerId'])
        meanData['score'] = meanData['score'].astype('float32')
        meanData = meanData.groupby(['gamesFinished'], as_index = False).agg({'score': 'mean'})
        meanData['meanCol'] = meanData['score'].rolling(5).mean()
        meanData['meanCol'] = meanData['meanCol'].fillna(0)
        data.append(go.Scatter(
                        x = meanData['gamesFinished'].astype(int),
                        y = meanData['meanCol'].astype(float),
                        mode = 'lines+markers',
                        name = "moving average"))                
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'scores.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId) & (self.log['score'] != -999)]
            thisData = thisData.sort_values(['gamesFinished'])
            data.append(go.Scatter(
                        x = thisData['gamesFinished'].astype(int),
                        y = thisData['gameStep'].astype(float),
                        mode = 'lines+markers',
                        visible = 'legendonly',
                        name = "worker {0}".format(wId)))        
        # get average across workers and then moving average over 5 games
        meanData = copy.deepcopy(self.log[self.log['score'] != -999])
        meanData['gameStep'] = meanData['gameStep'].astype('float32')
        meanData = meanData.sort_values(['gamesFinished', 'workerId'])
        meanData = meanData.groupby(['gamesFinished'], as_index = False).agg({'gameStep': 'mean'})
        meanData['meanCol'] = meanData['gameStep'].rolling(5).mean()
        meanData['meanCol'] = meanData['meanCol'].fillna(0)
        data.append(go.Scatter(
                        x = meanData['gamesFinished'].astype(int),
                        y = meanData['meanCol'].astype(float),
                        mode = 'lines+markers',
                        name = "moving average"))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'episodeLengths.html'))
        
        data = []
        wId = np.unique(self.log['workerId'])[0]
        thisData = self.log[(self.log['workerId'] == wId)]
        thisData = thisData.sort_values(['updateSteps'])
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['expTime'].astype(float),
                mode = 'markers',
                name = "expTime worker {0}".format(wId)))
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['gradTime'].astype(float),
                mode = 'markers',
                name = "gradTime worker {0}".format(wId)))
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['updateTime'].astype(float),
                mode = 'markers',
                name = "updateTime worker {0}".format(wId)))            
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['totalTime'].astype(float),
                mode = 'markers',
                name = "totalTime worker {0}".format(wId)))                        
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = (thisData['totalTime'] - thisData['expTime'] - thisData['updateTime'] - thisData['gradTime']).astype(float),
                mode = 'markers',
                name = "residualTime worker {0}".format(wId)))          
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['logTime'].astype(float),
                mode = 'markers',
                name = "logTime worker {0}".format(wId)))  
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['rewardTime'].astype(float),
                mode = 'markers',
                name = "rewardTime worker {0}".format(wId))) 
        data.append(go.Scatter(
                x = thisData['updateSteps'].astype(int),
                y = thisData['advantageTime'].astype(float),
                mode = 'markers',
                name = "advantageTime worker {0}".format(wId))) 
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'timing.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'],
                        y = thisData['totalTime'] / thisData['updateSteps'],
                        mode = 'lines+markers',
                        name = "total worker {0}".format(wId)))        
            data.append(go.Scatter(
                        x = thisData['updatesDone'],
                        y = thisData['gradTime']/ thisData['updateSteps'],
                        mode = 'lines+markers',
                        name = "grad worker {0}".format(wId)))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'timePerStepOverupdatesDone.html'))
        
        data = []
        for wId in np.unique(self.log['workerId']):
            thisData = self.log[(self.log['workerId'] == wId)]
            thisData = thisData.sort_values(['updatesDone'])
            data.append(go.Scatter(
                        x = thisData['updatesDone'],
                        y = thisData['normalizedRewards'],
                        mode = 'lines+markers',
                        name = "worker {0}".format(wId)))        
        out = pi.plotlyInterface(data)
        out.plotToFile(os.path.join(dirname, 'averageRewards.html'))
        
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
        if savePlots and not all(self.log['score']==-999):
            self.getPerformancePlots(os.path.join(outFolder, "plots"), overwrite = True)
            
    