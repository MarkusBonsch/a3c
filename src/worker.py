# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
import threading
import mxnet as mx
import mxnetTools as mxT
import os
import pandas as pd
import pdb
import numpy as np

class worker(threading.Thread):
    """
    Setup for worker instances that perform the training.
    They synchronize their parameters with a server, play the game,
    compute gradients and push them back to the server for updates.
    """    
    def __init__(self, mainThread, id):
        """
        Args:
            mainThread (object of class paramServer): the parameter server to connect to.
            id (int): the id
            
        """
        threading.Thread.__init__(self)
        self.mainThread = mainThread      
        ## determine worker id from length of mainThread log
        self.id = id
        self.extendedLog = []
        self.initialState = 0
        self.paramMean = 0
        self.gradients = []
        self.meanLoss = {'total': 0, 'policy': 0, 'value': 0, 'entropy': 0}
        ## own copy of game environment
        self.environment = self.mainThread.envMaker()
        ## setup the local network.
        self.net = self.mainThread.netMaker()
        ## make sure gradients are added up
        self.net.set_grad_req("add")
        self.net.hybridize()
        ## one forward pass to really initialize the parameters
        self.net(self.environment.getNetState())
        ## make sure to initialize with mainThread Parameters
        self.net.copyParams(fromNet = self.mainThread.net)

        self.updateFrequency = self.mainThread.cfg['updateFrequency']
        self.nGames = self.mainThread.cfg['nGames']
        self.gamesPlayed = 0
        self.nSteps = 0
        self.rewardDiscount = self.mainThread.cfg['rewardDiscount']
        self.verbose = mainThread.verbose
        self.rewards     = []
        self.values      = []
        self.states      = []
        self.policies    = []
        self.resetTrigger= []
        self.rewardHistory = pd.DataFrame(columns = ['episode', 'reward', 'advantage'])
        
    def getDiscountedReward(self, lastValue):
        """
        calculates the reward
        Args:
            lastValue(float): value of step after last action
        Returns:
            discounted reward for each timestep
        """
        timeLeft = len(self.rewards)
        R = lastValue
        out = [None] * timeLeft
        for n in range(timeLeft):
            if self.resetTrigger[-(n+1)]: R = 0
            R = self.rewardDiscount * R + self.rewards[-(n+1)]
            out[n] = R
        return(out)
    
    def normalizeRewardAdvantage(self, discountedReward, advantages,nEpisodes = 10):
        """
        takes discounted reward and advantages and normalizes them
        with respect to the nEpisodes last episodes
        Args:
            discountedReward (list of ndArray): the discounted reward for each step
            advantages (list of ndArray): the advantages for each step
            nEpisodes (int): number of episodes that are included in history for normalization
        Returns:
            normalized rewards and advantages
        """
        lastEpisode = np.max(self.rewardHistory['episode'])
        if np.isnan(lastEpisode):
            lastEpisode = 1
        ## update episode history
        if lastEpisode == nEpisodes:
            ## need to drop oldest episode before update
            self.rewardHistory = self.rewardHistory.loc[self.rewardHistory['episode'] != 1]
            self.rewardHistory['episode'] -=1
        newcols = len(discountedReward)
        newData = {'episode':    [lastEpisode + 1] * newcols, 
                   'rewards':    [v.asnumpy().item() for v in discountedReward],
                   'advantages': [v.asnumpy().item() for v in advantages]}
        self.rewardHistory = self.rewardHistory.append(pd.DataFrame(newData))
        ## do the normalization
#        pdb.set_trace()
        nr = [(v - np.mean(self.rewardHistory['rewards']))   / (np.std(self.rewardHistory['episode'])    + 1e-7) for v in discountedReward]
        na = [(v - np.mean(self.rewardHistory['advantages'])) / (np.std(self.rewardHistory['advantages']) + 1e-7) for v in advantages]        
        return (nr, na)
        
#    def getDiscountedReward(self, rewards, values, lastValue, t, ga, la):
#        """
#        calculates the advantage according to generalized advantage estimation
#        Args:
#            rewards (list): rewards of all timesteps after t
#            values(list):   values of all timesteps after t
#            lastValue(float): value of step after last action
#            t(int): timestep. starts with 0
#            ga, la: hyperparameters
#        Returns:
#            discounted reward
#        """
#        timeLeft = len(rewards) - t
#        values.append(lastValue)
#        R = 0
#        for n in range(timeLeft):
#            R_n = ga**(n+1) * values[t + n+1]
#            for l in reversed(range(n+1)):
#                R_n = R_n + ga**l * rewards[t + l]
#            R = R + la**n * R_n
#        R = R * (1-la)
#        return(R)
#            
    
    def getPerformanceIndicators(self, verbose = True):
        """
        Returns a tuple with performance indicators for the given step
        Args:
            verbose (bool): if True, performance indicators get printed to console
        Returns:
            a tuple with the performance indicators
        """
        score = -999
        if self.environment.isDone():
            score = self.environment.getScore()
#        pdb.set_trace()
#        if any([type(v) == int for v in self.rewards]):
#            pdb.set_trace()
        ## count action appearances
        policies = [v.asnumpy().item() for v in self.policies]
        actionDst = [] 
        for i in range(len(self.environment.getValidActions())):
            actionDst.append(policies.count(i))
        tmp = {'workerId': self.id, 
               'step':self.nSteps,
               'updateSteps': self.updateSteps,
               'gamesFinished': self.gamesPlayed + 1, 
               'loss': self.meanLoss['total'], 
               'lossPolicy': self.meanLoss['policy'],
               'lossValue': self.meanLoss['value'],
               'lossEntropy': self.meanLoss['entropy'],
               'score': score,
               'rewards': sum([v.asnumpy() for v in self.rewards])[0,0] / len(self.rewards),
               'actionDst': actionDst}
        
        if verbose: 
            print "Worker: {0}, games: {1}, step: {2} loss: {3}, score: {4}, rewards: {5}".format(self.id, tmp['gamesFinished'], tmp['step'], tmp['loss'], tmp['score'], tmp['rewards'])
        return pd.DataFrame(tmp)
        
    def collectDiagnosticInfo(self):
        """
        adds gradients, actions, etc for diagnostic info at update time to extended log
        """
        actions = {'workerId': self.id, 'gamesFinished': self.gamesPlayed + 1, 'actions': [int(x.asnumpy()) for x in self.policies], 'initialState': self.initialState, 'paramMean': self.paramMean}
        self.extendedLog.append(actions)
#        gradients = {'workerId': self.id, 'gradients': self.module.getGradientsNumpy()}
#        self.gradients.append(gradients)
#        
    
    def run(self):
        """
        Do the actual training
        """
        print "Worker{0} started!".format(self.id)
        while(self.gamesPlayed < self.nGames):
        ## loop over the games         
            ## start a new game
            self.environment.reset()
            ## store initial state for diagnostic info
            self.initialState = self.environment.getRawState()
            ## reset model (e.g. lstm initial states)
            self.net.reset()
            ## check for model params by calculating the mean of all params
            paramMeans = {k: v.data().asnumpy().mean() for k,v in self.net.collect_params().items()}
            self.paramMean = np.mean(paramMeans.values())
            while(not self.environment.isDone()):
                self.nSteps += 1
                self.states.append(mx.nd.array(self.environment.getNetState()))                 
                ## do the forward pass for policy and value determination. No label known yet.
                value, policy = self.net(self.environment.getNetState())
                self.values.append(value)
                ## store policy. Only validActions allowed. Invalid actions are set to prob 0.
                allowedPolicies = mx.nd.zeros_like(policy)
                validIdx = np.where(self.environment.getValidActions())[0]
                allowedPolicies[0, validIdx] = policy[0,validIdx]
                if allowedPolicies.sum() == 0:
                    ## all valid actions have score 0. Assign equal scores
                    allowedPolicies[0, validIdx] = 1.0 / float(validIdx.size)
                else:
                    ## renormalize to 1
                    allowedPolicies = allowedPolicies / allowedPolicies.sum()
                self.policies.append(mx.nd.sample_multinomial(data  = allowedPolicies,
                                                              shape = 1))
                ## apply action on state. Important before update
                self.environment.update(self.policies[-1].asnumpy())
                ## store reward and value as mx.nd.arrays
                self.rewards.append(mx.nd.array([[self.environment.getLastReward()]]))
                ## if a part of an episode is finished, we need to reset the net, e.g. one ball in Pong
                self.resetTrigger.append(self.environment.isPartDone())
                if self.resetTrigger[-1]: self.net.reset()
                
                if self.nSteps%self.updateFrequency == 0 or self.environment.isDone():
                ## we are updating!
                    self.updateSteps = self.nSteps%self.updateFrequency
                    if self.updateSteps == 0:
                        self.updateSteps = self.updateFrequency
                    if self.verbose:
                        print "Worker{0}; game {1}; step {2}; updateSteps {3}!".format(self.id, self.gamesPlayed + 1, self.nSteps, self.updateSteps)
                        print '\npolicy scores:'
                        print policy
                        print 'Allowed policies:'                    
                        print allowedPolicies
                        print '\n'
                    self.meanLoss = {k:0 for k,v in self.meanLoss.items()}
                    if self.environment.isDone():
                        lastValue = 0
                    else:
                      ## get value of new state after policy update as
                      ## future reward estimate
                      lastValue, _ = self.net(self.environment.getNetState())
                    discountedReward = self.getDiscountedReward(lastValue)
                    advantages = [r - v for (r, v) in zip(discountedReward, self.values)]
                    ## normalize
                    discountedReward, advantages = self.normalizeRewardAdvantage(discountedReward, advantages)
                    ## reset model (e.g. lstm initial states)
                    ## make sure to remember initStates to reset later
                    if self.environment.isDone() or self.environment.isPartDone():
                        initStates = None
                    else:
                        initStates = self.net.getInitStates()
                    self.net.reset()                                        
                    
                    for t in range(self.updateSteps):
                    ## loop over all memory to do the update.
                        ## determine advantages. All are set to 0, except the one 
                        ## for the chosen policy
                        advantageArray = mx.nd.zeros(shape = policy.shape)
                        advantageArray[0,self.policies[t]] = advantages[t]
                        ## reset model if necessary
                        if self.resetTrigger[t]: self.net.reset()
                        ## do forward and backward pass to accumulate gradients
                        with mx.autograd.record(): ## per default in "is_train" mode
                            value, policy = self.net(self.states[t])
                            loss = self.net.lossFct[0][0](value, policy, discountedReward[t], advantageArray)
                        loss.backward() ## grd_req is add, so gradients are accumulated       
                        
                        self.meanLoss['total'] += self.net.lossFct[0][0].getLoss()
                        self.meanLoss['policy'] += self.net.lossFct[0][0].getPolicyLoss()
                        self.meanLoss['value'] += self.net.lossFct[0][0].getValueLoss()
                        self.meanLoss['entropy'] += self.net.lossFct[0][0].getEntropyLoss()
                    self.meanLoss = {k: float((v / self.updateSteps).asnumpy()) for k,v in self.meanLoss.items()}
                    ## send gradients to mainThread and do the update.
                    ## gradients on mainThread get cleared automatically
                    self.mainThread.net.updateFromWorker(fromNet = self.net, dummyData = self.environment.getNetState())
                    ## get new parameters from mainThread
                    self.net.copyParams(fromNet = self.mainThread.net)
                    ## store performance indicators after game is finished      
                    self.mainThread.log = self.mainThread.log.append(self.getPerformanceIndicators( verbose=True) )                        
                    
                    self.collectDiagnosticInfo()
                    ## clear local gradients.
                    self.net.clearGradients()
                    ## clear memory
                    self.rewards     = []
                    self.values      = []
                    self.states      = []
                    self.policies    = []
                    self.resetTrigger= []
                    ## make sure to reset model to continue collecting experience
                    self.net.reset(initStates)
            
            self.gamesPlayed += 1
            self.mainThread.gameCounter += 1
            ## send extendedLog to mainThread after work is finished
            self.mainThread.extendedLog.append(self.extendedLog)
            self.mainThread.gradients. append(self.gradients)
            
            if self.mainThread.outputDir is not None and self.mainThread.gameCounter > 0:
                if self.mainThread.gameCounter % self.mainThread.saveInterval == 0:
                    self.mainThread.save("{0}_{1}".format(self.mainThread.outputDir, self.mainThread.gameCounter), savePlots = True, overwrite = True)
            self.nSteps = 0
        
                
        
        
        
    