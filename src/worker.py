# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
import threading
import time
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
        self.initialState = 0
        self.paramMean = 0
        self.meanLoss = {'total': 0, 'policy': 0, 'value': 0, 'entropy': 0}
        self.expTime = 0
        self.gradTime = 0
        self.discountTime = 0
        self.rewardTime = 0
        self.advantageTime = 0
        self.logTime = 0
        self.updateTime = 0
        self.totalTime = 0
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
        self.gamesPlayed = 0 # counter for full episodes that were played. Is triggered by env.isDone
        self.updatesDone = 0 # counter for the number of updates that were done. Can be triggered by env.isDone, env.isPartDone and by updateFrequency. 
        self.nSteps = 0 # steps for each update. Can be triggered by update frequency or env.isPartDone
        self.gameSteps = 0 # steps for a whole game. Only triggered by env.isDone()
        self.rewardDiscount = self.mainThread.cfg['rewardDiscount']
        self.normRange = self.mainThread.cfg['normRange']
        self.normalizeRewards = self.mainThread.cfg['normalizeRewards']
        self.normalizeAdvantages = self.mainThread.cfg['normalizeAdvantages']
        self.updateOnPartDone = self.mainThread.cfg['updateOnPartDone']
        self.verbose = mainThread.verbose
        self.rewards     = None
        self.values      = None
        self.policyOutput      = []
        self.states      = []
        self.chosenPolicy    = []
        self.resetTrigger= []
        self.rewardHistory = None
        self.advantageHistory = None
        
    def getDiscountedRewardNormal(self, lastValue):
        """
        calculates the reward
        Args:
            lastValue(float): value of step after last action
        Returns:
            (mx.nd.array) discounted reward for each timestep 
        """
        timeLeft = self.rewards.shape[0]
        R = lastValue
        out = mx.nd.zeros(shape = (1))
        for n in reversed(range(timeLeft)):
            if self.resetTrigger[n]: R = 0
            R = self.rewardDiscount * R + self.rewards[n]
            out = mx.nd.concat(out, R, dim = 0)
        ## delete dummy first element and reverse again
        out = out[1:]
        out = out[::-1]
        return(out)

    def getDiscountedRewardGAE(self, lastValue, la = 0.96):
        """
        calculates the advantage according to generalized advantage estimation
        Args:
           lastValue(float): value of step after last action
           la: hyperparameter. Make it 1 for normal reward estimation
        Returns:
           discounted reward
        """
        ### calculates advantage first using GAE. Then in the last step reward is calculated
        timeLeft = self.rewards.shape[0]
        GAE    = 0
        ndLastValue = mx.nd.zeros(shape = (1))
        ndLastValue[0] = lastValue
        values = mx.nd.concat(self.values, ndLastValue, dim = 0)
        out    = mx.nd.zeros(shape = (1))
        for n in reversed(range(timeLeft)):
            if self.resetTrigger[n]: 
                GAE = 0
                values[n+1] = 0 ## value after game ends is 0
            delta = self.rewards[n] + self.rewardDiscount * values[n+1] - values[n]
            GAE   = self.rewardDiscount * la * GAE + delta
            out   = mx.nd.concat(out, GAE, dim = 0)
        ## delete dummy first element and reverse again
        out = out[1:]
        out = out[::-1]
        ## add values to obtain rewards instead of advantages
        out = out + self.values
        return(out)
    
    def getDiscountedReward(self, lastValue, la = 0.96, useGAE = True):
        if useGAE:
            reward = self.getDiscountedRewardGAE(lastValue = lastValue, la = la)
        else: 
            reward = self.getDiscountedRewardNormal(lastValue = lastValue)
        return(reward)
    
    def normalizeAdvantage(self, advantages, nEpisodes = 5):
        """
        takes advantages and normalizes them
        with respect to the nEpisodes last episodes
        Args:
            advantages (ndArray): the advantages for each step
            nEpisodes (int): number of episodes that are included in history for normalization
        Returns:
            ndArray normalized advantages
            
        self.advantageHistory is an array with episode Number and advantage as columns
        """
        newAdvantageHistory = mx.nd.empty(shape = (advantages.shape[0], 2))
        newAdvantageHistory[:,1] = advantages[:]
        if self.advantageHistory is None: # start of the game
            lastEpisode = 0
            newAdvantageHistory[:,0] = 1
            self.advantageHistory = newAdvantageHistory
        else:
            lastEpisode = int(self.advantageHistory[:,0].max(0).asscalar())
            ## update episode history
            if lastEpisode == nEpisodes:
                ## need to drop oldest episode before update
                self.advantageHistory = self.advantageHistory[int((self.advantageHistory[:,0] > 1).argmax(0).asscalar()):,:]
                self.advantageHistory[:,0] -= 1
                newAdvantageHistory[:,0] = nEpisodes
            else:
                newAdvantageHistory[:,0] = lastEpisode + 1
            self.advantageHistory = mx.nd.concat(self.advantageHistory, newAdvantageHistory, dim = 0)
        ## do the normalization
#        pdb.set_trace()
        advHnp = self.advantageHistory.asnumpy() ## unfortunately no std method for ndArray
        na = (advantages - self.advantageHistory[:,1].mean()) / (advHnp[:,1].std() + 1e-7)
        return (na)
    
    def normalizeReward(self, discountedReward, nEpisodes = 5):
        """
        takes discounted reward and normalizes it
        with respect to the nEpisodes last episodes
        Args:
            discountedReward (ndArray): the discounted reward for each step
            nEpisodes (int): number of episodes that are included in history for normalization
        Returns:
            ndArray normalized rewards
            
        self.rewardHistory is an array with episode Number, reward as columns
        """
        newRewardHistory = mx.nd.empty(shape = (discountedReward.shape[0], 2))
        newRewardHistory[:,1] = discountedReward[:]
        if self.rewardHistory is None: # start of the game
            lastEpisode = 0
            newRewardHistory[:,0] = 1
            self.rewardHistory = newRewardHistory
        else:
            lastEpisode = int(self.rewardHistory[:,0].max(0).asscalar())
            ## update episode history
            if lastEpisode == nEpisodes:
                ## need to drop oldest episode before update
                self.rewardHistory = self.rewardHistory[int((self.rewardHistory[:,0] > 1).argmax(0).asscalar()):,:]
                self.rewardHistory[:,0] -= 1
                newRewardHistory[:,0] = nEpisodes
            else:
                newRewardHistory[:,0] = lastEpisode + 1
            self.rewardHistory = mx.nd.concat(self.rewardHistory, newRewardHistory, dim = 0)
        ## do the normalization
#        pdb.set_trace()
        rewHnp = self.rewardHistory.asnumpy() ## unfortunately no std method for ndArray
        nr = (discountedReward - self.rewardHistory[:,1].mean()) / (rewHnp[:,1].std() + 1e-7)
        return (nr)           
    
    def getPerformanceIndicators(self, normalizedRewards, normalizedAdvantages, verbose = True):
        """
        Returns a tuple with performance indicators for the given step
        Args:
            normalizedRewards (mx.ndArray): vector with normalized rewards.
            normalizedAdvantages (mx.ndArray): vector with normalized advantages 
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
        policies = [v.asnumpy().item() for v in self.chosenPolicy]
        actionDst = "" 
        for i in range(len(self.environment.getValidActions())):
            actionDst += "{0}: {1}; ".format(i, policies.count(i))
        
        tmp = {'workerId': self.id, 
               'step':self.nSteps,
               'gameStep': self.gameSteps,
               'updateSteps': self.updateSteps,
               'gamesFinished': self.gamesPlayed + 1, 
               'updatesDone': self.updatesDone + 1,
               'loss': self.meanLoss['total'], 
               'lossPolicy': self.meanLoss['policy'],
               'lossValue': self.meanLoss['value'],
               'lossEntropy': self.meanLoss['entropy'],
               'score': score,
               'normalizedRewards': normalizedRewards.sum().asscalar() / normalizedRewards.shape[0],
               'normalizedAdvantages': normalizedAdvantages.sum().asscalar() / normalizedAdvantages.shape[0],
               'actionDst': actionDst,
               'expTime': self.expTime,
               'gradTime': self.gradTime,
               'rewardTime': self.rewardTime,
               'advantageTime': self.advantageTime,
               'logTime': self.logTime,
               'updateTime': self.updateTime,
               'totalTime': self.totalTime
               }
        if verbose: 
            print("Worker: {0}, games: {1}, step: {2} loss: {3}, score: {4}, normalizedRewards: {5}, normalizedAdvantages: {6}".format(self.id, tmp['gamesFinished'], tmp['step'], tmp['loss'], tmp['score'], tmp['normalizedRewards'], tmp['normalizedAdvantages']))
        return pd.DataFrame(tmp, index = [self.updatesDone])
        
    def collectDiagnosticInfo(self):
        """
        adds gradients, actions, etc for diagnostic info at update time to extended log
        """
        gradSummary = mx.nd.moments(mx.nd.concat(*[x.grad().reshape(-1) for x in self.net.collect_params().values()], 
                                                 dim = 0), 
                                    axes = 0)
        actions = {'workerId': self.id, 'gamesFinished': self.gamesPlayed + 1, 'updatesDone': self.updatesDone, 'gradMean': gradSummary[0].asscalar(), 'gradSd': mx.nd.sqrt(gradSummary[1]).asscalar(), 'paramMean': self.paramMean}
        return pd.DataFrame(actions, index = [self.updatesDone])
        
        
    
    def run(self):
        """
        Do the actual training
        """
        print("Worker{0} started!".format(self.id))
        while(self.gamesPlayed < self.nGames):
        ## loop over the games         
            ## start a new game
            ts = time.time()
            ts0 = ts
            self.environment.reset()
            ## store initial state for diagnostic info
            self.initialState = self.environment.getRawState()
            ## reset model (e.g. lstm initial states)
            self.net.reset()
            if self.verbose:
                ## check for model params by calculating the mean of all params
                # pdb.set_trace()
                paramMeans = {k: v.data().asnumpy().mean() for k,v in self.net.collect_params().items()}
                self.paramMean = np.mean(list(paramMeans.values()))
            while(not self.environment.isDone()):
                self.nSteps += 1
                self.gameSteps += 1
                self.states.append(mx.nd.array(self.environment.getNetState()))                 
                ## do the forward pass for policy and value determination. No label known yet.
                value, policy = self.net(self.environment.getNetState())
                ## determine chosen policy. Only validActions allowed. Invalid actions are set to prob 0.
                allowedPolicies = mx.nd.zeros_like(policy)
                validIdx = np.where(self.environment.getValidActions())[0]
                allowedPolicies[0, validIdx] = policy[0,validIdx]
                if allowedPolicies.sum() == 0:
                    ## all valid actions have score 0. Assign equal scores
                    allowedPolicies[0, validIdx] = 1.0 / float(validIdx.size)
                else:
                    ## renormalize to 1
                    allowedPolicies = allowedPolicies / allowedPolicies.sum()
                self.chosenPolicy.append(mx.nd.sample_multinomial(data  = allowedPolicies,
                                                              shape = 1))
                ## Store value and probability of chosen policy for loss calculation
                if self.values is None:
                    self.values = value[0,0]
                else:
                    self.values = mx.nd.concat(self.values, value[0,0], dim = 0)
                self.policyOutput.append(policy[0, self.chosenPolicy[-1]])
                self.environment.update(self.chosenPolicy[-1].asnumpy())
                if self.rewards is None:
                    self.rewards = mx.nd.array([self.environment.getLastReward()])
                else:
                    self.rewards = mx.nd.concat(self.rewards, mx.nd.array([self.environment.getLastReward()]), dim = 0)
                ## if a part of an episode is finished, we need to reset the net, e.g. one ball in Pong
                self.resetTrigger.append(self.environment.isPartDone())
                if self.resetTrigger[-1]: self.net.reset()
                
                ts1 = time.time()
                self.expTime += ts1 - ts
                ts = ts1
                # check, whether an update needs to be performed.
                updateTrigger = self.nSteps%self.updateFrequency == 0 or self.environment.isDone()
                if self.updateOnPartDone:
                    updateTrigger = updateTrigger or self.environment.isPartDone()
                if updateTrigger:
                ## we are updating!
                    ts2 = time.time()    
                    self.updateSteps = self.nSteps%self.updateFrequency
                    if self.updateSteps == 0:
                        self.updateSteps = self.updateFrequency
                    if self.verbose:
                        print("Worker{0}; game {1}; step {2}; updateSteps {3}!".format(self.id, self.gamesPlayed + 1, self.nSteps, self.updateSteps))
                        print('\npolicy scores:')
                        print(policy)
                        print('Allowed policies:')
                        print(allowedPolicies)
                        print('\n')
                    self.meanLoss = {k:0 for k,v in self.meanLoss.items()}
                    if self.environment.isDone() or self.environment.isPartDone():
                        lastValue = 0 # this is debatable for partDone. It might make sense to remember e.g. lstm init states across partDone epsiodes.
                    else:
                      ## get value of new state after policy update as
                      ## future reward estimate
                      lastValue, _ = self.net(self.environment.getNetState())
                      lastValue = int(lastValue[0].asscalar())
                    ts3 = time.time()
                    self.gradTime += ts3 - ts2    
                    
                    discountedReward = self.getDiscountedReward(lastValue, la = self.mainThread.cfg['lambda'], useGAE = self.mainThread.useGAE)
                    if self.normalizeRewards and (self.normRange is not None):
                        discountedReward = self.normalizeReward(discountedReward, nEpisodes=self.normRange)
                    ts4 = time.time()
                    self.rewardTime += ts4 - ts3
                    advantages = (discountedReward - self.values)
                    if self.normalizeAdvantages and (self.normRange is not None):
                        advantages = self.normalizeAdvantage(advantages, nEpisodes=self.normRange)
                    ts5 = time.time()
                    self.advantageTime += ts5 - ts4
                    ## reset model (e.g. lstm initial states)
                    ## make sure to remember initStates to reset later
                    if self.environment.isDone() or self.environment.isPartDone():
                        initStates = None
                    else:
                        initStates = self.net.getInitStates()
                    self.net.reset()                                        
                    
                    for t in range(self.updateSteps):
                    ## loop over all memory to do the update.
                        ## do forward and backward pass to accumulate gradients
                        #pdb.set_trace()
                        with mx.autograd.record(): ## per default in "is_train" mode
                            value, policy = self.net(self.states[t])
                            ## extract the prob for the chosen policy from the policy vector
                            policy = policy[0,self.chosenPolicy[t]]
                            loss = self.net.lossFct[0][0](value, policy, discountedReward[t], advantages[t], self.policyOutput[t])
                        loss.backward() ## grd_req is add, so gradients are accumulated       
                        ## reset model if necessary
                        if self.resetTrigger[t]: self.net.reset()   
                        self.meanLoss['total'] += self.net.lossFct[0][0].getLoss()
                        self.meanLoss['policy'] += self.net.lossFct[0][0].getPolicyLoss()
                        self.meanLoss['value'] += self.net.lossFct[0][0].getValueLoss()
                        self.meanLoss['entropy'] += self.net.lossFct[0][0].getEntropyLoss()
                    self.meanLoss = {k: float((v / self.updateSteps).asnumpy()) for k,v in self.meanLoss.items()}
                    ## send gradients to mainThread and do the update.
                    ## gradients on mainThread get cleared automatically
                    ts1 = time.time()
                    self.gradTime += ts1 - ts5
                    ts = ts1
                    self.mainThread.net.updateFromWorker(fromNet = self.net, dummyData = self.environment.getNetState())
                    ## get new parameters from mainThread
                    self.net.copyParams(fromNet = self.mainThread.net)
                    ts1 = time.time()
                    self.updateTime += ts1 - ts
                    ts = ts1
                    self.totalTime = ts - ts0
                    ## store performance indicators after game is finished 
                    ts3 = time.time()
                    self.mainThread.log = self.mainThread.log.append(self.getPerformanceIndicators( normalizedRewards = discountedReward, normalizedAdvantages = advantages, verbose=True), sort = True)                        
                    if self.verbose:
                        self.mainThread.extendedLog = self.mainThread.extendedLog.append(self.collectDiagnosticInfo(), sort=True)                        
                    self.logTime += time.time()-ts3
                    ## clear local gradients.
                    self.net.clearGradients()
                    ## make sure to reset model to continue collecting experience
                    self.net.reset(initStates)
                    self.updatesDone += 1
                    ## clear memory
                    self.nSteps = 0
                    self.rewards     = None
                    self.values      = None
                    self.policyOutput = []
                    self.states      = []
                    self.chosenPolicy    = []
                    self.resetTrigger= []
                    
            
            self.gamesPlayed += 1
            self.mainThread.gameCounter += 1
            if self.mainThread.outputDir is not None and self.mainThread.gameCounter > 0:
                if self.mainThread.gameCounter % self.mainThread.saveInterval == 0:
                    self.mainThread.save(os.path.join(self.mainThread.outputDir, str(self.mainThread.gameCounter)), savePlots = True, overwrite = True)
            self.nSteps = 0
            self.gameSteps = 0
            self.expTime = 0
            self.gradTime = 0
            self.updateTime = 0
            self.totalTime = 0
            self.discountTime = 0
            self.logTime = 0
            self.rewardTime = 0
            self.advantageTime = 0
        
                
        
        
        
    