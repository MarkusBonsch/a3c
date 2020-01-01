# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:28:01 2018

@author: markus
"""
import threading
from mxnetTools import a3cModule
from mxnetTools import mxnetTools as mxT
import mxnet as mx
import copy
import pdb
import numpy as np

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
        self.id = len(self.mainThread.log)
        ## initialize log
        self.mainThread.log.append([])
        self.extendedLog = []
        self.initialState = 0
        self.paramMean = 0
        self.gradients = []
        ## setup the local module. Optimizer doesn't matter since updates are performed by mainThread
        self.module = a3cModule(symbol   = self.mainThread.symbol, 
                                inputDim = self.mainThread.inputDim)
        ## own copy of game environment
        self.environment = self.mainThread.envMaker()
        self.updateFrequency = self.mainThread.cfg['updateFrequency']
        self.nGames = self.mainThread.cfg['nGames']
        self.gamesPlayed = 0
        self.nSteps = 0
        self.meanLoss = {'total': 0, 'policy': 0, 'value': 0, 'entropy': 0}
        self.rewardDiscount = self.mainThread.cfg['rewardDiscount']
        self.verbose = mainThread.verbose
        self.rewards     = []
        self.values      = []
        self.states      = []
        self.policies    = []
    
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
        tmp = {'workerId': self.id, 
               'step':self.nSteps,
               'updateSteps': self.updateSteps,
               'gamesFinished': self.gamesPlayed + 1, 
               'loss': self.meanLoss['total'], 
               'lossPolicy': self.meanLoss['policy'],
               'lossValue': self.meanLoss['value'],
               'lossEntropy': self.meanLoss['entropy'],
               'score': score}
        if verbose: 
            print "Worker: {0}, games: {1}, step: {2} loss: {3}, score: {4}".format(self.id, tmp['gamesFinished'], tmp['step'], tmp['loss'], tmp['score'])
        return tmp
        
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
            self.initialState = self.environment.state
            ## check for model params by calculating the mean of all params
            params = self.module.get_params()[0]
            paramMeans = {k: v.asnumpy().mean() for k,v in params.items()}
            self.paramMean = np.mean(paramMeans.values())
            while(not self.environment.isDone()):
                self.nSteps += 1
                self.states.append(mx.nd.array(self.environment.getState()))                 
                ## do the forward pass for policy and value determination. No label known yet.
                self.module.forward(data_batch=mxT.state2a3cInput(self.environment.getState()),
                                    is_train=True)
                ## store policy. Only validActions allowed. Invalid actions are set to prob 0.
                allowedPolicies = mx.nd.zeros_like(self.module.getPolicy())
                validIdx = np.where(self.environment.getValidActions())[0]
                allowedPolicies[0, validIdx] = self.module.getPolicy()[0,validIdx]
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
                self.values.append(self.module.getValue())
                
                if self.nSteps%self.updateFrequency == 0 or self.environment.isDone():
                ## we are updating!
                    self.updateSteps = self.nSteps%self.updateFrequency
                    if self.updateSteps == 0:
                        self.updateSteps = self.updateFrequency
                    if self.verbose:
                        print "Worker{0}; game {1}; step {2}; updateSteps {3}!".format(self.id, self.gamesPlayed + 1, self.nSteps, updateSteps)
                        print '\npolicy scores:'
                        print self.module.getPolicy()
#                        print 'validActions:'
#                        print self.environment.getValidActions()
                        print 'Allowed policies:'                    
                        print allowedPolicies
                        print '\n'
                    self.meanLoss = {k:0 for k,v in self.meanLoss.items()}
                    if self.environment.isDone():
                        discountedReward = 0
                    else:
                      ## get value of new state after policy update as
                      ## future reward estimate
                      self.module.forward(data_batch=mxT.state2a3cInput(self.environment.getState()),
                                          is_train=True)
                      discountedReward = self.module.getValue()
                    for t in reversed(range(self.updateSteps)):
                    ## loop over all memory to do the update.
                        ## update reward
                        discountedReward = (self.rewardDiscount * discountedReward
                                            + self.rewards[t])
                        ## determine advantages. All are set to 0, except the one 
                        ## for the chosen policy
                        advantages = mx.nd.zeros(shape = self.module.getPolicy().shape)
                        advantages[0,self.policies[t]] = discountedReward - self.values[t]
                        ## do forward and backward pass to accumulate gradients
                        self.module.forward(data_batch=mxT.state2a3cInput(state = self.states[t],
                                                                          label = [discountedReward, advantages]),
                                            is_train=True)
                        self.meanLoss['total'] += self.module.getLoss()
                        self.meanLoss['policy'] += self.module.getPolicyLoss()
                        self.meanLoss['value'] += self.module.getValueLoss()
                        self.meanLoss['entropy'] += self.module.getEntropyLoss()
                        self.module.backward() ## gradreq is add, so gradients add up              
                    self.meanLoss = {k: float((v / self.updateSteps).asnumpy()) for k,v in self.meanLoss.items()}
                    ## send gradients to mainThread
                    self.mainThread.module.copyGradients(fromModule = self.module, clip = self.mainThread.cfg['clip'])
                    ## perform update on mainThread
                    self.mainThread.module.updateParams() ## gradients on mainThread get cleared automatically
                    ## get new parameters from mainThread
                    self.module.copyParams(fromModule = self.mainThread.module)
                        
                    ## store performance indicators after game is finished
                    tmp = self.getPerformanceIndicators( verbose=True)
                    self.mainThread.log[self.id].append(tmp)
                    self.collectDiagnosticInfo()
                    ## clear local gradients.
                    self.module.clearGradients()
                    ## clear memory
                    self.rewards     = []
                    self.values      = []
                    self.states      = []
                    self.policies    = []
            
            self.gamesPlayed += 1
            self.nSteps = 0
        ## send extendedLog to mainThread after work is finished
        self.mainThread.extendedLog.append(self.extendedLog)
        self.mainThread.gradients. append(self.gradients)
                
        
        
        
    