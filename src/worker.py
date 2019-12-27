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
        ## setup the local module. Optimizer doesn't matter since updates are performed by mainThread
        self.module = a3cModule(symbol   = self.mainThread.symbol, 
                                inputDim = self.mainThread.inputDim)
        ## own copy of game environment
        self.environment = copy.deepcopy(self.mainThread.environment)
        self.updateFrequency = self.mainThread.cfg['updateFrequency']
        self.nGames = self.mainThread.cfg['nGames']
        self.gamesPlayed = 0
        self.nSteps = 0
        self.meanLoss = 0
        self.rewardDiscount = self.mainThread.cfg['rewardDiscount']
        self.lock = threading.Lock()
        self.verbose = mainThread.verbose
    
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
        tmp = {'workerId': self.id, 'step':self.nSteps, 'gamesFinished': self.gamesPlayed + 1, 'loss': self.meanLoss, 'score': score}
        if verbose: 
            print "Worker: {0}, games: {1}, step: {2} loss: {3}, score: {4}".format(self.id, tmp['gamesFinished'], tmp['step'], tmp['loss'], tmp['score'])
        return tmp
        
    def run(self):
        """
        Do the actual training
        """
        print "Worker{0} started!".format(self.id)
        rewards     = []
        values      = []
        states      = []
        policies    = []
        while(self.gamesPlayed < self.nGames):
        ## loop over the games            
            self.environment.reset() ## start a new game
            while(not self.environment.isDone()):
                self.nSteps += 1
                states.append(mx.nd.array(self.environment.getState()))                 
                ## do the forward pass for policy and value determination. No label known yet.
                self.module.forward(data_batch=mxT.state2a3cInput(self.environment.getState()),
                                    is_train=True)
                ## store policy. Only validActions allowed. Invalid actions are set to prob 0.
                allowedPolicies = mx.nd.zeros_like(self.module.getPolicy())
                allowedPolicies[0, self.environment.getValidActions()] = self.module.getPolicy()[0,self.environment.getValidActions()]
                if allowedPolicies.sum() == 0:
                    ## all valid actions have score 0. Assign equal scores
                    allowedPolicies[0, self.environment.getValidActions()] = 1 / float(self.environment.getValidActions().size)
                else:
                    ## renormalize to 1
                    allowedPolicies = allowedPolicies / allowedPolicies.sum()
                
                policies.append(mx.nd.sample_multinomial(data  = allowedPolicies,
                                                         shape = 1))
                ## store reward and value as mx.nd.arrays
                rewards.append(mx.nd.array(self.environment.getRewards())[policies[-1]])
                values.append(self.module.getValue())
                ## apply action on state. Important before update
                self.environment.update(policies[-1].asnumpy())
                if self.nSteps%self.updateFrequency == 0 or self.environment.isDone():
                ## we are updating!
                    updateSteps = self.nSteps%self.updateFrequency
                    if updateSteps == 0:
                        updateSteps = self.updateFrequency
                    if self.verbose:
                        print "Worker{0}; game {1}; step {2}; updateSteps {3}!".format(self.id, self.gamesPlayed + 1, self.nSteps, updateSteps)
#                        print '\npolicy scores:'
#                        print self.module.getPolicy()
#                        print 'validActions:'
#                        print self.environment.getValidActions()
#                        print 'Allowed policies:'                    
#                        print allowedPolicies
                        print '\n'
                    self.meanLoss = 0
                    if self.environment.isDone():
                        discountedReward = 0
                    else:
                      ## get value of new state after policy update as
                      ## future reweard estimate
                      self.module.forward(data_batch=mxT.state2a3cInput(self.environment.getState()),
                                          is_train=True)
                      discountedReward = self.module.getValue()
                    for t in reversed(range(updateSteps)):
                    ## loop over all memory to do the update.
                        ## update reward
                        discountedReward = (self.rewardDiscount * discountedReward
                                            + rewards[t])
                        ## determine advantages. All are set to 0, except the one 
                        ## for the chosen policy
                        advantages = mx.nd.zeros(shape = self.module.getPolicy().shape)
                        advantages[0,policies[t]] = discountedReward - values[t]
                        ## do forward and backward pass to accumulate gradients
                        self.module.forward(data_batch=mxT.state2a3cInput(state = states[t],
                                                                          label = [discountedReward, advantages]),
                                            is_train=True)
                        self.meanLoss += self.module.getLoss()
                        self.module.backward() ## gradreq is add, so gradients add up              
                    self.meanLoss = float((self.meanLoss / updateSteps).asnumpy())
                    ## be sure to synchronize before actual update                        
                    with self.lock:                                   
                        ## send gradients to mainThread
                        self.mainThread.module.copyGradients(fromModule = self.module)
                        ## perform update on mainThread
                        self.mainThread.module.updateParams() ## gradients on mainThread get cleared automatically
                        ## get new parameters from mainThread
                        self.module.copyParams(fromModule = self.mainThread.module)
                    ## clear local gradients.
                    self.module.clearGradients()
                    ## clear memory
                    rewards     = []
                    values      = []
                    states      = []
                    policies    = []
            
            ## store performance indicators after game is finished
            tmp = self.getPerformanceIndicators( verbose=True)
            self.mainThread.log[self.id].append(tmp)
                
            self.gamesPlayed += 1
            self.nSteps = 0
                
                
        
        
        
    