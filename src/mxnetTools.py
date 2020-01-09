# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:52:30 2018

@author: markus
"""

import mxnet as mx
from mxnet import gluon
import threading
import os

class a3cOutput(gluon.HybridBlock):
    """
    a3c output layer
    with policy output (Softmax of size n_policy)
    and value output (1 node linear output)
    """
    
    def __init__(self, n_policy, **kwargs):
        """
        Constructs the a3c output block
        Inputs:
            n_policy (int): number of possible actions
        """
        super(a3cOutput, self).__init__(**kwargs)
        with self.name_scope():
            self.valueOutput  = gluon.nn.Dense(units  = 1,
                                               prefix = "valueOutput_")
            self.policyOutput = gluon.nn.Dense(units  = n_policy,
                                              prefix  = "policyOutput_")
            
    
    def hybrid_forward(self, F, x):
        """
        Forward pass through a3cOutput Block
        Inputs:
            x (symbol or nd array): input to the layer
        Output:
            list with 2 elements: valueOutput (dim = (1,1)) 
                                  and policyOutput (dim = (1, n_policy))
        """
        value  = self.valueOutput(x)
        policy = self.policyOutput(x)
        policy = F.softmax(policy)
        return (value, policy)

class a3cLoss(gluon.loss.Loss):
    """
    Loss function for a3c
    """
    
    def __init__(self, valueLossScale = 0.5, entropyLossScale = 0.01, weight = None, batch_axis = None, **kwargs):
        """
        Loss function for a3c
        Inputs:
            valueLossScale (float): the scaling factor for the value loss
            entropyLossScale (float): the scaling factor for the policy entropy loss
            weight:     see mxnet.gluon.Loss
            batch_axis: see mxnet.gluon.Loss
        """
        super(a3cLoss, self).__init__(weight, batch_axis, **kwargs)
        self.vSc = valueLossScale
        self.eSc = entropyLossScale
        self.valueLoss  = 0
        self.policyLoss = 0
        self.entroplyLoss = 0
        
    def hybrid_forward(self, F, value, policy, valueLabel, policyLabel):
        self. valueLoss = self.vSc * F.nansum(F.square((valueLabel - value) / (valueLabel + value)),
                                   name = "valueLoss")
        self.policyLoss = -F.nansum(F.log(policy + 1e-7) * policyLabel,
                                    name = 'policyLoss')
        self.entropyLoss = self.eSc * F.nansum(F.log(policy + 1e-7) * policy,
                                               name = 'entropyLoss')
        return self.valueLoss + self.policyLoss + self.entropyLoss
    
    def getPolicyLoss(self):
        """
        returns the policy loss
        """
        return self.policyLoss
        
    def getValueLoss(self):
        """
        returns the value loss
        """
        return self.valueLoss
    
    def getEntropyLoss(self):
        """
        returns the entropy regularization loss
        """
        return self.entropyLoss
    
    def getLoss(self):
        """
        returns the total loss
        """
        return self.getValueLoss() + self.getPolicyLoss() + self.getEntropyLoss()
                                            

class a3cHybridSequential(mx.gluon.nn.HybridSequential):
    """
    Some nice extensions for exchangeing gradients and parameters
    """
    def __init__(self, **kwargs):
        super(a3cHybridSequential, self).__init__(**kwargs)
        self.lock = threading.Lock()
        self.trainer = None
        self.lossFct = []## list is a dirty trick to hide the loss from the 
        self.lossFct.append([]) ## forward computation of the sequential model
        self.lossFct[0].append(a3cLoss()) 
    
    def clearGradients(self):
        """
        resets all gradients to 0
        """
        for param in self.collect_params().values():
            param.zero_grad()
            
    def set_grad_req(self, grad_req):
        """
        Changes the gradient behaviour of all parameters within the model
        Arguments:
            grad_req (str): "write", "add", or "null"
        """
        for param in self.collect_params().values():
            param.grad_req = grad_req
            
    
    def copyGradients(self, fromNet, dummyData):
        """
        Copies the gradients from fromNet to self
        Unfortunately, a dummy forward / backward execution of self needs to be
        performed in advance. Otherwise, the updater will not accept the new weights
        Args:
            fromNet(mx.gluon.nn.HybridSequential): gradients will be copied from this module.
            dummyData: dummy input data for the neural net 
        """   
        if not fromNet.collect_params().keys() == self.collect_params().keys():
            raise ValueError("both models need to have identical parameter names")
        ## unfortunately, we have to do a dummy forward - backward pass. Otherwise, 
        ## the trainer will complain
        with mx.autograd.record():
            value, policy = self.__call__(dummyData)
            loss = self.lossFct[0][0](value, policy, value + 1, policy + 1)
        loss.backward()
            
        for name in self.collect_params().keys():
            self.collect_params()[name].grad()[:] = fromNet.collect_params()[name].grad()
        
    def copyParams(self, fromNet):
        """
        Copies all parameters (arg and aux) from fromModule to self
        Args:
            fromNet(mx.gluon.nn.HybridSequential): Parameters will be copied from this net.
        """
        if not fromNet.collect_params().keys() == self.collect_params().keys():
            raise ValueError("both models need to have identical parameter names")
        for name in self.collect_params().keys():
            self.collect_params()[name].set_data(fromNet.collect_params()[name].data())
        
    def updateParams(self):
        """ 
        Performs an update of the network parameters given the gradients.
        Clears the gradients afterwards
        """
        self.trainer.step(1)
        self.clearGradients()
    
    def updateFromWorker(self, fromNet, dummyData):
        """ 
        Copies gradients from worker and performs update inside a lock
        """
        with self.lock:
            self.copyGradients(fromNet, dummyData)
            self.updateParams()
            
    def initTrainer(self,optimizer = "rmsprop", 
                    optimizerArgs = {'learning_rate': 0.001, 'gamma1': 0.9,
                                     'gamma2': 0, 'centered': True}):
        """ 
        sets up the trainer. 
        ATTENTION: must be called after 
        all blocks have been added!
        Args:
            optimizer (string): type of optimizer to be used
            optimizerArgs (dict): additional options for the optimizer
        """
        self.trainer = gluon.Trainer(params=self.collect_params(),
                                     optimizer= optimizer,
                                     optimizer_params=optimizerArgs)
        
    def save(self, outFolder, overwrite = False):
        """ 
        saves the model including symbol, parameters, and trainer
        Args:
            outFolder (str): path to the target directory where files are saved.
                           Will be created if it doesn't exist
            overwrite(bool): whether to overwrite dir if it exists.
        """
        if not os.path.exists(outFolder):
                os.makedirs(outFolder)
        elif not overwrite:
                raise  IOError('Folder already exists: ' + outFolder + '. Specify overwrite = True if needed')
        ## save net
        self.export(os.path.join(outFolder, "net"), epoch = 1)
        ## save trainer params
        self.trainer.save_states(os.path.join(outFolder, "trainer.states"))