# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:52:30 2018

@author: markus
"""
import pdb
import mxnet as mx
from mxnet import gluon
import threading
import os

class a3cBlock(gluon.HybridBlock):
    """
    extensions for a3c.
    Currently just a wrapper that adds the reset() function
    """
    def __init__(self, block = None, **kwargs):
        super(a3cBlock, self).__init__(**kwargs)
        if block is not None:
            self.block = block
    
    def hybrid_forward(self, F, x):
        self.block(x) ## pass forward call to inner block
    
    def reset(self):
        """
        e.g. for resetting states
        Needs to be overwritten on inheritance if required
        """
#        print("Resetting Block {0}.\n".format(self.name))
        
            
class a3cOutput(a3cBlock):
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
                                               prefix = "policyOutput_")
            
    
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
    
    def __init__(self, valueLossScale = 0.5, entropyLossScale = 0.01, weight = None, batch_axis = None, policyClipValue = 1e-7,
                 usePPO = True,  **kwargs):
        """
        Loss function for a3c with ppo
        Inputs:
            valueLossScale (float): the scaling factor for the value loss
            entropyLossScale (float): the scaling factor for the policy entropy loss
            weight:     see mxnet.gluon.Loss
            batch_axis: see mxnet.gluon.Loss
            policyClipValue (float): policy will be clipped to [clipValue; 1-clipValue] in order to avoid problems with log and ratios
            usePPO (bool): if True, PPO loss is used, if False, standard policy gradient is used.
        """
        super(a3cLoss, self).__init__(weight, batch_axis, **kwargs)
        self.vSc = valueLossScale
        self.eSc = entropyLossScale
        self.policyClipValue = policyClipValue
        self.usePPO = usePPO
        self.valueLoss  = 0
        self.policyLoss = 0
        self.entropyLoss = 0
        
    def hybrid_forward(self, F, value, policy, valueLabel, advantageLabel, policyOldLabel = None):
        if self.usePPO and policyOldLabel is None:
            raise ValueError("if usePPO is True, the parameter policyOldLabel needs to be provided.")
        policy = F.clip(policy, self.policyClipValue, 1-self.policyClipValue)
        if self.usePPO:
            policyOldLabel = F.clip(policyOldLabel, self.policyClipValue, 1-self.policyClipValue)
            ppoRatio = F.exp(F.log(policy) - F.log(F.stop_gradient(policyOldLabel)))
        
            self.policyLoss = -F.nansum(F.minimum(ppoRatio * F.stop_gradient(advantageLabel), F.clip(ppoRatio, 0.8, 1.2) * F.stop_gradient(advantageLabel)),
                                        name = 'policyLoss')
        else: # standard policy Gradient is required.
             self.policyLoss = -F.nansum(F.log(policy) * advantageLabel, name = 'policyLoss')

        self. valueLoss = self.vSc * F.nansum(F.square((F.stop_gradient(valueLabel) - value)),
                                       name = "valueLoss")                                         
        self.entropyLoss = self.eSc * F.nansum(F.log(policy) * policy,
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
    def __init__(self, useInitStates = False, usePPO = True, **kwargs):
        super(a3cHybridSequential, self).__init__(**kwargs)
        self.lock = threading.Lock()
        self.trainer = None
        self.lossFct = []## list is a dirty trick to hide the loss from the 
        self.lossFct.append([]) ## forward computation of the sequential model
        self.lossFct[0].append(a3cLoss(usePPO = usePPO))
        self.useInitStates = useInitStates
    
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
            loss = self.lossFct[0][0](value, policy, value + 1, policy + 1, policy * 0.9)
        loss.backward()
            
        for name in self.collect_params().keys():
            self.collect_params()[name].grad()[:] = fromNet.collect_params()[name].grad()
        
    def copyParams(self, fromNet):
        """
        Copies all parameters (arg and aux) from fromModule to self
        Args:
            fromNet(mx.gluon.nn.HybridSequential): Parameters will be copied from this net.
        """
        if not sorted(fromNet.collect_params().keys()) == sorted(self.collect_params().keys()):
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
        
    def reset(self, initStates = None):
        """ 
        calls the reset method of all children if they are a3cBlocks
        Args: 
            initStates (dict): init States for all L=a3cLSTM layers. Key is the name of the layer.
        """
        def resetter(self):
            if isinstance(self, a3cBlock):
                if initStates is not None and self.name in initStates.keys():
                    self.reset(initStates[self.name])
                else:
                    self.reset()
        self.apply(resetter)
        
    def getInitStates(self):
        """ 
        searches for a3cLSTM layers and collects the initial parameters.
        returns: name of layer and init states.
        """
        out = {}
        for block in self._children.values():
            if isinstance(block, a3cLSTM) or isinstance(block, a3cLSTMLayer):
                out.update({block.name:  block.initStates})
        return out
        
    
    def hybrid_forward(self, F, x):
        """
        Need to overwrite forward to make it support two parameters for a3cLSTM layers
        """
        for block in self._children.values():
            if not self.useInitStates:
                x = block(x)
            else:
                if isinstance(block, a3cLSTM):
                    x = block(x, block.initStates)
                else:
                    x = block(x)
        return x
    
    
        
class a3cLSTM(a3cBlock):
    """
    Wrapper for LSTM that does the following things:
        1. Store and update initial states
        2. Return only output for last sequence
        3. Implement reset() method to set initial states to 0
    """
    def __init__(self, lstm, **kwargs):
        """
        Args:
            lstm(mxnet.gluon.rnn.LSTM): the lstm layer to be wrapped
        """
        super(a3cLSTM,self).__init__(lstm, **kwargs)
        self.defaultInitStates = self.block.begin_state(1,mx.symbol.zeros)
        self.initStates = self.defaultInitStates
        
    def hybrid_forward(self, F, x, initStates = None):
#        pdb.set_trace()
        output = self.block(x, initStates)
        if len(output) == 2:
            states = output[1]
            output = output[0]
        else: ## no states are returned
            states = None
            output = output[0]
        # select last element of output along sequence axis.
        if isinstance(self.block, mx.gluon.rnn.LSTM):
            seqDim = self.block._layout.find('T')
            output = F.reverse(output,axis = seqDim)
            output = F.slice_axis(output, axis = seqDim, begin = 0, end = 1)
        ## save last states as initStates
        if states is not None:
            self.initStates = states
        return(output)
    
    def reset(self, initStates = None):
        """
        resets the initStates
        Args:
            initStates(list of two symbols): if None, initStates will be reset to defaultInitStates.
                                             if not None, initStates will be reset to this value.
        """
        if initStates is None:
            self.initStates = self.defaultInitStates
        else:
            self.initStates = initStates


class a3cLSTMLayer(a3cBlock):
    """
    Wrapper for gluon.rnn.LSTM
    The basic idea is to store the input states to construct a sequence without having to recompute all conv net inputs
    Additionally, the following actions are performed:    
        1. Store and update initial states
        2. Return only output for last sequence
        3. Implement reset() method to set initial states to 0
    """
    def __init__(self, nHidden, seqLength, **kwargs):
        """
        Args:
            nHidden (int): the number of neurons in the lstm layer
            seqLength (int): the desired sequence length
        """
        self.seqLength = seqLength
       # construct the lstm layer
        lstm = mx.gluon.rnn.LSTM(hidden_size = nHidden, num_layers = 1, layout="TNC")
       
        super(a3cLSTMLayer,self).__init__(lstm, **kwargs)

        # initialize memory for init states
        # in this case, init states is a list of two components:
        # 1. the initStateHistory
        # 2. the inputHistory.
        self.defaultInitStates = self.block.begin_state(1,mx.symbol.zeros)
        initStateHistory = [self.defaultInitStates] * self.seqLength
        inputHistory = None
        self.initStates = [initStateHistory, inputHistory]
        
    def hybrid_forward(self, F, x, initStates = None):
#        pdb.set_trace()
        # here, we have to recycle old inputs in order to fill the inputs for the lstm
        # the layout is (sequenceLength, inputDim1, inputDim2, . . .)
        if self.initStates[1] is None: # this is the inputHistory
            # we are starting and no old inputs are present so far. We fill the whole memory with the current input.
            # first axis of the input history is sequence slot
            if isinstance(x, mx.symbol.symbol.Symbol):
                xCopy = x.__copy__()
            else: #it is an ndarray
                xCopy = x.copy()

            inputHistory = F.stop_gradient(F.expand_dims(xCopy, axis = 0))
            for i in range(self.seqLength-1):
                inputHistory = F.concat(inputHistory, F.stop_gradient(F.expand_dims(xCopy, axis = 0)), dim = 0)
        else:
            inputHistory = self.initStates[1]

        initStateHistory = self.initStates[0] 
        # add current input x the the input history to obtain the full set of inputs
        input = F.concat(inputHistory, F.expand_dims(x, axis = 0), dim = 0)
        # extract the relevant initStates from the initStateHistory (the oldest element)
        thisInitState = initStateHistory[0]
        output = self.block(input, thisInitState)
        if len(output) == 2:
            states = output[1]
            output = output[0]
        else: ## no states are returned
            states = None
            output = output[0]
        # select last element of output along sequence axis. Sequence axis is always 0 due to 'TNC' argument to constructor
        output = F.reverse(output,axis = 0)
        output = F.slice_axis(output, axis = 0, begin = 0, end = 1)
        ## update the initStateHistory: discard the oldest element and add new initStates at the end
        if states is not None:
            initStateHistory = initStateHistory[1:] + [states]
        else:
            initStateHistory = initStateHistory[1:] + [self.defaultInitStates]
        ## update the input history: discard the oldest input and add current input x at the end. Be sure to copy and stop gradients.
        inputHistory = F.concat(inputHistory[1:,], F.stop_gradient(F.expand_dims(x.copy(), axis = 0)), dim = 0)
        self.initStates[0] = initStateHistory
        self.initStates[1] = inputHistory
        return(output)
    
    def reset(self, initStates = None):
        """
        resets the initStates
        Args:
            initStates(list): list with two elements: initStateHistory and inputHistory as 
            returned by a3cLSTMLayer.getInitStates. 
            If None, initStates will be reset to defaultInitStates. If not None, initStates will be reset to this value.
        """
        if initStates is None:
            initStateHistory = [self.defaultInitStates] * self.seqLength
            inputHistory = None
            initStates = [initStateHistory, inputHistory]
        self.initStates = initStates
