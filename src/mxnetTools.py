# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:52:30 2018

@author: markus
"""

import mxnet as mx
import threading

class moduleExtensions(mx.mod.Module):
    """ 
    helper class that provides some nice extensions to the module class
    """
    def __init__(self, symbol, data_names, label_names, **kwargs):
        super(moduleExtensions, self).__init__(symbol, data_names, label_names, **kwargs)
        self.lock = threading.Lock()
    
    def getGradientsNumpy(self):
        """
        Returns a list with all gradients as numpy arrays
        """
        out = []
        for exe in self._exec_group.grad_arrays:
            for g in exe:
                tmp = mx.nd.zeros_like(g)
                g.copyto(tmp)
                out.append(tmp.asnumpy())
        return(out)
        
    def clearGradients(self):
        """
        resets all gradients of a module's executor to 0
        """
        for exe in self._exec_group.grad_arrays:
            for g in exe:
                g[:] = 0
                
    def updateParams(self):
        """ 
        Performs an update of the network parameters given the gradients.
        Clears the gradients afterwards
        """
        with self.lock:
            self.update()
            self.clearGradients()
    
    def copyGradients(self, fromModule, clip = None):
        """
        Copies the gradients from fromModule to self
        Args:
            fromModule (mx.mod.Module): gradients will be copied from this module.
            clip (float): the gradients will be clipped to range [-clip, clip]
        """   
        with self.lock:
            for i, valI in enumerate(fromModule._exec_group.grad_arrays):
                for g, valG  in enumerate(valI):
                    valG.copyto(self._exec_group.grad_arrays[i][g])
                    if clip is not None:
                        self._exec_group.grad_arrays[i][g].clip(a_min = -clip, a_max = clip)
                
    def copyParams(self, fromModule):
        """
        Copies all parameters (arg and aux) from fromModule to self
        Args:
            fromModule (mx.mod.Module): Parameters will be copied from this module.
        """
        with self.lock:
            params = fromModule.get_params()
            self.set_params(arg_params = params[0], aux_params = params[1])
        

class mxnetTools:
    """
    static helper methods for mxnet
    """
    
    @staticmethod
    def a3cOutput(symbol, nPolicy, valueLossScale = 0.5, entropyLossScale = 0.01):
        """
        Adds an a3c output layer to an existing symbol.
        The a3c output layer consists of the policy loss, the policy entropy, and the value loss.
        Args:
            symbol    (mx.symbol): The base neural network symbol
            nPolicy   (int): The number of possible policies.
            valueLossScale (float): the scaling factor for the value loss
            entropyLossScale (float): the scaling factor for the policy entropy loss
        
        """
        ## value output and loss
        ## need to compute value and loss seperately in order to retrieve loss afterwards
        valueLayer = mx.sym.FullyConnected(data = symbol, num_hidden = 1, name = 'valueLayer')
        valueLabel = mx.sym.Variable('valueLabel')
        valueLoss  = mx.sym.MakeLoss(data  = valueLossScale * mx.sym.nansum((valueLabel - valueLayer)**2), 
                                     name  = 'valueLoss')
        valueOutput = mx.sym.BlockGrad(valueLayer,
                                       name = 'valueOutput')
        ## policy output. Gradients are blocked because the policy loss 
        ## needs to be constructed separately
        policyLayer = mx.sym.FullyConnected(data = symbol, num_hidden = nPolicy, name = 'policyLayer')                                                    
        
        policyOutput    = mx.sym.BlockGrad(mx.sym.softmax(data  = policyLayer),
                                           name  = 'policyOutput')
        ## policy loss needs to be constructed by hand.
        advantageLabel = mx.sym.Variable('advantageLabel')        
        policySM = mx.sym.softmax(policyLayer)
        policyLoss = -mx.sym.nansum(mx.sym.log(policySM + 1e-7) * advantageLabel)
        ## entropy loss
        entropyLoss = mx.sym.nansum(mx.sym.log(policySM + 1e-7) * policySM)
        policyLossTotal = mx.sym.MakeLoss(policyLoss + entropyLossScale * entropyLoss,
                                          name = 'policyLoss')
        ## group everything to obtain the final symbol
        result = mx.sym.Group([policyOutput, policyLossTotal, valueOutput, valueLoss])
        return result
        
    @staticmethod
    def state2a3cInput(state, label = None):
        """
        Takes the state array, flattens it and returns a DataBatch for usage
        in the forward pass of an a3cModule.
        Args:
            state (numpy array): the state
            label (list of mx.nd.arrays): optional, the labels. Need to be specified 
                                    if backward pass needs to be done
        Returns:
            mx.io.DataBatch: the flattened state
        """
        data = mx.nd.array(state)
        data = data.expand_dims(axis = 0)
        data = data.flatten()
        
        return mx.io.DataBatch(data = [data], label = label)
   
class a3cModule(moduleExtensions):
    """
    a3c Module with special constructor, forward, backward, and update functions
    """
    def __init__(self, symbol, inputDim, optimizer = 'rmsprop', optimizerArgs = (('learning_rate', 0.001), ('gamma2', 0), ('centered', True))):
        """
        Takes an a3c symbol and constructs a module from it.
        Args:
            symbol(mx.sym.symbol): the a3c symbol
            optimizer(string): the optimizer to use
            optimizerArgs(tuple of (key, value) tuples): the configuration for the optimizer
            inputDim (int): the dimension of the input vector
        """
        super(a3cModule, self).__init__(symbol,     
                                        data_names = ['data'],
                                        label_names = ['valueLabel', 'advantageLabel'])
        ## determine required shape of advantage label                                  
        nPolicy = symbol.infer_shape(data = (1,inputDim), valueLabel = (1,1))[1][0][1]
        
        self.bind(data_shapes  = [('data', (1,inputDim))], 
                  label_shapes = [('valueLabel' , (1,1)), 
                                  ('advantageLabel', (1,nPolicy))],
                  grad_req='add')
        self.init_params(initializer = mx.init.Xavier())
        self.init_optimizer(optimizer = optimizer, optimizer_params=optimizerArgs)
        
    def getValue(self):
        """
        returns the value output
        """
        return self.get_outputs()[2]
    
    def getPolicy(self):
        """
        returns the policy vector
        """
        return self.get_outputs()[0]
    
    def getPolicyLoss(self):
        """
        returns the policy loss
        """
        return self.get_outputs()[1]
        
    def getValueLoss(self):
        """
        returns the value loss
        """
        return self.get_outputs()[3]
    
    def getLoss(self):
        """
        returns the total loss
        """
        return self.getValueLoss() + self.getPolicyLoss()
                                            