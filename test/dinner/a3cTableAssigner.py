# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/dinnerTest/src')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/dinner')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')

from dinner_env import dinner_env
import mxnet as mx
import numpy as np
import pdb
import os
import mxnetTools as mxT


class a3cTableAssigner:
    
    def __init__(self, netDir, random = False, **kwargs):
        ## load model
        params = mx.gluon.nn.SymbolBlock.imports(symbol_file = os.path.join(netDir, 'net-symbol.json'),
                                                 param_file  = os.path.join(netDir, "net-0001.params"),
                                                 input_names = ['data'])
        self.net = mxT.a3cHybridSequential(useInitStates= True)
        self.net.add(mx.gluon.nn.Conv1D(channels = 32, kernel_size = 270, strides = 270, activation = None, prefix = "c1"))
        self.net.add(mx.gluon.nn.ELU())
        self.net.add(mx.gluon.nn.Flatten())
        self.net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(128, prefix = 'lstm1')))
        self.net.add(mx.gluon.nn.ELU())
        self.net.add(mxT.a3cOutput(n_policy = 50, prefix = ""))
        self.net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
        if not random: 
            self.net.copyParams(fromNet=params)
        self.net.hybridize()
                   
        self.env = dinner_env(seqLength=1, useSeqLength=False)


    def chooseAction(self, state, random = False):
        """
        :Args:
            -state (state object): the current state of the dinnerEvent,
            -random (bool): ignored
        :Returns:
            :float: 
                the chosen action, i.e. the teamId where the state.activeTeam is seated
                for the state.activeCourse. np.nan if state.isDone
        """
        if state.isDone():
            return np.nan
        
        ## run state through net
        self.env.state = state
        _, actionScore = self.net(self.env.getNetState())  
        
        ## reduce to valid actions
        validActions = self.env.state.getValidActions()
        validScore = mx.nd.zeros_like(actionScore)
        validScore[0,validActions] = actionScore[0,validActions]
        validScore = validScore.asnumpy()
        ## choose best action
        action = np.argmax(validScore)
        return action
        
        