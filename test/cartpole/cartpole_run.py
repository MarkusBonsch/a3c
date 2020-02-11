# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/cartpole')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')

from cartpole_env import cartpole_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT

def cartpoleMaker():
    return cartpole_env(episodeLength = 1, seqLength = 1, useSeqLength = False)

def netMaker():
    net = mxT.a3cHybridSequential(useInitStates= True)
#    net.add(mx.gluon.nn.Dense(units = 32, prefix = 'd1', flatten = True, activation='relu'))
#    net.add(mx.gluon.nn.ELU())
    net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(hidden_size = 32,
                                          prefix = "lstm_")))
    net.add(mx.gluon.nn.ELU())
#    net.add(mx.gluon.nn.Dense(units = 32, activation = "relu", prefix = "fc_"))
    net.add(mxT.a3cOutput(n_policy = 2, prefix = ""))
    net.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = cartpoleMaker, 
                configFile = 'a3c/test/cartpole/cartpole.cfg')

mainThread.run()