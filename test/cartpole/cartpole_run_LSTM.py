# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'C:\\Users\\markus\\Documents\\Nerding\\python\\a3c\\src')
sys.path.insert(0,'C:\\Users\\markus\\Documents\\Nerding\\python\\a3c\\test\\cartpole')
sys.path.insert(0,'C:\\Users\\markus\\Documents\\Nerding\\python\\plotting\\src')

from cartpole_env import cartpole_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT

def cartpoleMaker():
    return cartpole_env()

def netMaker():
    net = mxT.a3cHybridSequential()
    net.add(mx.gluon.rnn.LSTM(hidden_size = 32,
                              input_size=4,
                              prefix = "lstm_"))
    net.add(mx.gluon.nn.Activation("relu"))
    net.add(mxT.a3cOutput(n_policy = 2, prefix = ""))
    net.add()
    net.initialize(init = mx.initializer.Xavier())
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = cartpoleMaker, 
                configFile = 'a3c/test/cartpole/cartpole.cfg', 
                verbose    = False)

mainThread.run()

mainThread.save("a3c/test/cartpole/smallModel", overwrite = True, savePlots=True)

#after = mainThread.module.get_params()[0]['fullyconnected0_weight'].asnumpy()

mainThread.getPerformancePlots("test", overwrite=True)