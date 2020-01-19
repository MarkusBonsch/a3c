# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/pong')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')
#import os
#os.chdir("Documents/Nerding/python/")

from pong_env import pong_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT

def pongMaker():
    return pong_env(seqLength=1, useSeqLength=False)

def netMaker():
    net = mxT.a3cHybridSequential(useInitStates= True)
    net.add(mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), padding = (1,1), activation = None, prefix = "c1"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), padding = (1,1), activation = None, prefix = "c2"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), padding = (1,1), activation = None, prefix = "c3"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), padding = (1,1), activation = None, prefix = "c4"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Flatten())
#    net.add(mx.gluon.nn.Dense(units = 256, activation = None,  prefix = "d1"))
    net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(256, prefix = 'lstm_')))
    net.add(mx.gluon.nn.ELU())
    net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
    net.initialize(init = mx.initializer.Xavier(magnitude = 1), ctx= mx.cpu())
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = pongMaker, 
                configFile = 'a3c/test/pong/pong.cfg',  
                outputDir  = 'a3c/test/pong/test/out', 
                saveInterval = 50,
                verbose    = False)

mainThread.run()