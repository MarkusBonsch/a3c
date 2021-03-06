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
    return pong_env(seqLength=1, nBallsEpisode=20, useSeqLength=False)

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
    net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(256, prefix = 'lstm_')))
    net.add(mx.gluon.nn.ELU())
    net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
    net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
    ## set inital parameters from per-trained model
    params = mx.gluon.nn.SymbolBlock.imports(symbol_file = "/home/markus/Documents/Nerding/python/a3c/test/pong/array/1200/net-symbol.json",
                                      param_file  = "/home/markus/Documents/Nerding/python/a3c/test/pong/array/1200/net-0001.params",
                                      input_names = ['data'])
    net.copyParams(fromNet=params)
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = pongMaker, 
                configFile = 'a3c/test/pong/pong.cfg')

mainThread.run()