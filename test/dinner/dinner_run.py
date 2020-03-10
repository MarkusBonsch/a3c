# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/dinner')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')
#import os
#os.chdir("Documents/Nerding/python/")

from dinner_env import dinner_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT


def dinnerMaker():
    return dinner_env(seqLength=1, 
                      useSeqLength=False,
                      restrictValidActions= False
                      )

test = dinnerMaker()
nTeams = len(test.getValidActions())
nVars = test.getNetState().shape[2]//nTeams
def netMaker():
    net = mxT.a3cHybridSequential(useInitStates= True)
    net.add(mx.gluon.nn.Conv1D(channels = 64, kernel_size = 270, strides = 270, activation = None, prefix = "c1"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Flatten())
    net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(128, prefix = 'lstm1')))
    net.add(mx.gluon.nn.ELU())
    net.add(mxT.a3cOutput(n_policy = 50, prefix = ""))
    net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
    ## set inital parameters from per-trained model
    params = mx.gluon.nn.SymbolBlock.imports(symbol_file = "/home/markus/Documents/Nerding/python/a3c/test/dinner/test50noPadnoIntoleranceNoValidRestriction_continued/final/net-symbol.json",
                                      param_file  = "/home/markus/Documents/Nerding/python/a3c/test/dinner/test50noPadnoIntoleranceNoValidRestriction_continued/final/net-0001.params",
                                      input_names = ['data'])
    net.copyParams(fromNet=params)
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = dinnerMaker, 
                configFile = 'a3c/test/dinner/dinner.cfg')

mainThread.run()