# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
import os
import shutil
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/test/pong')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/plotting')
# import os
os.chdir("C:/users/markus_2/Documents/Nerding/python/a3c")

from pong_env import pong_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT

def pongMaker():
    return pong_env(seqLength=1, nBallsEpisode=20, useSeqLength=False, maxStepsBall=20000)

def netMaker():
    net = mxT.a3cHybridSequential(useInitStates= True, usePPO = True)
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
   # #set inital parameters from per-trained model
   # params = mx.gluon.nn.SymbolBlock.imports(symbol_file = "C:/Users/markus_2/Documents/Nerding/python/a3c/test/pong/v3_Episode20_noNormalization_updateAfterBall/1400/net-symbol.json", param_file  = "C:/Users/markus_2/Documents/Nerding/python/a3c/test/pong/v3_Episode20_noNormalization_updateAfterBall/1400/net-0001.params", input_names = ['data'])
    #net.copyParams(fromNet=params)
    return(net)
    
mainThread = mT(netMaker   = netMaker , 
                envMaker   = pongMaker, 
                configFile = 'test/pong/pong.cfg')

## copy run script to output dir
if mainThread.outputDir is not None:
    if not os.path.exists(mainThread.outputDir):
                os.makedirs(mainThread.outputDir)
    shutil.copyfile('C:/users/markus_2/Documents/Nerding/python/a3c/test/pong/pong_run.py', os.path.join(mainThread.outputDir, 'pong_run.py'))


mainThread.run()