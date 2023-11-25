# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'C:/Users/markus_2/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'C:/Users/markus_2/Documents/Nerding/python/a3c/test/pong')
sys.path.insert(0,'C:/Users/markus_2/Documents/Nerding/python/plotting')

from pong_env import pong_env
import mxnet as mx
import numpy as np
import pdb
import time
import mxnetTools as mxT
## load model
params = mx.gluon.nn.SymbolBlock.imports(symbol_file = "C:/users/markus_2/Documents/Nerding/python/a3c/test/pong/array_continued2/960/net-symbol.json",
                                      param_file  = "C:/users/markus_2/Documents/Nerding/python/a3c/test/pong/array_continued2/960/net-0001.params",
                                      input_names = ['data'])
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

net.copyParams(fromNet=params)

net.hybridize()
                                

env = pong_env(seqLength = 1, nBallsEpisode=5)

env.reset()
net.reset()
state = env.getNetState()
env.env.render()
done = False
step = 0
while step < 10:
    ## get action
    _, policy = net(state)    
    action = np.argmax(policy.asnumpy())
    env.update(action)
    time.sleep(0.01)
    state = env.getNetState()
    env.env.render()
    if env.isPartDone():
        net.reset()
    if env.isDone(): 
        env.reset()
        net.reset()
        step +=1
    
    
