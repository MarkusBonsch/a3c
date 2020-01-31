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
import mxnet as mx
import numpy as np
import time

## load model
net = mx.gluon.nn.SymbolBlock.imports(symbol_file = "/home/markus/Documents/Nerding/python/a3c/test/cartpole/episode5/out_600/net-symbol.json",
                                      param_file  = "/home/markus/Documents/Nerding/python/a3c/test/cartpole/episode5/out_600/net-0001.params",
                                      input_names = ['data'])
net.hybridize()
                                

env = cartpole_env()

env.reset()
state = env.getNetState()
env.env.render()
done = False
step = 0
while not done:
    step +=1
    print step
    ## get action
    _, policy = net(state)    
    action = np.argmax(policy.asnumpy())
    tmp = env.update(action)
    state = env.getNetState()
#    done = tmp[2]
    env.env.render()
    if step == 300:
        env.reset()
        step = 0
    
    
