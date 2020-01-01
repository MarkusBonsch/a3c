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
from mxnetTools import a3cModule
from mxnetTools import mxnetTools as mxT 
import mxnet as mx
import numpy as np
import time

## load model

module = mx.mod.Module.load("teresasModel", 1)

module.bind(data_shapes  = [('data', (1,4))], 
                  label_shapes = [('valueLabel' , (1,1)), 
                                  ('advantageLabel', (1,2))],
                                  for_training=False)

env = cartpole_env().env

state = env.reset()
env.render()
time.sleep(5)
done = False
step = 0
while not done:
    step +=1
    print step
    ## get action
    module.forward(mxT.state2a3cInput(state))    
    action = np.argmax(module.get_outputs()[0].asnumpy())
    tmp = env.step(action)
    state = tmp[0]
#    done = tmp[2]
    env.render()
    
    
