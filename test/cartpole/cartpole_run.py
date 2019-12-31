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
import mxnet as mx
from mainThread import mainThread as mT


tmp = mx.sym.Variable('data')
tmp = mx.sym.FullyConnected(data = tmp, num_hidden = 128)
#tmp = mx.sym.Dropout(data = tmp, p = 0.2)
tmp = mx.sym.Activation(data = tmp, act_type = 'relu')
tmp = mx.sym.FullyConnected(data = tmp, num_hidden = 128)
tmp = mx.sym.Activation(data = tmp, act_type = 'relu')

def cartpoleMaker():
    return cartpole_env()
mainThread = mT(tmp, cartpoleMaker, 'a3c/test/cartpole/cartpole.cfg', verbose = False)

#before = mainThread.module.get_params()[0]['fullyconnected0_weight'].asnumpy()

mainThread.run()

mainThread.module.save_checkpoint("testModel", 1, save_optimizer_states=True)

#after = mainThread.module.get_params()[0]['fullyconnected0_weight'].asnumpy()

mainThread.getPerformancePlots("test", overwrite=True)



tmp = mxT.a3cOutput(tmp,environment.getRewards().size)

mod = a3cModule(tmp, 13500)

data = mxT.state2a3cInput(environment.state)
mod.forward(data)

mod = mx.mod.Module(tmp)
mod.bind(data_shapes = [('data', (1,10))], 
         label_shapes=[('valueLabel' , (1,1)), 
                       ('advantageLabel', (1,environment.getRewards().size))],
         grad_req='add')

mod.init_params(initializer = mx.init.Xavier())
mod.init_optimizer(optimizer = 'rmsprop', optimizer_params=())
help(mx.mod.module.__init__)