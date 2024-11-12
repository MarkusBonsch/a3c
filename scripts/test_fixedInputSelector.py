import sys
import os
os.chdir('C:/users/markus_2/Documents/Nerding/python/a3c')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')

import mxnet as mx
import mxnetTools as mxT
from mxnet import gluon
import numpy as np
import pdb


## test fixedInpuSelector for 
input = mx.nd.zeros(18)
input[:] = (10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,997, 998, 999)
input = mx.nd.expand_dims(input, 0)
input = mx.nd.expand_dims(input, 0)

net = mx.gluon.nn.HybridSequential()
net.add(mxT.fixedInputSelector(inSize = 18, nTeams = 3, nTeamVars = 5, selectedTeamVars = [1,3], selectedAddVars = [1,2]))
net.initialize(mx.initializer.One())
net(input)