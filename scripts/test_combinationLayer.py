## the goal is to have two seperate models that do something with their input
## then concatenate the outpu of the two networks and add additional layers
## on top

import sys
import os
os.chdir('C:/users/markus_2/Documents/Nerding/python/a3c')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')

import mxnet as mx
import mxnetTools as mxT
from mxnet import gluon
import numpy as np
import pdb

input = mx.nd.zeros(18)
input[:] = (10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,997, 998, 999)
input = mx.nd.expand_dims(input, 0)
input = mx.nd.expand_dims(input, 0)

# construct first net. IT is a fixedInputSelector
net1 = mx.gluon.nn.HybridSequential()
net1.add(mxT.fixedInputSelector(inSize = 18, nTeams = 3, nTeamVars = 5, selectedTeamVars = [0], selectedAddVars = [15]))
net1.initialize(mx.initializer.One())
net1(input)

# construct second net. It is different a fixedInputSelector
net2 = mx.gluon.nn.HybridSequential()
net2.add(mxT.fixedInputSelector(inSize = 18, nTeams = 3, nTeamVars = 5, selectedTeamVars = [3], selectedAddVars = [17], prefix = "fIS2_"))
net2.initialize(mx.initializer.One())
net2(input)

#resulting network
finalNet = mxT.a3cHybridSequential()
finalNet.add(mxT.combinationLayer(net1 = net1, net2 = net2, prefix = "combination_"))
finalNet.add(mxT.a3cBlock(mx.gluon.nn.Dense(units = 64, prefix = "fc1")))
finalNet.initialize(mx.initializer.One())
finalNet(input)
finalNet.initTrainer()
for child in finalNet._children.values():
    print("child \n")
    print(child)
net2(input)

    


