# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
import os
os.chdir('C:/users/markus_2/Documents/Nerding/python/a3c')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')

import mxnet as mx
import mxnetTools as mxT
from mxnet import gluon
import numpy as np
import pdb

# test 1: gradients are passed through to previous layers except the mxnet warning that blocks are not registered
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(gluon.nn.Dense(units  = 1, prefix = "d1_"))
net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(1, prefix = 'lstm_')))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
net.hybridize()

state = mx.nd.ones(1)
net(state)
net.collect_params()['d1_weight'].grad() # is 0 as expected

with mx.autograd.record():
    value, policy = net(state)
    loss = net.lossFct[0][0](value, policy, 1, 1, policy * 0.9)
loss.backward()
net.collect_params()['d1_weight'].grad() # is not 0 anymore, works


# test 2: value in advantage is not included in gradient calculation
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(gluon.nn.Dense(units  = 1, prefix = "d1_"))
net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(1, prefix = 'lstm_')))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
net.hybridize()

state = mx.nd.ones(1)
net(state)
net.clearGradients()
net.collect_params()['d1_weight'].grad() # is 0 as expected


with mx.autograd.record():
    value, policy = net(state)
    valueLabel = value + 1
    advantageLabel = value + 2
    loss = net.lossFct[0][0](value, policy, valueLabel, advantageLabel, mx.nd.stop_gradient(policy*0.9))
loss.backward()
gradWithValueNormal = net.collect_params()['d1_weight'].grad().asnumpy() # is not 0 anymore, works


net.clearGradients()
net.collect_params()['d1_weight'].grad() # is 0 as expected


with mx.autograd.record():
    value, policy = net(state)
    valueLabel = mx.nd.stop_gradient(value + 1)
    advantageLabel = mx.nd.stop_gradient(value + 2)
    loss = net.lossFct[0][0](value, policy, valueLabel, advantageLabel, mx.nd.stop_gradient(policy*0.9))
loss.backward()
gradWithValueStopped = net.collect_params()['d1_weight'].grad().asnumpy() # is not 0 anymore, works

gradWithValueNormal - gradWithValueStopped 



# test3: check dimensions of policy loss
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(gluon.nn.Dense(units  = 1, prefix = "d1_"))
net.add(mxT.a3cLSTM(mx.gluon.rnn.LSTMCell(1, prefix = 'lstm_')))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
net.hybridize()

state = mx.nd.ones(1)
value, policy = net(state)
thisPolicy = policy[0,0]
valueLabel = mx.nd.stop_gradient(value + 1)
advantageLabel = mx.nd.stop_gradient(value[0,0] + 2)
loss = net.lossFct[0][0](value, thisPolicy, valueLabel, advantageLabel, mx.nd.stop_gradient(thisPolicy*0.9))

# test4: check omitting parameters from trainer
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(mxT.a3cBlock(gluon.nn.Dense(units  = 1, prefix = "d1_")))
net.add(mxT.a3cBlock(gluon.nn.Dense(units  = 1, prefix = "d2_"), fixParams= True))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
net.hybridize()

# check how to obtain a list of individual parameters
allParams = net.collect_params() # lookup table with all parameters
trainerParams = mx.gluon.ParameterDict(shared = allParams)
for child in net._children.values():
    fixParams = False # normally, parameters should be updated by the trainer
    if isinstance(child, mxT.a3cBlock) and child.fixParams: # in this case, parameters should be fixed.
        fixParams = True
    if not fixParams: # parameters should be added to the trainer
        paramNames = child.collect_params().keys()
        for paramName in paramNames:
            print(paramName)
            # add this individual param to trainerParams
            trainerParams.get(paramName) # "get" tries to retrieve the parameter from "shared" if it is not yet available in trainerPArams

trainer = gluon.Trainer(params = trainerParams, optimizer = "rmsprop")

# print values of parameters
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all weights are 1 and all biases are 0.

with mx.autograd.record():
    state = mx.nd.ones(1)
    value, policy = net(state)
    thisPolicy = policy[0,0]
    valueLabel = mx.nd.stop_gradient(value + 1)
    advantageLabel = mx.nd.stop_gradient(value[0,0] + 2)
    loss = net.lossFct[0][0](value, thisPolicy, valueLabel, advantageLabel, mx.nd.stop_gradient(thisPolicy*0.9))
loss.backward() # accumulate gradients

# check that parameters are unchanged so far
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all weights are 1 and all biases are 0.

# update params using the trainer
trainer.step(1)

# check that parameters are unchanged so far
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all are changed, except for d1. Great success!

# test5: check fixParams in nested a3cBlocks

## test fixPArams with nested a3cBlocks
class myLayer(mxT.a3cBlock):
    def __init__(self, **kwargs):
        super(myLayer,self).__init__(**kwargs)
        self.layer1 = mxT.a3cBlock(mx.gluon.nn.Dense(units = 1, prefix = "nested1"))
        self.layer2 = mxT.a3cBlock(mx.gluon.nn.Dense(units = 1, prefix = "nested2"), fixParams = True)
    
    def hybrid_forward(self, F, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return(out)

net = mxT.a3cHybridSequential(useInitStates= True)
net.add(mxT.a3cBlock(myLayer()))
net.add(mxT.a3cBlock(gluon.nn.Dense(units  = 1, prefix = "d2_"), fixParams= True))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
net.hybridize()

# check how to obtain a list of individual parameters
allParams = net.collect_params() # lookup table with all parameters
trainerParams = mx.gluon.ParameterDict(shared = allParams)

def fn(x):
    print("#########################")
    print(x.name)
    print(x.params)
    if hasattr(x, "fixParams"):
        print(x.fixParams)

net.apply(fn)

for child in net._children.values():
    pdb.set_trace()
    fixParams = False # normally, parameters should be updated by the trainer
    if isinstance(child, mxT.a3cBlock) and child.fixParams: # in this case, parameters should be fixed.
        fixParams = True
    if not fixParams: # parameters should be added to the trainer
        paramNames = child.collect_params().keys()
        for paramName in paramNames:
            print(paramName)
            # add this individual param to trainerParams
            trainerParams.get(paramName) # "get" tries to retrieve the parameter from "shared" if it is not yet available in trainerPArams

trainer = gluon.Trainer(params = trainerParams, optimizer = "rmsprop")

# print values of parameters
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all weights are 1 and all biases are 0.

with mx.autograd.record():
    state = mx.nd.ones(1)
    value, policy = net(state)
    thisPolicy = policy[0,0]
    valueLabel = mx.nd.stop_gradient(value + 1)
    advantageLabel = mx.nd.stop_gradient(value[0,0] + 2)
    loss = net.lossFct[0][0](value, thisPolicy, valueLabel, advantageLabel, mx.nd.stop_gradient(thisPolicy*0.9))
loss.backward() # accumulate gradients

# check that parameters are unchanged so far
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all weights are 1 and all biases are 0.

# update params using the trainer
trainer.step(1)

# check that parameters are unchanged so far
for param in net.collect_params().values():
    print(param.name)
    print(param.data())
## all are changed, except for d1. Great success!

## test 6 test the fixedInputSelector

nTeams = 2
nTeamVars = 10
nAddVars = 3
inSize = nTeams * nTeamVars + nAddVars

selectedTeamVars = [0,5,9]
selectedAddVars = [22]
test = mxT.fixedInputSelector(inSize = inSize, nTeams = nTeams, nTeamVars = nTeamVars, selectedTeamVars = selectedTeamVars, selectedAddVars = selectedAddVars)

input = mx.nd.array([10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,26,27,28,29, 1000,1001,1002])
input = mx.nd.expand_dims(input, axis=0)
input = mx.nd.expand_dims(input, axis=0)
out = test(input)
# test that update does not change params
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(mxT.fixedInputSelector(inSize = inSize, nTeams = nTeams, nTeamVars = nTeamVars, selectedTeamVars = selectedTeamVars, selectedAddVars = selectedAddVars))
net.add(mx.gluon.nn.Flatten())
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
net.hybridize()
net.initTrainer()
out = net(input)

with mx.autograd.record():
    state = mx.nd.ones((1,23))
    value, policy = net(state)
    thisPolicy = policy[0,0]
    valueLabel = mx.nd.stop_gradient(value + 1)
    advantageLabel = mx.nd.stop_gradient(value[0,0] + 2)
    loss = net.lossFct[0][0](value, thisPolicy, valueLabel, advantageLabel, mx.nd.stop_gradient(thisPolicy*0.9))
loss.backward() # accumulate gradients

net.updateParams()
net[0].collect_params()["dense1_weight"].data() # should be unchanged.

# # test 3: policies that have an advantage label of 0 do not affect gradients.
# state = mx.nd.ones(1)

# net3outputs = mxT.a3cHybridSequential(useInitStates= True)
# net3outputs.add(gluon.nn.Dense(units  = 1, prefix = "d1_", use_bias = False))
# net3outputs.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
# net3outputs.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
# net3outputs.hybridize()

# net3outputs(state)
# net3outputs.collect_params()['policyOutput_weight'].data()
# net3outputs.collect_params()['policyOutput_bias'].data()

# # all policies are at 0.33 
# # do forward backward with policy 1 active
# advantageLabel = mx.nd.zeros(3)
# advantageLabel[0]=20000
# net3outputs.clearGradients()
# with mx.autograd.record():
#     value, policy = net3outputs(state)
#     valueLabel = mx.nd.ones(1)
#     policyOldLabel = mx.nd.ones(1)
#     loss = net3outputs.lossFct[0][0](value, policy, valueLabel, advantageLabel, policyOldLabel)
# loss.backward()
# grad_1 = net3outputs.collect_params()['d1_weight'].grad().asnumpy()

# # now change parameters, so that first policy stays the same, but others are different.
# net3outputs.clearGradients()
# net3outputs.collect_params()['policyOutput_weight'].data()
# newWeights = mx.nd.array(np.array([[1],[0],[2]]))
# net3outputs.collect_params()['policyOutput_weight'].set_data(newWeights)
# net3outputs(state)


# # net4outputs = mxT.a3cHybridSequential(useInitStates= True)
# # net4outputs.add(gluon.nn.Dense(units  = 1, prefix = "d1_", use_bias = False))
# # net4outputs.add(mxT.a3cOutput(n_policy = 4, prefix = ""))
# # net4outputs.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
# # net4outputs.hybridize()
# # net4outputs(state)
# # ## set policy weight manually so that first policy
# # net4outputs.collect_params()['policyOutput_weight'].data()
# # net4outputs.collect_params()['policyOutput_bias'].data()



# with mx.autograd.record():
#     value, policy = net(state)
#     valueLabel = value + 1
#     advantageLabel = value + 2
#     loss = net.lossFct[0][0](value, policy, valueLabel, advantageLabel, mx.nd.stop_gradient(policy*0.9))
# loss.backward()
# gradWithValueNormal = net.collect_params()['d1_weight'].grad().asnumpy() # is not 0 anymore, works

