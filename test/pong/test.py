## -*- coding: utf-8 -*-
#"""
#Created on Sat Dec 29 08:29:53 2018
#
#@author: markus
#"""
#
#import sys
#sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/src')
#sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/pong')
#sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')
#
#from pong_env import pong_env
#import mxnetTools as mxT
#import numpy as np
#import mxnet as mx
#import cv2 as cv
#from mainThread import mainThread as mT
#import random
#def pongMaker():
#    return pong_env(seqLength = 1, useSeqLength = False, nBallsEpisode = 5)
#
#env = pongMaker()
#
#for i in range (50):
#    env.update(random.choice([0,1,2]))
#    state = env.getNetState().asnumpy().astype(np.uint8)[0,0,:,:]
#    cv.imshow('image',state)
#    cv.waitKey(0) 
#    cv.destroyAllWindows()
#    if env.isDone(): env.reset()
#state.shape
#
#    
#    
#env.render()
#env.reset()
##class convBlock(mx.gluon.HybridBlock):
##    def __init__(self, **kwargs):
##        super(convBlock, self).__init__(**kwargs)
##        self.conv1 = mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), activation = None, prefix = "c1")
##        self.elu1  = mx.gluon.nn.ELU()
##        self.conv2 = mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), activation = None, prefix = "c2")
##        self.elu2  = mx.gluon.nn.ELU()
##        self.conv3 = mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), activation = None, prefix = "c3")
##        self.elu3  = mx.gluon.nn.ELU()
##        self.conv4 = mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), activation = None, prefix = "c4")
##        self.elu4  = mx.gluon.nn.ELU()
##
##    def hybrid_forward(self, F, x):
##        ## the first dimension of input is the sequence length.
##        ## input will be split in this dimension and all the computation done
##        ## on each element and concatenated back together.
##        
#
#
#
#net = mxT.a3cHybridSequential(useInitStates= True)
#net.add(mx.gluon.nn.Conv2D(channels = 1, kernel_size = (3,3), strides = (3,3), groups = 1, activation = None, prefix = "c1", use_bias= False))
#net.add(mx.gluon.nn.Conv2D(channels = 1, kernel_size = (3,3), strides = (3,3), groups = 1, activation = None, prefix = "c2", use_bias= True))
#
##net.add(mx.gluon.nn.Flatten())
##net.add(mx.gluon.nn.Dense(units = 256, activation = None,  prefix = "d1"))
##net.add(mx.gluon.nn.ELU())
##net.add(mxT.a3cOutput(n_policy = env.getValidActions().size))
#net.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
#
#
#data = mx.nd.zeros(shape = (3,1,1,9,9))
#data[0,:,:,:,:] = 1
#data[1,:,:,:,:] = 2
#data[2,:,:,:,:] = 3
##
#out = net(data)
#net.collect_params()['c1weight'].data()[0,:,:,:] = 1
#net.collect_params()['c1weight'].data()[1,:,:,:] = 2
#net.collect_params()['c1weight'].data()[2,:,:,:] = 3
#out = net(data)
#
#
#test2 = [x.grad().flatten() for x in test.values()]
#newShape = (out.shape[1], out.shape[0], out.shape[2] * out.shape[3])
#out.reshape(shape = newShape)
#
#net = mxT.a3cHybridSequential(useInitStates= True)
#net.add(mx.gluon.nn.Conv2D(channels = 32, kernel_size = (3,3), strides = (2,2), padding = (1,1), activation = None, prefix = "c1"))
#net.add(mx.gluon.nn.ELU())
#net.add(mx.gluon.rnn.LSTM(3,1))
#net.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
#data = env.getNetState()
#
#out = net(data)
#
#
##    net.add(mx.gluon.nn.Dense(units = 32, activation = "relu", prefix = "fc_"))
#net.add(mxT.a3cOutput(n_policy = 2, prefix = ""))
#net.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
#
#    
#mainThread = mT(netMaker   = pongMaker , 
#                envMaker   = pongMaker, 
#                configFile = 'a3c/test/cartpole/cartpole.cfg', 
#                verbose    = False)
#
#mainThread.run()
#
#mainThread.save("a3c/test/cartpole/test", overwrite = False, savePlots=True)
#
##after = mainThread.module.get_params()[0]['fullyconnected0_weight'].asnumpy()
#
#mainThread.getPerformancePlots("test", overwrite=True)
#
#import pandas as pd
#rewardHistory = pd.DataFrame(columns = ['episode', 'reward', 'advantage'])
#rewardHistory = rewardHistory.append(pd.DataFrame({'episode': [1,1,2,2,3,3,],'reward': [1,1,2,2,3,3], 'advantage': [2,2,3,3,4,4]}, columns=rewardHistory.columns))
#rewardHistory = rewardHistory.loc[rewardHistory['episode'] != 1]
#
#
#net = mxT.a3cHybridSequential(useInitStates= True)
#net.add(mx.gluon.nn.Dense(2, prefix = "l1", use_bias=False))
#net.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
#out = net(data)
#net.set_grad_req('add')
#
#net2 = mxT.a3cHybridSequential(useInitStates= True)
#net2.add(mx.gluon.nn.Dense(2, prefix = "l1", use_bias=False))
#net2.initialize(init = mx.initializer.Xavier(), ctx= mx.cpu())
#out = net2(data)
#net2.copyParams(fromNet=net)
#net.set_grad_req('add')
#
#data = mx.nd.zeros(shape = (4))
#data[0] = 1
#data[1] = 2
#data[2] = 3
#
#label = mx.nd.zeros(shape = (4,2))
#lossFct = mx.gluon.loss.L2Loss()
#
#with mx.autograd.record():
#    out = net(data)
#    loss = lossFct(out, label)
#loss.backward()
#
#with mx.autograd.record():
#    out = net(data)
#    loss = lossFct(out, label)
#loss.backward()
#net[0].weight.grad()
#
#with mx.autograd.record():
#    out = net2(data)
#    loss = lossFct(out, label)
#    
#with mx.autograd.record(): 
#    out = net2(data)
#    loss = lossFct(out, label)
#
#loss.backward()
#net2[0].weight.grad()
#
#test = mx.nd.moments(mx.nd.concat(*[x.grad().flatten() for x in net.collect_params().values()]), axes = 0)
#test[1] = mx.nd.sqrt(test[1])
#out = net(data)
#
#import cv2 as cv
#import numpy as np
#import random
#def pongMaker():
#    return pong_env(seqLength = 1, useSeqLength = False, nBallsEpisode = 5)
#
#env = pongMaker()
#env.reset()
#
#raw = env.getRawState()
#for i in range(20): env.update(random.choice([2]))
#env.env.render()
#test = env.getNetState()
#cv.imshow('image',test[0,0,:,:].asnumpy().astype(np.uint8))
#cv.waitKey(0) 
#cv.destroyAllWindows() 
#
#raw = env.getRawState()
#cv.imshow('image',state[34:193,0:159])
#cv.waitKey(0) 
#cv.destroyAllWindows() 
#
#
#
#test = mx.nd.array([[1,0,2],[1,1,3],[2,4,5],[2,7,8]])
#test[int(test[:,0].argmax(0).asscalar()):,:]
