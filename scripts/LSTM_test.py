import sys
import os
import shutil
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/test/pong')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/plotting')
# import os
os.chdir("C:/users/markus_2/Documents/Nerding/python/a3c")
import mxnet as mx
import mxnetTools as mxT

# test 1: init state recycling history
seqLength = 3
x = mx.nd.zeros((3,4)) # input with 40*60 pixels greyscale
input = mx.nd.expand_dims(x.copy(), axis = 0)
for i in range(seqLength-1):
    input = mx.nd.concat(input, mx.nd.expand_dims(x, axis = 0), dim = 0)

x[:,:] = [(1,2,3,4),(5,6, 7,8),(9,10,11,12)]
test = mxT.a3cLSTMLayer(nHidden = 5, seqLength = 5)
test.hybridize()
mx.nd.reshape(x, shape = (3,3,4))
inputHistory = mx.nd.empty((seqLength-1,)+x.shape)

inputHistory[:,]=mx.nd.stop_gradient(mx.nd.expand_dims(x.copy(),axis = 0))
input = mx.nd.concat(inputHistory, mx.nd.expand_dims(x, axis = 0), dim = 0)

initStateHistory = [[1.1, 1.2],[2.1, 2.2],[3.1, 3.2]]
initStateHistory[1:] + [[4.1,4.2]]

y = mx.symbol.zeros((3,4))
y2 = y.__copy__()
y = y.tojson()
inputHistory = mx.nd.empty((seqLength-1,)+y.infer_shape()[1][0])

# test 2: simple net with a3cLSTMLayer
input = mx.nd.array([10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,26,27,28,29, 1000,1001,1002])
input = mx.nd.expand_dims(input, axis=0)
input = mx.nd.expand_dims(input, axis=0)
# test that update does not change params
net = mxT.a3cHybridSequential(useInitStates= True)
net.add(mx.gluon.nn.Flatten())
net.add(mxT.a3cLSTMLayer(nHidden = 16, seqLength = 5))
net.add(mxT.a3cOutput(n_policy = 3, prefix = ""))
net.initialize(init = mx.initializer.Constant(1), ctx= mx.cpu())
net.hybridize()
net.initTrainer()
out = net(input)