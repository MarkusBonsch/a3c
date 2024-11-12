import mxnet as mx
import sys
import

input = mx.nd.zeros(11)
input[:] = (10,11,12,13,14,20,21,22,23,24,999)
input = mx.nd.expand_dims(input, 0)
input = mx.nd.expand_dims(input, 0)

net = mx.gluon.nn.HybridSequential()
net.add(mx.gluon.nn.Conv1D(channels = 3, kernel_size = 5, strides = 5, activation = None, prefix = "teamSummary"))
net.initialize(mx.initializer.One())
net(input)

## test structure of dense weights
input = mx.nd.zeros(11)
input[:] = (10,11,12,13,14,20,21,22,23,24,999)
input = mx.nd.expand_dims(input, 0)
input = mx.nd.expand_dims(input, 0)

net = mx.gluon.nn.HybridSequential()
net.add(mx.gluon.nn.Dense(units = 4, use_bias=False, prefix = "dense_"))
net.initialize(mx.initializer.One())
net(input)

net[0].name
weights = net[0].collect_params()["dense_weight"].data()