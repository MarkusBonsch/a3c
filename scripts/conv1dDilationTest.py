import mxnet as mx
import sys
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/plotting')
from dinner_simple_env import dinner_env
import mxnetTools as mxT
import mxnet as mx
from mainThread import mainThread as mT
import numpy as np
from mxnet import gluon



input = mx.nd.zeros(10)
input[:] = (10,11,12,13,14,20,21,22,23,24)
input = mx.nd.expand_dims(input, 0)
input = mx.nd.expand_dims(input, 0)

# attempt to get a net that takes the 0 element of each team
test = mx.gluon.nn.Conv1D(channels = 1, kernel_size = 2, dilation = 5, strides = 1000, activation = None, prefix = "teamSummary")
test.initialize(mx.initializer.One())
test(input)


x = mx.symbol.zeros_like(input)

# attempt to get a net that takes certain prespecified elements of each team
class variablePickLayer(mxT.a3cBlock):
    def __init__(self, channels, idxFirstTeam, nTeams, nVars, **kwargs):
        """
        ## the input to the layer is assumed to be a 3dim vector with dimensions (1,1,nVars * nTeams)
        Inputs:
            channels (int): number of output channels, similar to convLayer
            idxFirstTeam (tuple of ints): index of the desired inputs for the first team. Each entry must be between 0 and nVars -1
            nTeams (int): number of teams
            nVars (int): number of variables per team
        Output: the layer will be a fully connected layer with nChannel nodes. The special feature is, that not the whole input x is connected to the two nodes,
                but only the idxFirstTeam values of x and their corresponding counterparts for the other teams.
                e.G. nTeams = 3, nVars = 4, idxFirstTeam = 2
                Here, x would be a 12 element vector: (10,11,12,13,20,21,22,23,30,31,32,33). The layer would choose only the elements 12, 22 and 32.
        """
        super(variablePickLayer, self).__init__(**kwargs)
        if max(idxFirstTeam) >= nVars:
            raise ValueError("maximum IdxFirstTeam exceeds nVars.")
        self.positions = np.array(idxFirstTeam)
        if nTeams > 1: ## add idx for other teams
            for team in range(1,nTeams):
                newPositions = self.positions + nVars
                self.positions = np.append(self.positions, newPositions)
        self.positions = mx.nd.array(self.positions)
        with self.name_scope():
            self.denseLayer  = gluon.nn.Dense(units  = channels, activation = None, prefix = "customLayer")
        
    def hybrid_forward(self, F, x):
        ## get the correct elements of x
        thisInput = F.take(x, self.positions, axis = 2)
        output = self.denseLayer(thisInput)
        return output

net = mxT.a3cHybridSequential(useInitStates= True)
net.add(variablePickLayer(channels = 1, idxFirstTeam = (1,), nTeams = 2, nVars = 5))
net.initialize(init = mx.initializer.One())
net.hybridize()
net(input)


test.initialize(mx.initializer.One())
test(input)



class convLayers(a3cBlock):
      def __init__(self, nTeams, nVars, **kwargs):
            super(a3cOutput, self).__init__(**kwargs)
            with self.name_scope(): # takes care of appropriate naming of layers
                  ## Add a conv1D layer to collect all info for each team
                  self.teamSummary = mx.gluon.nn.Conv1D(channels = 32, kernel_size = nVars, strides = nVars, activation = None, prefix = "teamSummary")
                  ## Add a conv1D layer to collect all info about free seats
                  ## Add a conv1D layer to collect all info about rela &padded team
                  self.teamSummary = mx.gluon.nn.Conv1D(channels = 32, kernel_size = nVars, strides = nVars, activation = None, prefix = "teamSummary")    