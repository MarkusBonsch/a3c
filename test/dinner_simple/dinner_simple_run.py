# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/src')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple')
sys.path.insert(0,'C:/users/markus_2/Documents/Nerding/python/plotting')
import os
import shutil
#os.chdir("Documents/Nerding/python/")

from dinner_simple_env import dinner_env
import mxnetTools as mxT
import pretrainedLayers as pL
import mxnet as mx
from mainThread import mainThread as mT


def dinnerMaker():
    return dinner_env(seqLength=1, 
                      useSeqLength=False,
                      nMinTeams = 9,
                      nMaxTeams = 24,
                      padSize = 24,
                      restrictValidActions= False,
                      wishStarterProbability=1/3,
                      wishMainCourseProbability=1/3,
                      wishDessertProbability=1/3
                      )

test = dinnerMaker()

nTeams = test.getNTeams()
inSize = test.getNetState().shape[2] 
nTeamVars = test.getNTeamVars()


def netMaker():
    net = mxT.a3cHybridSequential(useInitStates= True)
    net.add(pL.freeSeatSelectorLayer(env = test, fixParams = True, prefix = "FSSL_"))
    net.add(mx.gluon.nn.ELU())
    net.add(mx.gluon.nn.Flatten())
    net.add(mxT.a3cBlock(mx.gluon.nn.Dense(units = 64, prefix = "fc1")))
    net.add(mx.gluon.nn.ELU())
    net.add(mxT.a3cOutput(n_policy = nTeams, prefix = ""))
    net.initialize(init = mx.initializer.Xavier(magnitude = 0.1), ctx= mx.cpu())
    ### ### set inital parameters from per-trained model
    # params = mx.gluon.nn.SymbolBlock.imports(symbol_file = "C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple/v8_fIS_conv6channels_fc64_24teams_pad24_attempt2/final/net-symbol.json",
    #                                   param_file  = "C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple/v8_fIS_conv6channels_fc64_24teams_pad24_attempt2/final/net-0001.params",
    #                                   input_names = ['data'])
    # net.copyParams(fromNet=params)
    return(net)

mainThread = mT(netMaker   = netMaker , 
                envMaker   = dinnerMaker, 
                configFile = 'C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple/dinner_simple.cfg')

## copy run script and environment script to output dir
if mainThread.outputDir is not None:
    if not os.path.exists(mainThread.outputDir):
                os.makedirs(mainThread.outputDir)
    shutil.copyfile('C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple/dinner_simple_run.py', os.path.join(mainThread.outputDir, 'dinner_simple_run.py'))
    shutil.copyfile('C:/users/markus_2/Documents/Nerding/python/a3c/test/dinner_simple/dinner_simple_env.py', os.path.join(mainThread.outputDir, 'dinner_simple_env.py'))
  
# mainThread.run()
def run():
    mainThread.run()
