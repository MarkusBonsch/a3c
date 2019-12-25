# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'dinnerTest/src')
sys.path.insert(0,'a3c/src')

from state import state
from assignDinnerCourses import assignDinnerCourses
import pandas as pd
from datetime import datetime
from mxnetTools import mxnetTools as mxT
from mxnetTools import a3cModule
import mxnet as mx
from mainThread import mainThread as mT


excel_file = pd.ExcelFile('dinnerTest/test/dinnerTest18NoIntolerance.xlsx')
#finalDinnerLocation=xl.sheet_names[0]
dinner = pd.read_excel(excel_file)
finalPartyLocation = pd.read_excel(excel_file,'final_party_location',header=None)

## assign courses
assigner = assignDinnerCourses(dinner, finalPartyLocation)
dinnerAssigned = assigner.assignDinnerCourses()

dinnerTime = datetime(2018, 07, 01, 20)
environment = state(data = dinnerAssigned, dinnerTime = dinnerTime, travelMode = 'simple')

tmp = mx.sym.Variable('data')
tmp = mx.sym.FullyConnected(data = tmp, num_hidden = 100)
tmp = mx.sym.Dropout(data = tmp, p = 0.2)
tmp = mx.sym.Activation(data = tmp, act_type = 'relu')
tmp = mx.sym.FullyConnected(data = tmp, num_hidden = 100)
tmp = mx.sym.Dropout(data = tmp, p = 0.1)
tmp = mx.sym.Activation(data = tmp, act_type = 'relu')

mainThread = mT(tmp, environment, 'a3c/test/a3c.cfg', verbose = True)

mainThread.run()






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