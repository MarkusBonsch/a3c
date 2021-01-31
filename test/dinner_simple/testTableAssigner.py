# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:29:53 2018

@author: markus
"""

import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/dinnerTest/src')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/a3c/test/dinner')
sys.path.insert(0,'/home/markus/Documents/Nerding/python/plotting')

import numpy as np
import pdb
import os
from dinner_env import dinner_env

class testTableAssigner:
    
    def __init__(self, **kwargs):
        self.env = dinner_env(seqLength=1, useSeqLength=False)
        
    def chooseAction(self, state, random = False):
        """
        :Args:
            -state (state object): the current state of the dinnerEvent,
            -random (bool): ignored
        :Returns:
            :float: 
                the chosen action, i.e. the teamId where the state.activeTeam is seated
                for the state.activeCourse. np.nan if state.isDone
        """
        if state.isDone():
            return np.nan
        self.env.reset(initState = state)
        action = np.random.choice(np.where(self.env.getNetState().asnumpy() == 1)[2], 1).item()
#        if not np.any(self.env.state.getValidActions() == action):
#            pdb.set_trace()
        return action
        
        