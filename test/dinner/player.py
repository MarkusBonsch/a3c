#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:08:43 2018

@author: markus
"""
import sys
sys.path.insert(0,'dinnerTest/src')
sys.path.insert(0,'a3c/src')
sys.path.insert(0,'a3c/test/dinner')
from dinnerEvent import dinnerEvent
from a3cTableAssigner import a3cTableAssigner
import pandas as pd
from datetime import datetime
import statistics
import numpy as np

excel_file = pd.ExcelFile('/home/markus/Documents/Nerding/python/a3c/test/dinner/data/dinnerTest50NoIntolerance.xlsx')
netDir = '/home/markus/Documents/Nerding/python/a3c/test/dinner/test50noPadnoIntolerance_continued/final'
#finalDinnerLocation=xl.sheet_names[0]
dinner = pd.read_excel(excel_file,'teams')
finalPartyLocation = pd.read_excel(excel_file,'final_party_location',header=None)
dinnerTime = datetime(2018, 07, 01, 20, 0, 0)

randEvent = dinnerEvent(dinnerTable = dinner, 
                      finalPartyLocation=finalPartyLocation, 
                      dinnerTime=dinnerTime, travelMode='simple', shuffleTeams = False, padSize = 50,
                      tableAssigner = a3cTableAssigner,
                      assignerArgs = {'netDir': netDir, 'random': True})

myEvent = dinnerEvent(dinnerTable = dinner, 
                      finalPartyLocation=finalPartyLocation, 
                      dinnerTime=dinnerTime, travelMode='simple', shuffleTeams = False, padSize = 50,
                      tableAssigner = a3cTableAssigner,
                      assignerArgs = {'netDir': netDir, 'random': False})

randScores = []
myScores = []
for i in range(1000):
    randTest = randEvent.assign(repCourseAssign = 1, 
                                repTableAssign = 0)
    randScores.append(randTest[1][0,4])
    myTest = myEvent.assign(repCourseAssign = 1, 
                                repTableAssign = 0)
    myScores.append(myTest[1][0,4])

print statistics.mean(myScores)
print statistics.mean(randScores)
print statistics.stdev(myScores)
print statistics.stdev(randScores)

import scipy
scipy.stats.ttest_ind(np.array(myScores), np.array(randScores))