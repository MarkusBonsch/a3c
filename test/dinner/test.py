#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:51:36 2018

@author: markus

Everything around the state: reward update, etc.
"""
import sys
sys.path.insert(0,'/home/markus/Documents/Nerding/python/dinnerTest/src')

from state import state
from assignDinnerCourses import assignDinnerCourses
from randomDinnerGenerator import randomDinnerGenerator

from datetime import datetime


Dinner1 = randomDinnerGenerator(numberOfTeams=20
                                    ,centerAddress={'lat':53.551086, 'lng':9.993682}
                                    ,radiusInMeter=5000
                                    ,wishStarterProbability=0.3
                                    ,wishMainCourseProbability=0.4
                                    ,wishDessertProbability=0.3
                                    ,rescueTableProbability=0.5
                                    ,meatIntolerantProbability=0
                                    ,animalProductsIntolerantProbability=0
                                    ,lactoseIntolerantProbability=0
                                    ,fishIntolerantProbability=0
                                    ,seafoodIntolerantProbability=0
                                    ,dogsIntolerantProbability=0
                                    ,catsIntolerantProbability=0
                                    ,dogFreeProbability=0
                                    ,catFreeProbability=0
                                    ,verbose=0
                                    ,checkValidity = True)
dinner,finalPartyLocation=Dinner1.generateDinner()

assigner = assignDinnerCourses(dinner, finalPartyLocation)
dinnerAssigned = assigner.assignDinnerCourses()

dinnerTime = datetime(2018, 07, 01, 20)
environment = state(data = dinnerAssigned, dinnerTime = dinnerTime, travelMode = 'simple')

