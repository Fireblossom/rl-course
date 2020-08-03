# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:10:20 2020

@author: Gregor
"""
import numpy as np

x1 = -12


for i in range(20):
    y1 = int(np.round((x1+12)*19/18))
    x2 = -7
    for j in range(20):
        y2 = int(np.round((x2+7)*19/14))
        x2 +=(14/19)
        state = 20 * y1 + y2
        print("State = "+str(state))
    x1 +=(18/19)
    
    