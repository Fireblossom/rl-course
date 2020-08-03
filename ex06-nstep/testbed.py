# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:57:31 2020

@author: Gregor
"""

import matplotlib.pyplot as plt
import numpy as np

def outerFunction():
    
    a = 10
    print("Outer Function: " +str(a))
    def innerFunction():
        a = 2
        print("Inner Function: " +str(a))
    innerFunction()
    print("Outer Function 2: " +str(a))

outerFunction()

#plt.figure()
y = []
[y.append(i*100) for i in range(20)]
x = np.arange(20)*100
plt.plot(x,y)



