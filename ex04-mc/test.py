# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:36:19 2020

@author: Gregor
"""


import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

states = np.zeros((2,10,10))
#states = np.reshape(states,(200,1))
rewards = []
for x in range(0,2):
    for y in range(3,7):
        for z in range(3,7): 
            states[x][y][z]=1
  

def f(x,y):
    return np.multiply(x,y)          
            
            
x = [1,2]
print(x[True])
print(True)

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.arange(1,11,1)
y = np.arange(12,22,1)
x,y = np.meshgrid(x,y)
z = states[0]
ax.plot_wireframe(x,y,z)
#ax.plot3D(x,y,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

liste = [0.2,0.4]
tupel = (1,2,3)
