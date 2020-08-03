# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:34:36 2020

@author: Gregor
"""

import itertools
import numpy as np

n=4
numbers=(1,2)
iteration=itertools.product(numbers, repeat=n-2)
iteration = itertools.chain(iteration,(0,0))
print(list(iteration))
x1 = np.array([2,2])
x2 = np.array([1,1])
x3 =np.greater(x1, x2)
print(np.sum(x3))