# -*- coding: utf-8 -*-
'''
Created on 2017年10月19日

@author: zwp12
'''

import numpy as np


i = np.array([1,2,3])
i = np.mat(i);

j = np.array([3,2,1])

print i + j * 2;
j = np.mat(j);
print (i*j.transpose())[0,0];




