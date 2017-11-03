# -*- coding: utf-8 -*-
'''
Created on 2017年10月22日

@author: zwp12
'''

import numpy as np

path = r'E:/Dataset/my/mytest2.txt';

def savetext():
    mat = np.arange(9).reshape((3,3));
    #print mat;
    np.savetxt(path, mat,'%.2f');

    #mat_load = np.loadtxt(path, dtype=float);
    
    #print mat_load;
    
    
if __name__ == '__main__':
    savetext();
    pass;
    
