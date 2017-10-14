# -*- coding: utf-8 -*-
'''
Created on 2017年10月14日

@author: zwp12
'''
import numpy as np
import CF.cf_userbase as ub
if __name__ == '__main__':
    
    t = np.array([[3,2,1],[3,1,2],[1,2,3]]);
    print t
    #  print np.sort(t);
    # ordt = np.argsort(t);
    
    #print t.shape[0]
    S,W,MeanX=ub.neighbourhood(t, 3);
    print S
    print W
    print MeanX
    
    
    
    