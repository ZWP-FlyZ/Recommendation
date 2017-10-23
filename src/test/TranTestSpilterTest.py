# -*- coding: utf-8 -*-
'''
Created on 2017年10月22日

@author: zwp12
'''

import numpy as np
import time
from sklearn.model_selection import train_test_split

input_path = r'E:/Dataset/my/mytest2.txt';

output_base_path = r'E:/Dataset/my/tt';
output_train_path = output_base_path+'/train'
output_test_path = output_base_path+'/test'

def spilter():
    tranSet = np.loadtxt(input_path, dtype=float);
    attrSet = np.array(tranSet[:,0:2]);
    targetSet = np.array(tranSet[:,2]);
    n = np.alen(attrSet);
    tran_x,left_x,tran_y,left_y = train_test_split(tranSet[:,0:2],targetSet[:,2],train_size=50000);
    tran_y = tran_y.reshape((50000,1));
    new_tran = np.hstack((tran_x,tran_y));
    del tran_x;
    del tran_y;
    train_path = output_train_path + r'/sparseness%d/training%d.txt'%(5,1);
    np.savetxt(train_path, new_tran, '%.2f');
    del new_tran;
    
    test_x,left_x,test_y,left_y = train_test_split(left_x,left_y,train_size=50000);
    test_y = test_y.reshape((50000,1));
    new_test = np.hstack((test_x,test_y));
    del test_x;
    del test_y;
    test_path = output_test_path + r'/sparseness%d/test%d.txt'%(5,1);
    np.savetxt(test_path, new_test, '%.2f');
#     del new_tran;

    
#     print np.alen(tran_x);
#     print np.alen(left_x);
#     print np.alen(tran_y);
#     print np.alen(left_y);
    pass;


if __name__ == '__main__':
    now = time.time()
    spilter();
    print '%.2f'%(time.time() - now);
    pass