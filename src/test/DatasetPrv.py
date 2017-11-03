# -*- coding: utf-8 -*-
'''
Created on 2017年10月30日

@author: zwp12
'''


import time 
import numpy as np
from numba.tests.npyufunc.test_ufunc import dtype


input_path = r'E:/Dataset/rtdata.txt';


output_base_path = r'E:/Dataset/my/tt';
output_train_path = output_base_path+'/train'
output_test_path = output_base_path+'/test'


#input_path = output_train_path+'/sparseness5/training0.txt';
#input_path = r'E:/Dataset/my/mytest2.txt';
def spilter():

    xyz = (142,4532,64);
#    xyz = (10,10,10);
    
    print '初始化数据开始';
    now = time.time();
    tranSet = np.loadtxt(input_path, dtype=int);
    attrSet = np.array(tranSet[:,0:3]);
    n = np.alen(attrSet);
    targetSet = np.zeros(n,dtype=int);
    targetSet.fill(1);
    XYZ = np.zeros(xyz,dtype=int);
    for i in range(n):
        XYZ[attrSet[i,0],attrSet[i,1],attrSet[i,2]] +=targetSet[i];
    
    del tranSet;
    del attrSet;
    del targetSet;
    print '初始化数据完成，耗时 %.2f秒，数据总条数%d  '%((time.time() - now),n);
    print XYZ;
    XY = np.sum(XYZ,2);
    XZ = np.sum(XYZ,1);
    YZ = np.sum(XYZ,0);
    
    print XY.shape
    print XY;
    
    np.savetxt(output_base_path+'/xy.txt', XY,'%d');
    
    print '-----------'
    print XZ.shape
    print XZ;

    np.savetxt(output_base_path+'/xz.txt', XZ,'%d');
    print '-----------'
    
    print YZ.shape
    print YZ;
    
    np.savetxt(output_base_path+'/yz.txt', YZ,'%d');
    
    
#     for spa in spas:
#         print '开始密集度为%d的分割'%(spa);
#         snow = time.time();
#         tran_size = int(spa / 100.0 * n);
#         print 'tran_size=',tran_size;
#         for i in range(repeat_cot):
#             inow = time.time();
#             #############
#             tran_x,left_x,tran_y,left_y = train_test_split(attrSet,targetSet,train_size=tran_size);
#             tran_y = tran_y.reshape((tran_size,1));
#             new_tran = np.hstack((tran_x,tran_y));
#             del tran_x;
#             del tran_y;
#             train_path = output_train_path + r'/sparseness%d/training%d.txt'%(spa,i);
#             np.savetxt(train_path, new_tran, '%.2f');
#             del new_tran;
#             
#             test_x,left_x,test_y,left_y = train_test_split(left_x,left_y,train_size=tran_size);
#             test_y = test_y.reshape((tran_size,1));
#             new_test = np.hstack((test_x,test_y));
#             del test_x;
#             del test_y;
#             test_path = output_test_path + r'/sparseness%d/test%d.txt'%(spa,i);
#             np.savetxt(test_path, new_test, '%.2f');
#             del new_test;
#             
#             del left_x;
#             del left_y;
#             ################
#             print '--->第%d个子集分割结束，耗时%.2f秒 '%(i+1,(time.time() - inow));
#         print '集度为%d的数据分割结束，耗时%.2f秒 '%(spa,(time.time() - snow));
#     
#     print '任务完成，总耗时%d秒 '%(time.time() - now)
    



if __name__ == '__main__':
    spilter();
    pass