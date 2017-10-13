# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:49:58 2016

@author: fishsey
"""
import numpy as np
#import common
from math import sqrt

NoneValue = 111;
def initBaiasLfm():
    p = np.random.rand(userNum, feature) * (1/sqrt(feature))
    q = np.random.rand(itemNum, feature) * (1/sqrt(feature))
    bu = np.zeros(userNum)
    bi = np.zeros(itemNum)
    return np.mat(p), np.mat(q), bu, bi

def predict(u, i, p, q, bu, bi, mean):
    ret = mean + bu[u] + bi[i]
    ret += (p[u] * q[i].T)[0,0]
    return ret
     
def learningBiasLfm(n, alpha, lamd):
    p, q, bu, bi = initBaiasLfm()
    #originalAyyay = common.load(fileNameAllPui)
    global trainArray
    means = np.mean(trainArray)
    print means
    for step in range(0, n):
        number = 0
        mae = 0
        for u in range(userNum):
            for i in range(itemNum):
                if trainArray[u, i] != 111:  #如果u-i 在训练集中
                    number += 1 #number 记录训练集中的评分总数
                    rui = trainArray[u, i]
                    pui = predict(u, i, p, q, bu, bi, means)
                    eui = rui - pui
                    mae += abs(eui)
                    #修正偏置项
                    bu[u] += alpha * (eui - lamd * bu[u])                
                    bi[i] += alpha * (eui - lamd * bi[i]) 
                    #修正 p\q                 
                    temp = p[u,] + alpha * (q[i, ] * eui - lamd * p[u,])
                    q[i,] += alpha * (p[u,] * eui - lamd * q[i,])
                    p[u,] = temp  
                else:
                    bu[u] -= alpha * lamd * bu[u]               
                    bi[i] -= alpha * lamd * bi[i]
                    p[u,] -= alpha * lamd * p[u,]
                    q[i,] -= alpha * lamd * q[i,]
        nowMae = mae / number
        print 'step: %d      mae: %f' % ((step + 1), nowMae)
        alpha *= 0.9
    return p, q, bu, bi
    
    
def createArray(fileName,userNum=339, wsNum=5825):
    trainObj=np.loadtxt(fileName,dtype=float)
    #print trainObj
    userId, itemId, rt  = trainObj[:, 0], trainObj[:, 1], trainObj[:, 2]
    #print userId
    userId = np.array(userId, dtype=int) 
    itemId = np.array(itemId, dtype=int) 
    rt = np.array(rt, dtype=float)
    arrayObj = np.empty((userNum, wsNum))
    arrayObj.fill(NoneValue)
    arrayObj[userId, itemId] = rt 
    return arrayObj
    
    
if __name__ == '__main__':
    #设置参数
    sparseness = 5
    number = 2
    print sparseness, number
    userNum = 339
    itemNum = 5825
    feature = 80 #特征数
    steps = 20#迭代次数
    alpha = 0.02 #step-length
    lamd = 0.0005 #正则化参数(惩罚因子)
    #测试多个训练集，取平均误差
    trainFile=r'E:/Dataset/ws/train/sparseness%d/training%d.txt' % (sparseness,1)
    trainArray=createArray(trainFile)
    #print trainArray
    testFile=r'E:/Dataset/ws/test/sparseness%d/test%d.txt' % (sparseness,1)
    testArray=createArray(testFile)
    #fileNameAllPui = 'euiAnalysis/all-pui-%s-%d.txt' % (sparseness, number)
    p, q, bu, bi = learningBiasLfm(steps, alpha, lamd)
#==============================================================================
#     pFile = r'p%d-%s.txt' % (number, sparseness)
#     qFile = r'q%d-%s.txt' % (number, sparseness)
#     buFile = r'bu%d-%s.txt' % (number, sparseness)
#     biFile = r'bi%d-%s.txt' % (number, sparseness)
#==============================================================================
#==============================================================================
#     common.save(p, pFile)
#     common.save(q, qFile)
#     common.save(bu, buFile)
#     common.save(bi, biFile)
#==============================================================================
    
    