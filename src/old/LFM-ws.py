# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 14:13:52 2017

@author: c
"""

#矩阵分解
import numpy as np
import math
import cPickle
NoneValue = 111111.0
#初始化矩阵  
def InitLFM():
    p=np.random.rand(userId,F)/math.sqrt(F)
    q=np.random.rand(itemId,F)/math.sqrt(F) 
    bu=np.zeros(userId)
    bi=np.zeros(itemId)
    return np.mat(p),np.mat(q),bu,bi
    
def predict(u,i,p,q,bu,bi,mean):
    ret = mean + bu[u] + bi[i]
    ret += (p[u] * q[i].T)[0,0]#转置
    return ret
    
def learningLFM(n,alpha,lamb):
    p,q,bu,bi=InitLFM()
    global trainArray
    for u in range(userId):
        len1=0
        sum1=0
        for i in range(itemId):
            if trainArray[u,i] != NoneValue:
                  len1+=1
                  sum1+=trainArray[u,i]
    means = sum1/len1
    print means
    for step in range(0,n):#迭代次数
        count=0
        Mae=0
        for u in range(userId):
            for i in range(itemId):
                if trainArray[u,i] != NoneValue:
                    count+=1
                    pui=predict(u,i,p,q,bu,bi,means)
                    rui=trainArray[u,i]
                    eui=rui-pui
                    #print pui,rui,eui
                    Mae+=abs(eui)
                    #修正偏置项
                    temp=p[u,]+alpha *(q[i,]*eui - lamb*p[u,])
                    q[i,]+=alpha *(p[u,]*eui - lamb*q[i,])                      
                    p[u,]=temp
                    #修正偏置项
                    bu[u]+=alpha*(eui-lamb*bu[u])
                    bi[i]+=alpha*(eui-lamb*bi[u])
                else:
                    p[u,]-=alpha *lamb*p[u,]
                    q[i,]-=alpha *lamb*q[i,]
                    bu[u]-=alpha*lamb*bu[u]
                    bi[i]-=alpha*lamb*bi[u]
        nowMae = Mae / count
        print 'step: %d      mae: %f' % ((step + 1), nowMae)
        alpha *=0.9
    return p,q,bu,bi

#转化为矩阵
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


def load(file):
    with open(file, 'rb') as f:
        obj = cPickle.load(f)
    return obj 
    
def save(obj, file):
    with open(file, 'wb') as f:
        cPickle.dump(obj, f)
        

if __name__ == '__main__':

    F=80#特征数
    n=20#迭代次数
    alpha=0.01#学习速率
    lamb=0.02#正则化参数？
    i=1
    sparseness=5
    userId=339
    itemId=5825
    trainFile=r'F:/DataSet/ws/train/sparseness%d/training%d.txt' % (sparseness,i)
    trainArray=createArray(trainFile)
    testFile=r'F:/DataSet/ws/test/sparseness%d/test%d.txt' % (sparseness,i)
    testArray=createArray(testFile)
    #print trainArray
    #print InitLFM()
    #print p,p[0],p[0,0],p[0][0]
    #print p[0],q[0]
    
    print 'LFM:'
#==============================================================================
#     originalAyyay = load(trainFile)
#     print originalAyyay
#==============================================================================
    p,q,bu,bi = learningLFM(n,alpha,lamb)
    print p,q,bu,bi
    #pFile = r'p%d-%s.txt' % (i, sparseness)
    #paper.save(p, pFile)