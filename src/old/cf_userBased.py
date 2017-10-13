# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 10:17:13 2016

@author: fishsey
"""  
#缺省值设定为 111

import analysisEui_With_1   
def calMaeAndRmse():
    global testDict
    MAE = 0.0
    RMSE = 0.0
    number = 0
    euipf = r'euiAnalysis/euiNotGet.txt'
    result = np.loadtxt(euipf)
    userWsList = result[:, [-5, -4]]
#    userWsList = (analysisEui_With_1.loadData())[:, [0, 1]]
    userWsList = userWsList.tolist()
#    print userWsList
    for u in testDict:
        for i in testDict[u]:
            if [u, i] in userWsList:
                userMemberlist = topKUser(u, i)
                pui = predict(u, i, userMemberlist)
                eui = testDict[u][i] - pui
#                pf.write(str(eui) + '\n')
                RMSE += pow(eui, 2)
                MAE += abs(eui)
                number += 1
    print number
    RMSE = math.sqrt(RMSE / number)
    MAE /= number
    return MAE, RMSE

def topKUser(u, i):
    global userSimArrayObj, trainArrayObj, K
    rowNum = trainArrayObj.shape[0]
    ls = []
    #遍历每个用户
    for otherUser in range(rowNum):
        if trainArrayObj[otherUser, i] != 111 and otherUser != u:
            ls.append([userSimArrayObj[u, otherUser], otherUser])
    ls.sort(reverse = True)
    return ls[0: min(K, len(ls))]
    
def predict(u, i, clusterMemberArray):
    global userSimArrayObj, trainArrayObj, wsAverageVector
    sums = 0.0
    simSum = 0.0
    amplifier = 3
    #如果相似邻居集为空,则使用所有用户对服务i的均值评分
    if not clusterMemberArray:  
        return wsAverageVector[i]
    #相似邻居不为空, 则基于相似用户给予评分估计
    for similarity, otherId in clusterMemberArray:
        if trainArrayObj[otherId, i] != 111: #相似用户对 i有评分
            similarity = pow(similarity, amplifier)
            simSum += similarity
            sums += similarity * trainArrayObj[otherId, i]
    if sums == 0.0:#如果相似用户对 i 都未评分
        return wsAverageVector[i] 
    else:
        return (sums/simSum)

def simMinkowskiDist(inA, inB, n=2.0):  
    tempA = []
    tempB = []
    for i in xrange(len(inA)):
        if inA[i] != 111 and inB[i] != 111:
            tempA.append(inA[i])
            tempB.append(inB[i])
    if len(tempA) == 0:#if no common rating item, then return 0
        return 0.0
    inA = np.array(tempA)
    inB = np.array(tempB)
    #闵可夫斯基距离：欧式距离的扩展
    distance = sum(abs(inA - inB) ** n) ** (1.0/n) / len(inA)
    sim = 1.0 - distance
    return sim 

def createSimArray(trainArray, simMethod, n=2.0):
    '''返回 trainArray 行向量之间的相似度矩阵
    '''
    rowNum, columnNum = trainArray.shape
    result = np.zeros((rowNum, rowNum))
    for i in range(rowNum):
        for j in range(i):
            result[i,j] = result[j, i] = simMethod(trainArray[i], trainArray[j], n)
    return result

import math
import numpy as np
from numpy import linalg as la
from numpy import random
import paper
import copy
import sys
import copy
if __name__ == '__main__': 
    simCalMethod = paper.simMinkowskiDist
    sparseness = 'sparseness5'
    fileNumbers = 1
    K = 2
    MAE = 0.0
    RMSE = 0.0
    #参数调整 0.519026063656 1.64540380436
    #文件对象
    for i in range(1, fileNumbers+1):
        #文件对象
        trainFileName = 'dataSet/%s/training%d.txt' % (sparseness,i)
        testFileName = 'dataSet/%s/test%d.txt' % (sparseness,i)
        euiFileName = 'result/eui/euiNR-%s-test%d.txt' % (sparseness,i)
        pf = open(euiFileName, 'w')
        #load data
        testDict = paper.loadTestDataForDict(testFileName)
        trainArrayObj = paper.createArrayObj(trainFileName)
        mean = paper.calMean(trainArrayObj)
        wsAverageVector = paper.columnAverageArray(trainArrayObj, mean) 
         #相似度矩阵数据
        userSimFileName = 'result/simArraySlopeoneUser-%s-test%d.txt' % (sparseness,i)
#        userSimArrayObj = paper.createSimArray(trainArrayNorm, simCalMethod, 2.0)
#        paper.save(userSimArrayObj, userSimFileName)
        userSimArrayObj = paper.load(userSimFileName)
        #计算预测准确
        print calMaeAndRmse()
    pf.close()
    print 'ok'