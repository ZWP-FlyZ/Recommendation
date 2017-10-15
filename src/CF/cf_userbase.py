# -*- coding: utf-8 -*-
'''
Created on 2017年10月13日

@author: zwp12
'''
from numpy import int
from math import sqrt



# 相似矩阵采用参数p=2的闵可夫斯基距离计算 Wuv= sqrt(sum(abs(ruk-rvk)^p ,k=n),p)
# 
    
    


### 不同操作系统路径支持
windows_base_path = r'E:/Dataset';
linux_base_path = r'/home/zwp/Dataset'

### 是否为推荐流程
isRecommend = False;

### 数据集选择,更改数据集路径
sparseness = 5
number = 1
#tran_path = r'/ws/train/sparseness%d/training%d.txt'%(sparseness,number);
#test_path =  r'/ws/test/sparseness%d/test%d.txt'%(sparseness,number);


steps = 2#迭代次数
K = 10 #邻域项数

version = 1.0
### 以上参数必须存在并且实现

# userNum = 339
# itemNum = 5825
# feature = 80 #特征数
# alpha = 0.02 #step-length
# lamd = 0.0005 #正则化参数(惩罚因子)


tran_path = r'/my/train/sparseness%d/training%d.txt'%(sparseness,number);
test_path =  r'/my/test/sparseness%d/test%d.txt'%(sparseness,number);
userNum = 4
itemNum = 4



NoneValue = 1111;
NoneSim = -1;

import numpy as np


def initMatrix(trainFilePath,testFilePath):
    ### 从数据集文件中读取数据
    ### 返回训练集和测试集的原始矩阵
    ### R = M[u][i]->[v] T = M[u][i]->[v]
    ### return R,T
    
    print trainFilePath,testFilePath
    tranMat = np.loadtxt(trainFilePath,dtype=float);
    testMat = np.loadtxt(testFilePath,dtype=float);
    
    uId,iId,rt = tranMat[:,0],tranMat[:,1],tranMat[:,2];
    uId = np.array(uId,dtype=int)
    iId = np.array(iId,dtype=int)
    rt  = np.array(rt,dtype=float)
    tranMat = np.empty((userNum,itemNum));
    tranMat.fill(NoneValue);
    #print uId,iId,rt
    tranMat[uId,iId] = rt;
    
    uId,iId,rt = testMat[:,0],testMat[:,1],testMat[:,2];
    uId = np.array(uId,dtype=int);
    iId = np.array(iId,dtype=int);
    rt =  np.array(rt);
    testMat = np.empty((userNum,itemNum));
    testMat.fill(NoneValue);
    testMat[uId,iId] = rt;
    
    print tranMat
    print testMat    
    return np.mat(tranMat),np.mat(testMat);

def compleMatrix(R,T):
    ### 使用特殊方法更改矩阵R,T中稀疏部分元素，以适应之后预测计算
    ### 如果不需要更改，直接返回R,T
    ### 返回补全后的矩阵,R,T
    ### return R,T
    return R,T

def neighbourhood(R,K):
    ### 该步骤中统一处理基于用户或者基于项目协同过滤算法获取邻域元素，
    ### 当使用基于用户的cf时，x为用户数量，否则x为物品数量。
    ### 从训练矩阵中计算出关于x的邻域矩阵：与x值最接近的K个其他x值
    ### 可以选择输出x的相似矩阵，x横向量中所有分量平均值
    ### 领域矩阵S = M[x][K] ， 相似矩阵  W = M[x][x]，平均矩阵Mean=[MeanX,MeanY]
    ### 横向平均值数组 MeanX = A[x]，竖向平均值数组 MeanY = A[y]
    ### 当需要输出其他数据时，可将数据加入Other中，提供给下一步中使用
    ### return S,W,Mean,Other
    p = 2.0;
    uc = R.shape[0];
    ic= R.shape[1];
    MeanX = np.zeros(uc,dtype=float)
    MeanY = np.zeros(ic,dtype=float)
    W = np.zeros((uc,uc));
    W.fill(-1);
    cotj = np.zeros(ic);
    sumj = np.zeros(ic);
    for i in range(uc):
        sumi=0;coti=0;
        for j in range(ic):
            if R[i,j] == NoneValue:
                continue;
            sumi += R[i,j];coti+=1;
            sumj[j] += R[i,j];cotj[j] += 1;
        MeanX[i] = sumi*1.0/coti;
    for i in range(ic):
        if cotj[i] != 0:
            MeanY[i]= sumj[i] * 1.0 / cotj[i];
    Mean = [MeanX,MeanY];
        
    for i in xrange(uc-1):
        print i
        for j in range(i+1,uc):
            ds=0.0;
            for t in range(ic):
                if R[i,t] !=NoneValue and R[j,t] != NoneValue:
                    ds += abs(R[i,t]-R[j,t])**p;
            W[i,j] = W[j,i] = 1.0 - (ds ** (1.0/p) / ic );## 本身的相似度
            S = np.argsort(-W)[:,0:K];## 排序 并得到 0到(k-1)列
    return S,W,Mean,None
                    
        
def predict(u,i,R,W,Mean,S,Other=None):
    ### 通过关系矩阵，平均数组和，邻域矩阵预测出用户u对物品i的评分
    ### return v
    MeanY = Mean[1];
    
    simU = S[u,];
    wU = W[u,];
    simSum = 0.0;
    sums = 0.0;
    pw = 3.0;
    for v in simU:
        #v = simU[k];
        sim = wU[v];
        if  sim <= 0.0:
            break;
        if R[v,i]==NoneValue: # v与i 无关时 ，跳过这一组
            continue;
        sim = pow(sim,pw);
        simSum += sim;
        sums += sim * R[v,i];
    if sums == 0.0:
        return  MeanY[i];
    else:
        return (sums *1.0/simSum);
        
def evalPredict(predc,W,S,RaT,Mean,Other=None):
    ### 评价系统预测的精度，并返回MAE，RMSE等评价结果对象
    ### 输入预测算法predc,相似度矩阵W，邻域矩阵S，训练矩阵与测试矩阵组成的数组RaT=[R,T]，平均矩阵Mean
    ### 例如 ：return {'MAE':0.0,'RMSE':0.0}
    R = RaT[0];
    T = RaT[1];
    uc = T.shape[0];
    ic= T.shape[1];
    MAE = 0.0;
    RMSE = 0.0;
    cot = 0;
    
    for u in range(uc):
        for i in range(ic):
            r = T[u,i];
            if r == NoneValue:
                continue;
            eui = r - predc(u,i,R,W,Mean,S,Other);
            MAE += abs(eui);
            RMSE += pow(eui,2.0);
            cot += 1;         
    MAE = MAE* 1.0 / cot;
    RMSE = sqrt(RMSE *1.0 / cot);
    
    print 'W:\n',W;
    print 'S:\n',S;
    print 'Mean:\n',Mean;
    return {'MAE':MAE,'RMSE':RMSE}

def recommend(u,W,S,RT,M,Other=None):
    ### 为用户u进行推荐M件物品，RT参数表示R矩阵或者T矩阵输入
    ### 返回一个最大长度为M的物品数组
    pass;

def evalRecommend(recom,W,S,RaT,Mean=None,Other=None):
    ### 评价推荐的效果，并返回推荐矩阵rank，和其他评价参数
    ### 输入推荐算法recom,相似度矩阵W，邻域矩阵S，测试矩阵T,MeanX参数不需要实现
    ### return rank
    pass;




