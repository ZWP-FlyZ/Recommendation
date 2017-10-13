# -*- coding: utf-8 -*-

'''
Created on 2017年10月13日

@author: zwp12

本文件是一个模板接口文件，修改参数并实现方法后，在cf_main.py文件头部更改实现后接口后的py文件
'''

### 不同操作系统路径支持
windows_base_path = r'E:/Dataset';
linux_base_path = r'/home/zwp/Dataset'

### 是否为推荐流程
isRecommend = False;

### 数据集选择
sparseness = 5
number = 1
tran_path = r'/ws/train/sparseness%d/training%d.txt'%(sparseness,number);
test_path =  r'/ws/test/sparseness%d/test%d.txt'%(sparseness,number);


steps = 20#迭代次数
K = 10 #邻域项数

### 以上参数必须存在并且实现



# userNum = 339
# itemNum = 5825
# feature = 80 #特征数
# alpha = 0.02 #step-length
# lamd = 0.0005 #正则化参数(惩罚因子)


def initMatrix(trainFilePath,testFilePath):
    ### 从数据集文件中读取数据
    ### 返回训练集和测试集的原始矩阵
    ### R = M[u][i]->[v] T = M[u][i]->[v]
    ### return R,T
    pass

def compleMatrix(R,T):
    ### 使用一个特殊值填补矩阵R,P中稀疏部分
    ### 返回补全后的矩阵,R,T
    ### return R,T
    pass;

def nbh(R,K):
    ### 该步骤中统一处理基于用户或者基于项目查找邻域
    ### 从训练数据矩阵中计算出关于x的邻域矩阵：与x值最接近的K个其他x值
    ### 可以选择输出x的相似矩阵，x横向量中所有分量平均值
    ### 领域矩阵S = M[x][K] ， 相似矩阵  W = M[x][x]，平均值数组 MeanX = A[x]
    ### return S,W,MeanX
    pass

def predict(u,i,W,MeanX,S):
    ### 通过关系矩阵，平均数组和，邻域矩阵预测出用户u对物品i的评分
    ### return v
    pass;

def recommend(u,W,S,RT,M):
    ### 为用户u进行推荐M件物品，RT参数表示R矩阵或者T矩阵输入
    ### 返回一个最大长度为M的物品数组
    pass;

def evalPredict(predc,W,S,T,MeanX):
    ### 评价系统预测的精度，并返回MAE和RMSE
    ### 输入预测算法predc,相似度矩阵W，邻域矩阵S，测试矩阵T，平均数组MeanX
    ### return MAE,RMSE
    pass;

def evalRecommend(recom,W,S,T,MeanX=None):
    ### 评价推荐的效果，并返回推荐矩阵rank，和其他评价参数
    ### 输入推荐算法recom,相似度矩阵W，邻域矩阵S，测试矩阵T,MeanX参数不需要实现
    ### return rank
    pass;


