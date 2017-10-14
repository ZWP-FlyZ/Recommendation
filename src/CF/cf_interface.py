# -*- coding: utf-8 -*-
'''
Created on 2017年10月13日

@author: zwp12

本文件是一个模板接口文件，修改参数并实现方法后，在cf_main.py文件头部更改实现后接口后的py文件

请勿在本文件中直接修改，如需实现复制以下内容并更改。
'''

### 不同操作系统路径支持
windows_base_path = r'E:/Dataset';
linux_base_path = r'/home/zwp/Dataset'

### 是否为推荐流程
isRecommend = False;

### 数据集选择,更改数据集路径
sparseness = 5
number = 1
tran_path = r'/ws/train/sparseness%d/training%d.txt'%(sparseness,number);
test_path =  r'/ws/test/sparseness%d/test%d.txt'%(sparseness,number);


steps = 20#迭代次数
K = 10 #邻域项数

### 接口版本，不需要更改
version = 1.0
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
    ### 使用特殊方法更改矩阵R,T中稀疏部分元素，以适应之后预测计算
    ### 如果不需要更改，直接返回R,T
    ### 返回补全后的矩阵,R,T
    ### return R,T
    pass;

def neighbourhood(R,K):
    ### 该步骤中统一处理基于用户或者基于项目协同过滤算法获取邻域元素，
    ### 当使用基于用户的cf时，x为用户数量，否则x为物品数量。
    ### 从训练矩阵中计算出关于x的邻域矩阵：与x值最接近的K个其他x值
    ### 可以选择输出x的相似矩阵，x横向量中所有分量平均值
    ### 领域矩阵S = M[x][K] ， 相似矩阵  W = M[x][x]，平均矩阵Mean=[MeanX,MeanY]
    ### 横向平均值数组 MeanX = A[x]，竖向平均值数组 MeanY = A[y]
    ### return S,W,Mean
    pass

def predict(u,i,W,Mean,S):
    ### 通过关系矩阵，平均数组和，邻域矩阵预测出用户u对物品i的评分
    ### return v
    pass;

def recommend(u,W,S,RT,M):
    ### 为用户u进行推荐M件物品，RT参数表示R矩阵或者T矩阵输入
    ### 返回一个最大长度为M的物品数组
    pass;

def evalPredict(predc,W,S,T,Mean):
    ### 评价系统预测的精度，并返回MAE和RMSE
    ### 输入预测算法predc,相似度矩阵W，邻域矩阵S，测试矩阵T，平均数组MeanX
    ### return MAE,RMSE
    pass;

def evalRecommend(recom,W,S,T,Mean=None):
    ### 评价推荐的效果，并返回推荐矩阵rank，和其他评价参数
    ### 输入推荐算法recom,相似度矩阵W，邻域矩阵S，测试矩阵T,MeanX参数不需要实现
    ### return rank
    pass;


