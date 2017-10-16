# -*- coding: utf-8 -*-


'''
Created on 2017年10月13日

@author: zwp12
'''



### 更改的实现接口的py文件
import cf_userbase as cfi
###

import platform
import cf_interface

def getDataSetFileName():
    if 'Windows' in platform.system():
        basepath = cfi.windows_base_path;
    else:
        basepath = cfi.linux_base_path;
    file_train = basepath + cfi.tran_path
    file_test  = basepath + cfi.test_path
    return file_train,file_test

### 替换接口的具体实现函数

#初始化矩阵
initMatrix = cfi.initMatrix
#矩阵补全
compleMatrix = cfi.compleMatrix
#邻域矩阵计算
neighbourhood = cfi.neighbourhood
#系统预测或推荐评价

if cfi.version != cf_interface.version:
    print '接口版本不同，运行停止，请更新到最新的接口版本！'

if cfi.isRecommend:
    function = cfi.recommend
    evaluatioan = cfi.evalRecommend
else:
    function = cfi.predict
    evaluatioan = cfi.evalPredict
###

steps = cfi.steps#迭代次数
K = cfi.K #邻域项数
if __name__ == '__main__':

    tranFileName,textFileName = getDataSetFileName()
    R,T = initMatrix(tranFileName,textFileName)# 返回了训练矩阵，与测试矩阵
    R,T = compleMatrix(R,T)# 补全训练矩阵与测试矩阵
    S,W,Mean,Other = neighbourhood(R,K) # 计算邻域矩阵
    for i in range(steps):
        print 'step:%d result='%(i+1),evaluatioan(function,W,S,[R,T],Mean,Other) 
        