# -*- coding: utf-8 -*-
'''
Created on 2017年10月19日

@author: zwp12
'''

### 神经网络的一些类
import numpy as np
import random
import math

def rand(a,b):
    return (b-a)*random.random() + a


### 神经元
class Neuron:
    ### 初始化单元，W是这个神经元的初始连接权。
    def __init__(self, threshold,f,W=None,name=None):
        self.threshold = threshold;
        self.W = W;
        self.f = f;
        self.name = name;
    ### 输入一个n维向量，维度n必须与 连接权的相同，否则任何其他情况将结束任务。   
    def output(self,i):
        if self.f is None or self.W is None:
            exit('单元',self.name,'未正确初始化!');
        elif self.W.shape[0] != i.shape[0]:
            exit('单元',self.name,'连接权向量不匹配');
        f = self.f;
        W = self.W;
        W = np.mat(W);
        beta = (i * W.transpose())[0,0];
        return f(beta-self.threshold);  
    def getW(self):
        return self.W;
    def setW(self,W):
        self.W = W;
    def getThreshold(self):
        return self.threshold;
    def setThreshold(self,threshold):
        self.threshold=threshold;

### 神经网络层
class Layer:
    ### 在该层中初始化n个神经元，使用f激活函数，f函数对于求导函数是df 
    def __init__(self,n,f,df,name=None):
        if n < 1:
            exit('网络层',name,'初始化错误！'); 
        self.n = n;
        if name == None:
            name = self;
        self.name = name;
        
        ### 网络层输出值向量
        self.lo = np.zeros(n);
        ### 网络层输入值向量，保留最近一次的输入值。
        self.li = None;
        
        self.f = f;
        self.df = df;
        ### 初始化n个神经元
        nl = [];
        
        for i in range(n):
            tmp=name+str('-')+str(i);
            nl.append(Neuron(rand(-0.2,0.2),f,name=tmp));
        self.neurons= nl;
    
    def output(self,i):
        nl = self.neurons;
        m = i.shape[0]; 
        if self.li is None:
            for n in nl:
                n.setW(np.random.uniform(-0.2,0.2,size=m));
            self.li = i;
        for nc in range(len(nl)):
            self.lo[nc] = nl[nc].output(i);
        return self.lo;
    
    ### 输入上一层的连接权Wij和梯度向量Gj，输出层标签值Y向量,如果Y不是None则输出层计算。
    def update(self,alpha=0.5,Wij=None,Gj=None,Y=None):
        if self.li is None:
            exit('神经网络层',self.name,'未曾输入过信号向量！');
        li = self.li;
        lo = self.lo;
        m = li.shape[0];
        n = lo.shape[0];
        nl = self.neurons;
        curW = np.zeros((m,n));
        curG = np.zeros(n);
        for j in range(n):
            curW[:,j]=nl[j].getW();
        notY =  Y is None;
        for j in range(n):
            if(notY):
                g = self.df(lo[j])*(Wij[j,:]*Gj.transpose())[0,0]
            else:
                g = self.df(lo[j])*(Y[j]-lo[j]);
            neurJ = nl[j]; 
            curG[j] = g;
            tmpW = curW[:,j] + alpha*g*li;
            tmp = neurJ.getThreshold() - alpha*g;
            neurJ.setThreshold(tmp);
            neurJ.setW(tmpW);
            curW[:,j] = tmpW;
        return curW,curG;

class BPNNet:
    def __init__(self):
        pass;
    pass;
    

### 激活函数
def Func(x):
    return math.tanh(x);
### 激活函数的导数与原函数关系，x为原函数值
def DeFunc(x):
    return 1- x** 2; 

### 激活函数
def Func2(x):
    return 1.0/ (1.0 + math.exp(-x));
### 激活函数的导数与原函数关系，x为原函数值
def DeFunc2(x):
    return x*(1- x); 

        
if __name__ == '__main__':
    print 'BPNN';
    
    L1 = Layer(1,Func2,DeFunc2,name='L1');
    it = np.zeros(4);
    it[1] = it[3] = 1;
    y = [1];
    #while
    t = 200;
    for i in range(t):
        lt = 0.9;
        print '[0,0]=',L1.output(np.array([0,0]));
        L1.update(Y=[1],alpha=lt);
        
        print '[1,0]=',L1.output(np.array([1,0]));
        L1.update(Y=[1],alpha=lt);

        print '[0,1]=',L1.output(np.array([0,1]));
        L1.update(Y=[1],alpha=lt);
        
        print '[1,1]=',L1.output(np.array([1,1]));
        L1.update(Y=[0],alpha=lt);
        
        print '---------------------------------------'
        #lt *= 0.9;
        
    print  '[1,1]=',L1.output(np.array([1,0.5]));
    
        
