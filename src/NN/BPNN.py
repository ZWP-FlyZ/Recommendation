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
                ### Gj 可能有bug
                Gj = np.mat(Gj);
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

### 一个BP神经网络的类
class BPNNet:
    ### layers_cot是一个整数数组，f激活函数，求导函数是df
    ### layers_cot[0]为输入层神经元个数，layers_cot[1] 为输出层神经元个数
    ### neuron_cot之后的元素为隐层中单元个数，若为多隐层网络时，隐层排列顺序为输出层->输出层
    def __init__(self,layers_cot,f,df):
        len_layer = len(layers_cot);
        if len_layer<2 or f is None or df is None:
            exit('BP网络初始化失败');
        self.layers = [];
        for i in range(2,len_layer):
            tmp = 'L_hid_'+str(i);
            self.layers.append(Layer(layers_cot[i],f,df,name=tmp));
        self.layers.append(Layer(layers_cot[1],f,df,name='L_out'));
    
    ### 输入向量i
    def output(self,i):
        layers = self.layers;
        out=i;
        for la in range(0,len(layers)):
            out=layers[la].output(out);
        return out;
    
    ### 预测输出向量out，真实输出y,更新所有参数
    def update(self,y,al=0.5):
        layers = self.layers;
        len_la = len(layers);
        layer_out = layers[len_la-1];
        W,G = layer_out.update(Y=y,alpha=al);
        for i in range(0,len_la-1)[::-1]:
            W,G = layers[i].update(Wij=W,Gj=G,alpha=al);
    
    def error(self,y,out):
        tmp = (y - out) ** 2 ;
        return np.sum(tmp);
            
    ### 训练集，迭代次数，学习速率
    def tran(self,tran_set,iterations=1000, alpha=0.5):
        for it in range(iterations):
            lasterr = err = 0.0;
            for kv in tran_set:
                #print kv;
                out = self.output(kv[0]);
                #print out;
                err += self.error(kv[1], out);
                self.update(kv[1], alpha)
            if lasterr != 0.0 and lasterr <err:
                print 'end hear !!! it= %d ，error= %-.10f' % (it,err)
            lasterr = err;
            if it % 50 == 0:
                print 'it= %d ，error= %-.10f' % (it,err)
            #alpha *= 0.9;
    def test(self,test_set):
        for kv in test_set:
            out = self.output(kv[0]);
            print kv[0],'-->',out;
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

def bpnn221():
    data = [
            [np.array([0,0]),np.array([1])],
            [np.array([1,0]),np.array([0])],
            [np.array([0,1]),np.array([0])],
            [np.array([1,1]),np.array([1])]
        ];
    data2 = [
        [np.array([2,2]),np.array([1])],
        [np.array([1,2]),np.array([0])],
        [np.array([0.1,0.121]),np.array([1])],
        [np.array([0.5,1]),np.array([0])]
    ];
    
    bpnn = BPNNet([2,1,2],Func,DeFunc);
    print bpnn.tran(data,iterations=1200,alpha=0.4);
    print bpnn.test(data);
    print bpnn.test(data2);


def layertest2():
    L1 = Layer(1,Func2,DeFunc2,name='L1');
    y = [1];
    #while
    t = 100;
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
        
    print  '[0,0]=',L1.output(np.array([0,0]));
    print  '[1,0]=',L1.output(np.array([1,0]));
    print  '[0,1]=',L1.output(np.array([0,1]));
    print  '[1,1]=',L1.output(np.array([1,1]));


def layertest():
    L1 = Layer(1,Func2,DeFunc2,name='L1');
    y = [1];
    #while
    t = 100;
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
        
    print  '[0,0]=',L1.output(np.array([0,0]));
    print  '[1,0]=',L1.output(np.array([1,0]));
    print  '[0,1]=',L1.output(np.array([0,1]));
    print  '[1,1]=',L1.output(np.array([1,1]));
        
if __name__ == '__main__':
    print 'BPNN';
    
    ##layertest();
    bpnn221();    
