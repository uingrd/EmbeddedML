#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

import pylab as plt
import IPython

####################
# 基于MCM算法的快速FIR滤波演示
# 代码清单 4-28
####################

# 计算多常数乘法
# 原始滤波器抽头：[0.0821,0.2019,0.4066,0.6703,0.9048,1,0.9048,0.6703,0.4066,0.2019,0.0821]
# 定点化的滤波器抽头：[11,26,52,86,116,128,116,86,52,26,11]
def mul_x(x):
    c1=x 
    c10=(c1<<1)+(c1<<3)
    c11=c1+c10
    c29=(c10<<2)-c11
    c43=-c1+(c11<<2)
    c13=(c1<<1)+c11
    return c11,c13,c29,c43

# 基于多常数快速乘法的滤波函数
def filter_x(x_in):
    y_out=[]
    c11,c13,c29,c43=[0]*11,[0]*11,[0]*11,[0]*11
    xn=[0]*11
    for x in x_in:
        v11,v13,v29,v43=mul_x(x)
        xn =[x  ]+xn [:10]
        c11=[v11]+c11[:10]
        c13=[v13]+c13[:10]
        c29=[v29]+c29[:10]
        c43=[v43]+c43[:10]
        #print(xn)
        
        y= c11[0]+(c13[1]<<1)+(c13[2]<<2)+(c43[3]<<1)+(c29[4]<<2)+(xn[5]<<7)+\
           (c29[6]<<2)+(c43[7]<<1)+(c13[8]<<2)+(c13[9]<<1)+c11[10]
        y_out.append(y)
        #print(y)

    return y_out

####################
## 单元测试
####################

N=100
x_in=np.random.randint(low=-128,high=127,size=N)
h=[11,26,52,86,116,128,116,86,52,26,11]
y_ref=np.convolve(x_in,h)
y_out=filter_x(x_in)

plt.plot(y_ref)
plt.plot(y_out)
plt.legend(['reference','fast algo.'])
plt.title('fast FIR filter')
plt.show()
