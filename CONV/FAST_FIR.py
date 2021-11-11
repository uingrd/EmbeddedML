#!/usr/bin/python3
# coding=utf-8

import numpy as np
np.random.seed(1234)

###################
# FIR快速滤波算法
###################

# 代码清单 5-10
#
# F(2,2)快速FIR滤波算法
# (2抽头，一次输出2个滤波结果)
# 如果不考虑滤波系数计算的话，
# 运算量是3乘4加
def fir_2_2(x0,x1,x2,h0,h1):
    m1=(x0-x1)*h1
    m2=x1*(h0+h1)
    m3=(x1-x2)*h0

    y1=m1+m2
    y2=m2-m3
    
    return y1,y2

# 算法 5-9
#
## F(2,2)快速FIR滤波算法，另一种相同复杂度的算法
def fir_2_2a(x0,x1,x2,h0,h1):
    X0=x2+x1
    X1=-x1
    X2=x1+x0

    H0=h0
    H1=h0-h1
    H2=h1

    Y0=H0*X0
    Y1=H1*X1
    Y2=H2*X2

    y1=-Y1+Y2
    y2= Y0+Y1
    return y1,y2

# 代码清单 5-11
#
## F(2,3)快速FIR滤波算法
# (3抽头，一次输出2个滤波结果)
# 如果不考虑滤波系数计算的话，
# 运算量是4乘8加
def fir_2_3(x0,x1,x2,x3,h0,h1,h2):
    m1=(x0-x2)*h2
    m2=(x1+x2)*(h0+h1+h2)/2.
    m3=(x2-x1)*(h0-h1+h2)/2.
    m4=(x1-x3)*h0

    y2=m1+m2+m3
    y3=m2-m3-m4
    
    return y2,y3

# 代码清单 5-14
#
## F(4,4)FIR滤波算法（使用F(2,2)嵌套构成）
# (4抽头，一次输出4个滤波结果)
# 如果不考虑滤波系数计算的话，
# 运算量是3*3=9乘，4*3+4+6=22加
def fir_4_4(x0,x1,x2,x3,x4,x5,x6,h0,h1,h2,h3):
    M1=fir_2_2(x0-x2,x1-x3,x2-x4,h2   ,h3   )   
    M2=fir_2_2(x2   ,x3   ,x4   ,h0+h2,h1+h3)
    M3=fir_2_2(x2-x4,x3-x5,x4-x6,h0   ,h1   )
    
    y3=M1[0]+M2[0]
    y4=M1[1]+M2[1]
    y5=M2[0]-M3[0]
    y6=M2[1]-M3[1]
    
    return y3,y4,y5,y6

# 代码清单 5-12
#
## F(2,4)FIR滤波算法（使用F(2,2)拼接构成）
# (4抽头，一次输出2个滤波结果)
# 如果不考虑滤波系数计算的话，
# 运算量是3*2=6乘，4*2+2=10加
def fir_2_4(x0,x1,x2,x3,x4,h0,h1,h2,h3):
    y0=fir_2_2(x0,x1,x2,h2,h3)
    y1=fir_2_2(x2,x3,x4,h0,h1)
    return y0[0]+y1[0],y0[1]+y1[1]

# 代码清单 5-13
#
## F(2,5)FIR滤波算法（使用F(2,2)和F(2,3)拼接构成）
# 如果不考虑滤波系数计算的话，
# (5抽头，一次输出2个滤波结果)
# 运算量是3+4=7乘，4+8+2=14加
def fir_2_5(x0,x1,x2,x3,x4,x5,h0,h1,h2,h3,h4):
    y0=fir_2_2(x0,x1,x2,h3,h4)
    y1=fir_2_3(x2,x3,x4,x5,h0,h1,h2)
    return y0[0]+y1[0],y0[1]+y1[1]


####################
# 矩阵形式的FIR快速算法
####################

## 快速FIR滤波算法的矩阵形式
def fir_mat(A,B,C,x,h):
    X=np.dot(B,x.reshape(-1,1))
    H=np.dot(C,h.reshape(-1,1))
    Y=X*H
    y=np.dot(A,Y)
    return y.ravel()

## F(2,2)，快速FIR滤波算法的矩阵形式
def fir_2_2_mat(x0,x1,x2,h0,h1):
    A=np.array([[1,1, 0],
                [0, 1,-1]])
    B=np.array([[1,-1, 0],
                [0, 1, 0],
                [0, 1,-1]])
    C=np.array([[0, 1],
                [1, 1],
                [1, 0]])
    y=fir_mat(A,B,C,np.array([x0,x1,x2]),np.array([h0,h1]))
    return y[0],y[1]

def fir_2_2a_mat(x0,x1,x2,h0,h1):
    A=np.array([[0,-1, 1],
                [1, 1, 0]])
    B=np.array([[1, 1, 0],
                [0,-1, 0],
                [0, 1, 1]])
    C=np.array([[1, 0],
                [1,-1],
                [0, 1]])
    # 注意这里x的排列顺序是[x2 x1 x0]
    y=fir_mat(A,B,C,np.array([x2,x1,x0]),np.array([h0,h1]))
    return y[0],y[1]

## F(3,2)FIR滤波算法
# (2抽头，一次输出3个滤波结果)
# y=A*[(B*x)#(C*h)]
def fir_3_2_mat(x0,x1,x2,x3,h0,h1):
    A=np.array([[1,1, 1,0],
                [0,1,-1,0],
                [0,1, 1,1]])
    B=np.array([[1, 0,-1,0],
                [0, 1, 1,0],
                [0,-1, 1,0],
                [0,-1, 0,1]])
    C=np.array([[  0,   1],
                [ 1/2,1/2],
                [-1/2,1/2],
                [  1,   0]])
    y=fir_mat(A,B,C,np.array([x0,x1,x2,x3]),np.array([h0,h1]))
    return y[0],y[1],y[2]

## F(4,3)FIR滤波算法
# (3抽头，一次输出4个滤波结果)
# y=A*[(B*x)#(C*h)]
def fir_4_3_mat(x0,x1,x2,x3,x4,x5,h0,h1,h2):
    A=np.array([[1,1, 1,1, 1,0],
                [0,1,-1,2,-2,0],
                [0,1, 1,4, 4,0],
                [0,1,-1,8,-8,1]])
    B=np.array([[4, 0,-5, 0,1,0],
                [0,-4,-4, 1,1,0],
                [0, 4,-4,-1,1,0],
                [0,-2,-1, 2,1,0],
                [0, 2,-1,-2,1,0],
                [0, 4, 0,-5,0,1]])
    C=np.array([[  0 ,  0  , 1/4],
                [-1/6,-1/6 ,-1/6 ],
                [-1/6, 1/6 ,-1/6 ],
                [ 1/6, 1/12, 1/24],
                [ 1/6,-1/12, 1/24],
                [  1 ,  0,    0  ]])
    y=fir_mat(A,B,C,np.array([x0,x1,x2,x3,x4,x5]),np.array([h0,h1,h2]))
    return y[0],y[1],y[2],y[3]

##########
# 单元测试
##########
if True:    
    import matplotlib.pylab as plt
    
    # 测试数据
    N=100
    x=np.random.randint(-10,10,100).astype(float)
    
    print('[INF] ==== verify 2-tap FIR')
    # 生成2抽头测试FIR滤波器
    h=np.random.randint(-10,10,2).astype(float)
    y_ref=np.convolve(x, h, mode='full') # 参考答案

    # 算法验证
    y=[]
    for n in range(0,N-2,2):
        y+=fir_2_2(x[n],x[n+1],x[n+2],h[0],h[1])
    plt.title('F(2,2)')
    plt.plot(y,'.-')
    plt.plot(y_ref[1:],'+-')
    plt.legend(['y','y0','yref'])
    plt.show()
    
    y=[]
    for n in range(0,N-2,2):
        y+=fir_2_2_mat(x[n],x[n+1],x[n+2],h[0],h[1])
    plt.title('F(2,2) mat')
    plt.plot(y,'.-')
    plt.plot(y_ref[1:],'+-')
    plt.legend(['y','y0','yref'])
    plt.show()

    y=[]
    for n in range(0,N-2,2):
        y+=fir_2_2a(x[n],x[n+1],x[n+2],h[0],h[1])
    plt.title('F(2,2) a')
    plt.plot(y,'.-')
    plt.plot(y_ref[1:],'+-')
    plt.legend(['y','y0','yref'])
    plt.show()
    
    y=[]
    for n in range(0,N-2,2):
        y+=fir_2_2a_mat(x[n],x[n+1],x[n+2],h[0],h[1])
    plt.title('F(2,2) a mat')
    plt.plot(y,'.-')
    plt.plot(y_ref[1:],'+-')
    plt.legend(['y','y0','yref'])
    plt.show()
    
    y=[]
    for n in range(0,N-3,3):
        y+=fir_3_2_mat(x[n],x[n+1],x[n+2],x[n+3],h[0],h[1])
    plt.title('F(3,2) mat')
    plt.plot(y,'.-')
    plt.plot(y_ref[1:],'+-')
    plt.legend(['y','y0','yref'])
    plt.show()

    print('[INF] ==== verify 3-tap FIR')    
    # 生成3抽头测试FIR滤波器
    h=np.random.randint(-10,10,3).astype(float)
    y_ref=np.convolve(x, h, mode='full') # 参考答案

    y=[]
    for n in range(0,N-3,2):
        y+=fir_2_3(x[n],x[n+1],x[n+2],x[n+3],h[0],h[1],h[2])

    plt.title('F(2,3)')
    plt.plot(y,'.-')
    plt.plot(y_ref[2:],'+-')
    plt.legend(['y','yref'])
    plt.show()

    y=[]
    for n in range(0,N-6,4):
        y+=fir_4_3_mat(x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],h[0],h[1],h[2])

    plt.title('F(4,3) mat')
    plt.plot(y,'.-')
    plt.plot(y_ref[2:],'+-')
    plt.legend(['y','yref'])
    plt.show()

    print('[INF] ==== verify 4-tap FIR')
    # 生成4抽头测试FIR滤波器
    h=np.random.randint(-10,10,4).astype(float)
    y_ref=np.convolve(x, h, mode='full')  # 参考答案

    y=[]
    for n in range(0,N-6,4):
        y+=fir_4_4(x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6],h[0],h[1],h[2],h[3])

    plt.title('F(4,4)')
    plt.plot(y,'.-')
    plt.plot(y_ref[3:],'+-')
    plt.legend(['y','yref'])
    plt.show()

    y=[]
    for n in range(0,N-4,2):
        y+=fir_2_4(x[n],x[n+1],x[n+2],x[n+3],x[n+4],h[0],h[1],h[2],h[3])

    plt.title('F(2,4)')
    plt.plot(y,'.-')
    plt.plot(y_ref[3:],'+-')
    plt.legend(['y','yref'])
    plt.show()

    print('[INF] ==== verify 5-tap FIR')
    # 生成5抽头测试FIR滤波器
    h=np.random.randint(-10,10,5).astype(float)
    y_ref=np.convolve(x, h, mode='full')  # 参考答案

    y=[]
    for n in range(0,N-5,2):
        y+=fir_2_5(x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],h[0],h[1],h[2],h[3],h[4])

    plt.title('F(2,5)')
    plt.plot(y,'.-')
    plt.plot(y_ref[4:],'+-')
    plt.legend(['y','yref'])
    plt.show()
