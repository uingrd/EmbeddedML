#!/usr/bin/python3
# coding=utf-8

import numpy as np

# 快速2D卷积的代码

####################
# (2,2)线性卷积 
#   计算{x[0],x[1]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2}
#   不考虑系数h的加法运算的话，需要3次乘法和3次加法
def conv_2_2(x,h):
    H0,H1,H2 = h[0],h[0]-h[1],h[1]
    X0,X1,X2 = x[0],x[0]-x[1],x[1]
    Y0,Y1,Y2 = H0*X0,H1*X1,H2*X2
    y0,y1,y2 = Y0,Y0-Y1+Y2,Y2
    return np.array([y0,y1,y2])


####################
# (3,3)线性卷积 
#   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
#   输出{y0,y1,y2,y3,y4}
#   不考虑系数预计算以及整数乘法的话，运算量是5次乘法和20次加法
def conv_3_3(x,h):
    H0= h[0]/2
    H1=(h[0]+h[1]+h[2])/2
    H2=(h[0]-h[1]+h[2])/6
    H3=(h[0]+2*h[1]+4*h[2])/6
    H4= h[2]
    
    X0=x[0]
    X1=x[0]+x[1]+x[2]
    X2=x[0]-x[1]+x[2]
    X3=x[0]+2*x[1]+4*x[2]
    X4=x[2]
    
    Y0=H0*X0
    Y1=H1*X1
    Y2=H2*X2
    Y3=H3*X3
    Y4=H4*X4
    
    y0= 2*Y0
    y1=-Y0+2*Y1-2*Y2-Y3+2*Y4
    y2=-2*Y0+Y1+3*Y2-Y4
    y3= Y0-Y1-Y2+Y3-2*Y4
    y4= Y4
    
    return np.array([y0,y1,y2,y3,y4])


####################
# (3,2)线性卷积 
#   计算{x[0],x[1],x[2]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2,y3}
#   不考虑系数预计算的话，运算量是4次乘法和7次加法
def conv_3_2(x,h):
    H0 =  h[0]
    H1 = (h[0]+h[1])/2
    H2 = (h[0]-h[1])/2
    H3 =  h[1]
         
    X0 = x[0]
    X1 = x[0]+x[2]+x[1]
    X2 = x[0]+x[2]-x[1]
    X3 = x[2]
         
    Y0 = H0*X0
    Y1 = H1*X1
    Y2 = H2*X2
    Y3 = H3*X3
         
    y0 = Y0
    y1 = Y1-Y2-Y3
    y2 = Y1+Y2-Y0
    y3 = Y3
    
    return np.array([y0,y1,y2,y3])


####################
# (2x2,2x2)线性卷积 
# 代码清单5-15
def conv_2x2_2x2(x,h):
    H0 = h[0,:]
    H1 = h[0,:]-h[1,:]
    H2 = h[1,:]
    
    X0 = x[0,:]
    X1 = x[0,:]-x[1,:]
    X2 = x[1,:]
    
    Y0 = np.array(conv_2_2(X0,H0)) # H0*X0
    Y1 = np.array(conv_2_2(X1,H1)) # H1*X1
    Y2 = np.array(conv_2_2(X2,H2)) # H2*X2

    y = np.array([Y0,
                  Y0-Y1+Y2,
                  Y2])
    return y


####################
# (3x3,3x3)二维线性卷积 
# 代码清单5-16
def conv_3x3_3x3(x,h):
    H0= h[0,:]/2
    H1=(h[0,:]+  h[1,:]+  h[2,:])/2
    H2=(h[0,:]-  h[1,:]+  h[2,:])/6
    H3=(h[0,:]+2*h[1,:]+4*h[2,:])/6
    H4= h[2,:]
    
    X0=x[0,:]
    X1=x[0,:]+  x[1,:]+  x[2,:]
    X2=x[0,:]-  x[1,:]+  x[2,:]
    X3=x[0,:]+2*x[1,:]+4*x[2,:]
    X4=x[2,:]
    
    Y0 = conv_3_3(X0,H0) # H0*X0
    Y1 = conv_3_3(X1,H1) # H1*X1
    Y2 = conv_3_3(X2,H2) # H2*X2
    Y3 = conv_3_3(X3,H3) # H3*X3
    Y4 = conv_3_3(X4,H4) # H4*X4
    
    y0= 2*Y0
    y1=-Y0+2*Y1-2*Y2-Y3+2*Y4
    y2=-2*Y0+Y1+3*Y2-Y4
    y3= Y0-Y1-Y2+Y3-2*Y4
    y4= Y4
    
    return np.array([y0,
                     y1,
                     y2,
                     y3,
                     y4])


####################
# (3x2,2x3)二维线性卷积 
# 代码清单5-17
def conv_3x2_2x3(x,h):
    H0 =  h[0,:]
    H1 = (h[0,:]+h[1,:])/2
    H2 = (h[0,:]-h[1,:])/2
    H3 =  h[1,:]
         
    X0 = x[0,:]
    X1 = x[0,:]+x[2,:]+x[1,:]
    X2 = x[0,:]+x[2,:]-x[1,:]
    X3 = x[2,:]
         
    Y0 = conv_3_2(H0,X0)  # H0*X0, 注意：这里调用conv_3_2时入参次序
    Y1 = conv_3_2(H1,X1)  # H1*X1
    Y2 = conv_3_2(H2,X2)  # H2*X2
    Y3 = conv_3_2(H3,X3)  # H3*X3
         
    y0 = Y0
    y1 = Y1-Y2-Y3
    y2 = Y1+Y2-Y0
    y3 = Y3
    
    return np.array([y0,
                     y1,
                     y2,
                     y3])


####################
# 线性卷积的矩阵形式计算 
# y=A((Bx).*(Ch))
def conv_mat(x,h,A,B,C):
    X=np.dot(B,x)
    H=np.dot(C,h)
    Y=H*X
    y=np.dot(A,Y)
    return y


####################
# (2,2)线性卷积的矩阵形式计算
def conv_2_2_mat(x=None,h=None):
    A=np.array([[1, 0,0],
                [1,-1,1],
                [0, 0,1]])
    B=np.array([[1, 0],
                [1,-1],
                [0, 1]])
    C=B
    if x is None or h is None: 
        return A,B,C
    else:
        return conv_mat(x,h,A,B,C)


####################
# (3,3)线性卷积的矩阵形式计算
def conv_3_3_mat(x=None,h=None):
    A=np.array([[ 2, 0, 0, 0, 0],
                [-1, 2,-2,-1, 2],
                [-2, 1, 3, 0,-1],
                [ 1,-1,-1, 1,-2],
                [ 0, 0, 0, 0, 1]])
    B=np.array([[1, 0,0],
                [1, 1,1],
                [1,-1,1],
                [1, 2,4],
                [0, 0,1]])
    C=np.array([[1./2.,     0,     0],
                [1./2., 1./2., 1./2.],
                [1./6.,-1./6., 1./6.],
                [1./6., 1./3., 2./3.],
                [    0,     0,     1]])
    if x is None or h is None: 
        return A,B,C
    else:
        return conv_mat(x,h,A,B,C)


####################
# (3,2)线性卷积的矩阵形式计算
def conv_3_2_mat(x=None,h=None):
    A=np.array([[ 1,0, 0, 0],
                [ 0,1,-1,-1],
                [-1,1, 1, 0],
                [ 0,0, 0, 1]])
    B=np.array([[1, 0,0],
                [1, 1,1],
                [1,-1,1],
                [0, 0,1]])
    C=np.array([[    1,     0],
                [1./2., 1./2.],
                [1./2.,-1./2.],
                [    0,     1]])
    if x is None or h is None: 
        return A,B,C
    else:
        return conv_mat(x,h,A,B,C)


####################
# 二维线性卷积的矩阵形式 
# y=A1((B1xB2').*(C1hC2'))A2'
# 代码清单5-18
def conv_mat_2d(x,h,A1,B1,C1,A2,B2,C2):
    X=np.dot(np.dot(B1,x),B2.T)
    H=np.dot(np.dot(C1,h),C2.T)
    Y=H*X
    y=np.dot(np.dot(A1,Y),A2.T)
    return y


####################
# 代码清单 5-15
#
# (2x2,2x2)二维线性卷积 
def conv_2x2_2x2(x,h):
    H0 = h[0,:]
    H1 = h[0,:]-h[1,:]
    H2 = h[1,:]
    
    X0 = x[0,:]
    X1 = x[0,:]-x[1,:]
    X2 = x[1,:]
    
    Y0 = np.array(conv_2_2(X0,H0)) # H0*X0
    Y1 = np.array(conv_2_2(X1,H1)) # H1*X1
    Y2 = np.array(conv_2_2(X2,H2)) # H2*X2

    y = np.array([Y0,
                  Y0-Y1+Y2,
                  Y2])
    return y


####################
#
# 代码清单 5-16
#
# (3x3,3x3)二维线性卷积 
def conv_3x3_3x3(x,h):
    H0= h[0,:]/2
    H1=(h[0,:]+  h[1,:]+  h[2,:])/2
    H2=(h[0,:]-  h[1,:]+  h[2,:])/6
    H3=(h[0,:]+2*h[1,:]+4*h[2,:])/6
    H4= h[2,:]
    
    X0=x[0,:]
    X1=x[0,:]+  x[1,:]+  x[2,:]
    X2=x[0,:]-  x[1,:]+  x[2,:]
    X3=x[0,:]+2*x[1,:]+4*x[2,:]
    X4=x[2,:]
    
    Y0 = conv_3_3(X0,H0) # H0*X0
    Y1 = conv_3_3(X1,H1) # H1*X1
    Y2 = conv_3_3(X2,H2) # H2*X2
    Y3 = conv_3_3(X3,H3) # H3*X3
    Y4 = conv_3_3(X4,H4) # H4*X4
    
    y0= 2*Y0
    y1=-Y0+2*Y1-2*Y2-Y3+2*Y4
    y2=-2*Y0+Y1+3*Y2-Y4
    y3= Y0-Y1-Y2+Y3-2*Y4
    y4= Y4
    
    return np.array([y0,
                     y1,
                     y2,
                     y3,
                     y4])


####################
#
# 代码清单 5-17
#
# (3x2,2x3)二维线性卷积 
def conv_3x2_2x3(x,h):
    H0 =  h[0,:]
    H1 = (h[0,:]+h[1,:])/2
    H2 = (h[0,:]-h[1,:])/2
    H3 =  h[1,:]
         
    X0 = x[0,:]
    X1 = x[0,:]+x[2,:]+x[1,:]
    X2 = x[0,:]+x[2,:]-x[1,:]
    X3 = x[2,:]
         
    Y0 = conv_3_2(H0,X0)  # H0*X0
    Y1 = conv_3_2(H1,X1)  # H1*X1
    Y2 = conv_3_2(H2,X2)  # H2*X2
    Y3 = conv_3_2(H3,X3)  # H3*X3
         
    y0 = Y0
    y1 = Y1-Y2-Y3
    y2 = Y1+Y2-Y0
    y3 = Y3
    
    return np.array([y0,
                     y1,
                     y2,
                     y3])


####################
# (2x2,2x2)二维线性卷积的矩阵形式 
def conv_2x2_2x2_mat(x,h):
    A,B,C=conv_2_2_mat()
    if x is None or h is None:
        return A,B,C,A,B,C
    else:
        return conv_mat_2d(x,h,A,B,C,A,B,C)


####################
# (3x3,3x3)二维线性卷积的矩阵形式 
def conv_3x3_3x3_mat(x,h):
    A,B,C=conv_3_3_mat()
    if x is None or h is None:
        return A,B,C,A,B,C
    else:
        return conv_mat_2d(x,h,A,B,C,A,B,C)

 
####################
# (3x2,2x3)二维线性卷积的矩阵形式 
def conv_3x2_2x3_mat(x,h):
    A,B,C=conv_3_2_mat()
    if x is None or h is None:
        return A,B,C,A,C,B
    else:
        return conv_mat_2d(x,h,A,B,C,A,C,B)


####################
# (3x2,3x2)二维线性卷积的矩阵形式 
def conv_3x2_3x2_mat(x,h):
    A1,B1,C1=conv_3_3_mat()
    A2,B2,C2=conv_2_2_mat()
    if x is None or h is None:
        return A1,B1,C1,A2,B2,C2
    else:
        return conv_mat_2d(x,h,A1,B1,C1,A2,B2,C2)


####################
# 单元测试
####################

import scipy.signal as sp_sig

# (2,2)一维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,2).astype(float)
    h=np.random.randint(-10,10,2).astype(float)
    y_ref=np.convolve(x, h, mode='full') # 参考答案
    
    y=conv_2_2(x,h)
    print('conv_2_2()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_2_2_mat(x,h)
    print('conv_2_2_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))


# (3,3)一维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,3).astype(float)
    h=np.random.randint(-10,10,3).astype(float)
    y_ref=np.convolve(x, h, mode='full') # 参考答案
    
    y=conv_3_3(x,h)
    print('conv_3_3()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_3_3_mat(x,h)
    print('conv_3_3_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))


# (3,2)一维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,3).astype(float)
    h=np.random.randint(-10,10,2).astype(float)
    y_ref=np.convolve(x, h, mode='full') # 参考答案
    
    y=conv_3_2(x,h)
    print('conv_3_2()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_3_2_mat(x,h)
    print('conv_3_2_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))


# (2x2,2x2)二维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,(2,2)).astype(float)
    h=np.random.randint(-10,10,(2,2)).astype(float)
    y_ref=sp_sig.convolve2d(x,h)
    
    y=conv_2x2_2x2(x,h)
    print('conv_2x2_2x2()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_2x2_2x2_mat(x,h)
    print('conv_2x2_2x2_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))


# (3x3,3x3)二维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,(3,3)).astype(float)
    h=np.random.randint(-10,10,(3,3)).astype(float)
    y_ref=sp_sig.convolve2d(x,h)
    
    y=conv_3x3_3x3(x,h)
    print('conv_3x3_3x3()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_3x3_3x3_mat(x,h)
    print('conv_3x3_3x3_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))


# (3x2,2x3)二维线性卷积使用例程
if True:
    x=np.random.randint(-10,10,(3,2)).astype(float)
    h=np.random.randint(-10,10,(2,3)).astype(float)
    y_ref=sp_sig.convolve2d(x,h)
    
    y=conv_3x2_2x3(x,h)
    print('conv_3x2_2x3()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))

    y=conv_3x2_2x3_mat(x,h)
    print('conv_3x2_2x3_mat()')
    print('y:',y)
    print('y_ref:',y_ref)
    print('max(|y-y_ref|):',np.max(np.abs(y-y_ref)))






