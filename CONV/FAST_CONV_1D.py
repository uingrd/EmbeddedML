#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 快速线性卷积运算
####################

MODE=1 # 0: 使用快速算法(中间变量精简形式)
       # 1: 使用快速算法(原始形式)
       # 2: 使用快速算法(矩阵形式)
       # 3: 使用线性卷积定义
        

## (2,2)线性卷积 
#   计算{x[0],x[1]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2}
#   不考虑系数h的加法运算的话，需要3次乘法和3次加法
# code list 5-3
def conv_2_2(x,h, mode=MODE):
    if mode==0:
        ha=h[1]-h[0]
        
        ya=(x[0]-x[1])*ha  
        y0=x[0]*h[0]
        y2=x[1]*h[1]
        
        y1=y0+y2+ya

    elif mode==1:
        H0=h[0]
        H1=h[0]-h[1]
        H2=h[1]
        
        X0=x[0]
        X1=x[0]-x[1]
        X2=x[1]
        
        Y0=H0*X0
        Y1=H1*X1
        Y2=H2*X2
        
        y0=Y0
        y1=Y0-Y1+Y2
        y2=Y2

    elif mode==2:
        A=np.array([[1, 0, 0],
                    [1,-1, 1],
                    [0, 0, 1]])
        B=C=np.array([[1, 0],
                      [1,-1],
                      [0, 1]])
        y0,y1,y2=A.dot(B.dot(x.reshape(2,1))*C.dot(h.reshape(2,1))).ravel()
    elif mode==3:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[1]*h[1]
    return y0,y1,y2

## (2,2)线性卷积的另一种等效算法 
#   计算{x[0],x[1]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2}
#   不考虑系数h的加法运算的话，需要3次乘法和3次加法
def conv_2_2a(x,h, mode=MODE):
    if mode==0:
        y0=x[0]*h[0]
        y2=x[1]*h[1]
        y1=(x[0]+x[1])*(h[1]+h[0])-y0-y2

    elif mode==1:
        H0=h[0]
        H1=h[0]+h[1]
        H2=h[1]
        
        X0=x[0]
        X1=x[0]+x[1]
        X2=x[1]
        
        Y0=H0*X0
        Y1=H1*X1
        Y2=H2*X2
        
        y0=Y0
        y1=-Y0+Y1-Y2
        y2=Y2

    elif mode==2:
        A=np.array([[ 1, 0, 0],
                    [-1, 1,-1],
                    [ 0, 0, 1]])
        B=C=np.array([[1, 0],
                      [1, 1],
                      [0, 1]])
        y0,y1,y2=A.dot(B.dot(x.reshape(2,1))*C.dot(h.reshape(2,1))).ravel()
    elif mode==3:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[1]*h[1]
    return y0,y1,y2


## (3,2)线性卷积 
#   计算{x[0],x[1],x[2]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2,y3}
#   不考虑系数预计算的话，运算量是4次乘法和7次加法
# code list 5-4
def conv_3_2(x,h, mode=MODE):
    if mode==0:
        ha=(h[0]+h[1])/2
        hb=(h[0]-h[1])/2
        
        xc=x[0]+x[2]
        xa=xc+x[1]
        xb=xc-x[1]
        
        m1=ha*xa
        m2=hb*xb
        
        y0=h[0]*x[0]
        y3=h[1]*x[2]
        
        y1=m1-m2-y3
        y2=m1+m2-y0
    elif mode==1:
        H0= h[0]
        H1=(h[0]+h[1])/2
        H2=(h[0]-h[1])/2
        H3= h[1]
        
        X0= x[0]
        X1=(x[0]+x[2])+x[1]
        X2=(x[0]+x[2])-x[1]
        X3= x[2]
        
        Y0=H0*X0
        Y1=H1*X1
        Y2=X2*H2
        Y3=H3*X3
        
        y0=Y0
        y1=Y1-Y2-Y3
        y2=Y1+Y2-Y0
        y3=Y3
    elif mode==2:
        A=np.array([[ 1, 0, 0, 0],
                    [ 0, 1,-1,-1],
                    [-1, 1, 1, 0],
                    [ 0, 0, 0, 1]])
        B=np.array([[1, 0, 0],
                    [1, 1, 1],
                    [1,-1, 1],
                    [0, 0, 1]])
        C=np.array([[  1,   0],
                    [0.5, 0.5],
                    [0.5,-0.5],
                    [  0,   1]])
        y0,y1,y2,y3=A.dot(B.dot(x.reshape(3,1))*C.dot(h.reshape(2,1))).ravel()
    elif mode==3:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]
        y3=          x[2]*h[1]
    return y0,y1,y2,y3


## (3,3)线性卷积 
#   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
#   输出{y0,y1,y2,y3,y4}
#   不考虑系数预计算以及整数乘法的话，运算量是5次乘法和20次加法
# code list 5-5
def conv_3_3(x,h, mode=MODE):
    if mode==0:
        # 当需要多次卷积时，g0-g3的值可以预先保存下来，不需要重复计算
        g0= h[0]/2          
        g1=(h[0]+h[1]+h[2])/2
        g2=(h[0]-h[1]+h[2])/6
        g3=(h[0]+2*h[1]+4*h[2])/6
        
        d0=x[1]+x[2]
        d1=x[2]-x[1]
        d2=x[0]+d0
        d3=x[0]+d1
        d4=d0+d0+d1+d2
        
        m0=g0*x[0]
        m1=g1*d2
        m2=g2*d3
        m3=g3*d4
        
        y4=h[2]*x[2]
        
        u0=m1+m1
        u1=m2+m2
        u2=y4+y4-m0-m3
        u3=m1+m2
        
        y0= m0+m0
        y1= u0-u1+u2
        y2= u1+u3-y0-y4
        y3=-u2-u3
    elif mode==1:
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
        Y4=X4*H4
        
        y0=2*Y0
        y1=-Y0+2*Y1-2*Y2-Y3+2*Y4
        y2=-2*Y0+Y1+3*Y2-Y4
        y3=Y0-Y1-Y2+Y3-2*Y4
        y4=Y4
    elif mode==2:
        A=np.array([[ 2, 0, 0, 0, 0],
                    [-1, 2,-2,-1, 2],
                    [-2, 1, 3, 0,-1],
                    [ 1,-1,-1, 1,-2],
                    [ 0, 0, 0, 0, 1]])
        B=np.array([[1, 0, 0],
                    [1, 1, 1],
                    [1,-1, 1],
                    [1, 2, 4],
                    [0, 0, 1]])
        C=np.array([[0.5,   0,  0],
                    [0.5, 0.5,0.5],
                    [1/6,-1/6,1/6],
                    [1/6, 1/3,2/3],
                    [  0,   0,  1]])
        y0,y1,y2,y3,y4=A.dot(B.dot(x.reshape(3,1))*C.dot(h.reshape(3,1))).ravel()
    elif mode==3:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]+x[0]*h[2]
        y3=          x[2]*h[1]+x[1]*h[2]
        y4=                    x[2]*h[2]
    return y0,y1,y2,y3,y4


## (3,3)线性卷积（增加1次乘法换取加法次数的减少）
#   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
#   输出{y0,y1,y2,y3,y4}
#   不考虑系数预计算以及整数乘法的话，运算量是6次乘法和10次加法
# code list 5-6
def conv_3_3a(x,h):
    # 当需要多次卷积时，g0-g2的值可以预先保存下来，不需要重复计算
    g0=h[0]+h[1]
    g1=h[0]+h[2]
    g2=h[1]+h[2]
    
    d0=x[0]+x[1]
    d1=x[0]+x[2]
    d2=x[1]+x[2]
    
    m0=h[1]*x[1]
    m1=g0*d0
    m2=g1*d1
    m3=g2*d2
    
    y0=h[0]*x[0]
    y4=h[2]*x[2]
    
    y1=m1-m0-y0
    y2=m2+m0-y0-y4
    y3=m3-m0-y4
    
    return y0,y1,y2,y3,y4


## (3,3)线性卷积 
#   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
#   输出{y0,y1,y2,y3,y4}
# 不考虑系数预计算以及整数乘法的话，运算量是5次乘法和16次加法(*2和*4不计入加法)
def conv_3_3b(x,h):
    # 当需要多次卷积时，g0-g4的值可以预先保存下来，不需要重复计算
    g0= h[0]/2          
    g1=(h[0]+h[1]+h[2])/2
    g2=(h[0]-h[1]+h[2])/6
    g3=(h[0]+2*h[1]+4*h[2])/6
    g4= h[2]
   
    d0=x[0]
    da=x[0]+x[2]
    d1=da+x[1]
    d2=da-x[1]
    d3=x[0]+2*x[1]+4*x[2]
    d4=x[2]
    
    s0=g0*d0
    s1=g1*d1
    s2=g2*d2
    s3=g3*d3
    s4=g4*d4
    
    m0= s0*2
    m1=-s0  +s1*2-s2*2-s3+s4*2
    m2=-s0*2+s1  +s2*3   -s4
    m3= s0  -s1  -s2  +s3-s4*2
    m4=                   s4
    
    return m0,m1,m2,m3,m4


## (4,4)线性卷积，通过嵌入(2,2)线性卷积得到
#   计算{x[0],x[1],x[2],x[3]}和{h[0],h[1],h[2],h[3]}的线性卷积
#   输出{y0,y1,y2,y3,y4,y5,y6}
#   不考虑系数预计算以及整数乘法的话，运算量是9次乘法和18次加法
# code list 5-9
def conv_4_4(x,h, mode=MODE):
    if mode==0: # 使用conv_2_2对应的算法嵌套，中间变量经过了精简
        a01,a10,a12,a21 = x[0]+x[1], x[0]+x[2], x[1]+x[3], x[2]+x[3]       
        a11 = a10+a12
        
        # 应用时，如果h固定不变，可以把b预先计算后保存
        b01,b10,b12,b21 = h[0]+h[1], h[0]+h[2], h[1]+h[3], h[2]+h[3] 
        b11 = b10+b12
        
        y0 ,m01,m02 = x[0]*h[0], a01*b01, x[1]*h[1]
        m20,m21,y6  = x[2]*h[2], a21*b21, x[3]*h[3]
        m10,m11,m12 = a10*b10, a11*b11, a12*b12
        
        u0 = m02-m20
        u1 = m11-m10-m12
        
        y1 = m01-m02-y0 
        y2 = m10-y0+u0     
        y5 = m21-m20-y6 
        y3 = u1-y1-y5
        y4 = m12-y6-u0
    elif mode==1:   # 使用conv_2_2对应的算法嵌套
        a00=x[0]
        a01=x[0]+x[1]
        a02=x[1]
        
        b00=h[0]
        b01=h[0]+h[1]
        b02=h[1]
        
        m00=a00*b00
        m01=a01*b01
        m02=a02*b02
        
        y00=m00
        y01=m01-m00-m02
        y02=m02
        
        a10=x[0]+x[2]
        a12=x[1]+x[3]
        a11=a10+a12
        
        b10=h[0]+h[2]
        b12=h[1]+h[3]
        b11=b10+b12

        m10=a10*b10
        m11=a11*b11
        m12=a12*b12

        y10=m10
        y11=m11-m10-m12
        y12=m12

        a20=x[2]
        a21=x[2]+x[3]
        a22=x[3]
        
        b20=h[2]
        b21=h[2]+h[3]
        b22=h[3]
        
        m20=a20*b20
        m21=a21*b21
        m22=a22*b22

        y20=m20
        y21=m21-m20-m22
        y22=m22

        y0=y00
        y1=y01
        y2=y10-y00+(y02-y20)
        y3=y11-y01-y21
        y4=y12-y22-(y02-y20)
        y5=y21
        y6=y22
    elif mode==2:   # 使用conv_2_2a对应的算法嵌套
        a00=x[0]
        a01=x[0]-x[1]
        a02=x[1]
        
        b00=h[0]
        b01=h[0]-h[1]
        b02=h[1]
        
        m00=a00*b00
        m01=a01*b01
        m02=a02*b02
        
        y00=m00
        y01=m00-m01+m02
        y02=m02
        
        a10=x[0]-x[2]
        a12=x[1]-x[3]
        a11=a10-a12
        
        b10=h[0]-h[2]
        b12=h[1]-h[3]
        b11=b10-b12

        m10=a10*b10
        m11=a11*b11
        m12=a12*b12

        y10=m10
        y11=m10-m11+m12
        y12=m12

        a20=x[2]
        a21=x[2]-x[3]
        a22=x[3]
        
        b20=h[2]
        b21=h[2]-h[3]
        b22=h[3]
        
        m20=a20*b20
        m21=a21*b21
        m22=a22*b22

        y20=m20
        y21=m20-m21+m22
        y22=m22

        y0=y00
        y1=y01
        y2=y02+y00-y10+y20
        y3=y01-y11+y21
        y4=y02-y12+y22+y20
        y5=y21
        y6=y22
    elif mode==3:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]+x[0]*h[2]
        y3=x[3]*h[0]+x[2]*h[1]+x[1]*h[2]+x[0]*h[3]
        y4=          x[3]*h[1]+x[2]*h[2]+x[1]*h[3]
        y5=                    x[3]*h[2]+x[2]*h[3]
        y6=                              x[3]*h[3]
    return y0,y1,y2,y3,y4,y5,y6


## (4,2)线性卷积，通过拼接(2,2)线性卷积得到 
#   计算{x[0],x[1],x[2],x[3]}和{h[0],h[1]}的线性卷积
#   输出{y0,y1,y2,y3,y4}
#   不考虑系数预计算的话，运算量是6次乘法和7次加法
# code list 5-7
def conv_4_2(x,h, mode=MODE):
    if mode==0:
        y0 ,y1,y2a=conv_2_2(x[:2],h)
        y2b,y3,y4 =conv_2_2(x[2:],h)
        y2=y2a+y2b
    elif mode in [1,2,3]:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]
        y3=x[3]*h[0]+x[2]*h[1]
        y4=          x[3]*h[1]        
    return y0,y1,y2,y3,y4


## 通过拼接(3,2)线性卷积得到(6,2)线性卷积 
# 不考虑系数预计算的话，运算量是8次乘法和14次加法
def conv_6_2(x,h, mode=MODE):
    if mode==0:
        y0 ,y1,y2,y3a=conv_3_2(x[:3],h)
        y3b,y4,y5,y6 =conv_3_2(x[3:],h)
        y3=y3a+y3b
    elif mode in [1,2,3]:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]
        y3=x[3]*h[0]+x[2]*h[1]
        y4=x[4]*h[0]+x[3]*h[1] 
        y5=x[5]*h[0]+x[4]*h[1] 
        y6=          x[5]*h[1] 
    
    return y0,y1,y2,y3,y4,y5,y6


## (5,3)线性卷积，通过拼接(3,2)和(3,3)线性卷积得到 
#   计算{x[0],x[1],x[2],x[3],x[4]}和{h[0],h[1],h[2]}的线性卷积
#   输出{y0,y1,y2,y3,y4,y5,y6}
#   不考虑系数预计算的话，运算量是9次乘法和29次加法
#  code list 5-8
def conv_5_3(x,h, mode=MODE):
    if mode==0:
        y10,y11,y12,y13=conv_3_2(h,x[:2])
        y20,y21,y22,y23,y24=conv_3_3(x[2:],h)
        
        y0,y1   = y10,y11
        y2,y3   = y12+y20,y13+y21
        y4,y5,y6= y22,y23,y24
    elif mode in [1,2,3]:
        y0=x[0]*h[0]
        y1=x[1]*h[0]+x[0]*h[1]
        y2=x[2]*h[0]+x[1]*h[1]+x[0]*h[2]
        y3=x[3]*h[0]+x[2]*h[1]+x[1]*h[2]
        y4=x[4]*h[0]+x[3]*h[1]+x[2]*h[2] 
        y5=          x[4]*h[1]+x[3]*h[2] 
        y6=                    x[4]*h[2]
    return y0,y1,y2,y3,y4,y5,y6


####################
# 单元测试
####################

## 验证几种卷积模式的结果一致性
print('[INF] ==== verify fast conv (2,2)')
x=np.random.randint(-10,10,2).astype(float)
h=np.random.randint(-10,10,2).astype(float)
res=[conv_2_2(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (2,2)a')
x=np.random.randint(-10,10,2).astype(float)
h=np.random.randint(-10,10,2).astype(float)
res=[conv_2_2a(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (3,2)')
x=np.random.randint(-10,10,3).astype(float)
h=np.random.randint(-10,10,2).astype(float)
res=[conv_3_2(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (3,3)')
x=np.random.randint(-10,10,3).astype(float)
h=np.random.randint(-10,10,3).astype(float)
res=[conv_3_3(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

res_a=conv_3_3a(x,h)
print('[INF] error in mode a:',np.linalg.norm(np.array(res[3])-np.array(res_a)))
res_b=conv_3_3b(x,h)
print('[INF] error in mode b:',np.linalg.norm(np.array(res[3])-np.array(res_b)))

print('[INF] ==== verify fast conv (4,2)')
x=np.random.randint(-10,10,4).astype(float)
h=np.random.randint(-10,10,2).astype(float)
res=[conv_4_2(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (4,4)')
x=np.random.randint(-10,10,4).astype(float)
h=np.random.randint(-10,10,4).astype(float)
res=[conv_4_4(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (6,2)')
x=np.random.randint(-10,10,6).astype(float)
h=np.random.randint(-10,10,2).astype(float)
res=[conv_6_2(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

print('[INF] ==== verify fast conv (5,3)')
x=np.random.randint(-10,10,5).astype(float)
h=np.random.randint(-10,10,3).astype(float)
res=[conv_5_3(x,h, mode) for mode in range(4)]  # 几种模式下计算卷积
print('[INF] reference result:',res[3])
for n in range(4): print('[INF] error in mode %d:'%n,np.linalg.norm(np.array(res[3])-np.array(res[n])))

## 基于符号多项式运算验证卷积程序
if True:
    import sympy as sp
    def verify_conv(func,x,h):
        p=sp.symbols('p')   # 多项式变量

        # 使用多项式计算卷积
        xp=sum([x[i]*p**i for i in range(len(x))])
        hp=sum([h[i]*p**i for i in range(len(h))])
        yp=xp*hp
        yp=sp.collect(sp.expand(yp),p)
        
        # 使用被测函数计算卷积
        y=func(x,h)

        # 检验两种方法的差别
        yp_calc=sum([y[i]*p**i for i in range(len(y))])
        e=sp.simplify(yp-yp_calc)
        if e != 0:
            print(' **** Error!',e)
        else:
            print(' Pass!')

    print('[INF] ==== verify conv_2_2()',end='')
    verify_conv(conv_2_2, x=sp.symbols('x:2'), h=sp.symbols('h:2'))

    print('[INF] ==== verify conv_2_2a()',end='')
    verify_conv(conv_2_2a, x=sp.symbols('x:2'), h=sp.symbols('h:2'))

    print('[INF] ==== verify conv_3_2()',end='')
    verify_conv(conv_3_2, x=sp.symbols('x:3'), h=sp.symbols('h:2'))

    print('[INF] ==== verify conv_3_3()',end='')
    verify_conv(conv_3_3, x=sp.symbols('x:3'), h=sp.symbols('h:3'))

    print('[INF] ==== verify conv_3_3a()',end='')
    verify_conv(conv_3_3a, x=sp.symbols('x:3'), h=sp.symbols('h:3'))

    print('[INF] ==== verify conv_3_3b()',end='')
    verify_conv(conv_3_3b, x=sp.symbols('x:3'), h=sp.symbols('h:3'))

    print('[INF] ==== verify conv_4_4()',end='')
    verify_conv(conv_4_4, x=sp.symbols('x:4'), h=sp.symbols('h:4'))

    print('[INF] ==== verify conv_4_2()',end='')
    verify_conv(conv_4_2, x=sp.symbols('x:4'), h=sp.symbols('h:2'))

    print('[INF] ==== verify conv_6_2()',end='')
    verify_conv(conv_6_2, x=sp.symbols('x:6'), h=sp.symbols('h:2'))

    print('[INF] ==== verify conv_5_3()',end='')
    verify_conv(conv_5_3, x=sp.symbols('x:5'), h=sp.symbols('h:3'))
