#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.signal import filtfilt,lfilter

####################
# 近似1D卷积例程
####################

np.random.seed(1234)

####################
## 矩形卷积
####################

## 矩形卷积，仅仅输出有效部分
# 代码清单5-23
def const_ker_conv_valid(x,K,c=1.0):
    xcum=np.cumsum(x)*c
    return np.hstack((xcum[K-1],xcum[K:]-xcum[:N-K]))

# 矩形卷积，仅仅输出有效部分，使用显式循环算法
def const_ker_conv_valid_loop(x,K,c=1.0):
    y=np.zeros(N-K+1)
    y[0]=np.sum(x[:K])*c
    for n in range(1,N-K+1):
        y[n]=y[n-1]+(x[n+K-1]-x[n-1])*c
    return y
        
# 线性矩形卷积
def const_ker_conv(x,K,c=1.0):    
    N=np.size(x)
    y0=np.cumsum(x[:K-1])*c
    y1=np.cumsum(x[N-1:N-K:-1])[::-1]*c
    return np.hstack((y0,const_ker_conv_valid(x,K,c),y1))

## 线性矩形卷积，使用显式循环算法
# 代码清单5-24
def const_ker_conv_loop(x,K,c=1.0):
    N=np.size(x)
    y=np.zeros(N+K-1)
   
    y[:K]=np.cumsum(x[:K])*c

    for n in range(K,N):
        y[n]=y[n-1]+(x[n]-x[n-K])*c
    
    y[N+K-2:N-1:-1]=np.cumsum(x[N-1:N-K:-1])*c
    return y 


## 测试一维矩形卷积
if True:
    K=5
    N=100
    c=0.7

    h=np.ones(K)*c
    x=np.random.randint(-10,10,N).astype(float)
    y_ref=np.convolve(x, h, mode='valid') # 参考答案

    y=const_ker_conv_valid(x,K,c)
    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('const_ker_conv_valid')
    plt.show()

    y=const_ker_conv_valid_loop(x,K,c)
    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('const_ker_conv_valid_loop')
    plt.show()

    y_ref=np.convolve(x, h, mode='full') # 参考答案

    y =const_ker_conv(x,K,c)
    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('const_ker_conv')
    plt.show()
    
    y =const_ker_conv_loop(x,K,c)
    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('const_ker_conv_loop')
    plt.show()

    
## 测试用矩形卷积合成卷积
# 代码清单5-25
if True:
    N=100
    x=np.random.randint(-10,10,N).astype(float)
    #x=np.zeros(N)
    #x[N//2]=1
    
    h=np.array([1,1,2,4,4,2,2,2,2])
    #           1 1 1 1 1 1 1 1 1   K1,c1=9, 1
    #               1 1 1 1 1 1 1   K2,c2=7, 1
    #                 2 2 2 2 2 2   K3,c3=6, 2
    #                    -2-2-2-2   K4,c4=4,-2
    K1,c1 = 9, 1 
    K2,c2 = 7, 1 
    K3,c3 = 6, 2 
    K4,c4 = 4,-2
    
    y1=const_ker_conv_valid(x,K1,c1)
    y2=const_ker_conv_valid(x,K2,c2)
    y3=const_ker_conv_valid(x,K3,c3)
    y4=const_ker_conv_valid(x,K4,c4)

    N1=np.size(y1)
    N2=np.size(y2)
    N3=np.size(y3)
    N4=np.size(y4)

    K=len(h)
    y=np.zeros(N-K+1)
    for n in range(N-K+1):
        y[n]+=y1[n]
        y[n]+=y2[n]
        y[n]+=y3[n]
        y[n]+=y4[n]

    y_ref=np.convolve(x, h, mode='valid') # 参考答案

    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('combined const_ker_conv(1)')
    plt.show()


# 代码清单5-26
if True:
    N=100
    x=np.random.randint(-10,10,N).astype(float)
    #x=np.zeros(N)
    #x[N//2]=1
    
    h=np.array([1,1,2,4,4,2,2,2,2])
    #           1 1 1 1 1 1 1 1 1   K1,c1,d1=9, 1,0
    #               1 1 1 1 1 1 1   K2,c2,d2=7, 1,0
    #                 2 2           K3,c3,d3=6, 2,3
    K1,c1,d1 = 9, 1, 0 
    K2,c2,d2 = 7, 1, 0 
    K3,c3,d3 = 2, 2, 4 
    
    y1=const_ker_conv_valid(x,K1,c1)
    y2=const_ker_conv_valid(x,K2,c2)
    y3=const_ker_conv_valid(x,K3,c3)

    N1=np.size(y1)
    N2=np.size(y2)
    N3=np.size(y3)

    K=len(h)
    y=np.zeros(N-K+1)
    for n in range(N-K+1):
        if n>=d1: y[n-d1]+=y1[n]
        if n>=d2: y[n-d2]+=y2[n]
        if n>=d3: y[n-d3]+=y3[n]

    y_ref=np.convolve(x, h, mode='valid') # 参考答案

    plt.plot(y,'o-')
    plt.plot(y_ref,'--')
    plt.title('combined const_ker_conv(2)')
    plt.show()


# 绘制原始卷积核和近似卷积核图
if True:
    h_ori=np.array([1.0,0.9,2.1,4.2,3.8,2.2,2.0,1.8,2.0])
    h=np.array([1,1,2,4,4,2,2,2,2])
    
    plt.plot(h_ori,'o--k')
    plt.plot(np.arange(len(h))+0.5,h,color='k')#,linestyle='steps-')
    plt.ylim([-0.2,4.5])
    plt.legend(['kernel', 'approximated kernel'])
    plt.show()

####################
## 锯齿形卷积（只输出valid部分）
# h对应的冲击响应是[b,b+s,b+2*s,...b+(K-1)*s]
# 输出卷积结果需要
#   b!=0: 3N次乘法
#   b==0: 2N次乘法
# 代码清单5-28
####################
def saw_ker_conv0(x,K,s=1.0):
    N=np.size(x)
    Ks=(K-1)*s

    xcum=np.cumsum(x)           
    xsum=xcum[K-1:]-xcum[:N-K+1]        
    
    y=np.zeros(N-K+1)
    y[0]=np.sum(x[:K]*np.arange(K-1,-1,-1)*s)
    for n in range(1,N-K+1):        # 2N次乘法
        y[n]=y[n-1]+xsum[n-1]*s-Ks*x[n-1]
    return y

# 代码清单5-29
def saw_ker_conv(x,K,s=1.0,b=0):
    if b==0: 
        return saw_ker_conv0(x,K,s)
    else:
        return saw_ker_conv0(x,K,s)+const_ker_conv_valid(x,K,b)# 3N次乘法
    

if True:
    S=0.3
    B=3
    N=10
    K=3
    
    
    x=np.random.randint(-10,10,N).astype(float) # 测试随机信号
    #x=np.zeros(N).astype(float)                # 测试冲击响应
    #x[N//2]=1
    
    y=saw_ker_conv(x,K,S,B)
    
    h=np.arange(K)*S+B              
    y_ref=np.convolve(x, h, mode='valid')   # 参考答案
    
    plt.plot(y,'o-')
    plt.plot(y_ref,'.--')
    plt.title('saw_ker_conv')
    plt.show()

# 基于锯齿形卷积计算分段线性卷积，累积模式
if True:
    N=100
    x=np.random.randint(-10,10,N).astype(float)
    #x=np.zeros(N).astype(float)
    #x[N//2]=1

    h=np.array([  -1,  -2,  -3,  -1,   1,   3,   5, 4.5,   4, 3.5,   3, 2.5,   2, 1.5,   1, 0.5 ])
    # segment  |        1     |           2       |                     3                       | 
    #  b       |  -1   -1   -1|  -1   -1   -1   -1|  -1   -1   -1  -1   -1   -1   -1   -1   -1  |
    #  s1=-1   |   0   -1   -2|  -3   -4   -5   -6|  -7   -8   -9  -10  -11  -12  -13  -14  -15 |
    #  s2= 3   |             0|   3    6    9   12|  15   18   21   24   27   30   33   36   39 |
    #  s3=-2.5 |            |                    0|-2.5   -5 -7.5  -10 -12.5 -15 -17.5 -20 -22.5|     
    #          |     K1=16  |        K2=13+1    |      K3=9+1                                   |
    K1,K2,K3= 16, 14, 10
    s1,s2,s3=-1, 3,-2.5
    b1,b2,b3=-1, 0,   0

    # 分段线性卷积
    y1=saw_ker_conv(x,K1,s1,b1)
    y2=saw_ker_conv(x,K2,s2,b2)
    y3=saw_ker_conv(x,K3,s3,b3)

    # 分段合成
    N1=np.size(y1)
    N2=np.size(y2)
    N3=np.size(y3)

    K=len(h)    # 原始卷积核长度
    y=np.zeros(N-K+1)
    for n in range(N-K+1):
        y[n]+=y1[n]
        y[n]+=y2[n]
        y[n]+=y3[n]
    
    y_ref=np.convolve(x, h, mode='valid')   # 参考答案
    plt.plot(y,'o-')
    plt.plot(y_ref,'.--')
    plt.title('combined saw_ker_conv (cum mode)')
    plt.show()


# 基于锯齿形卷积计算分段线性卷积，分段模式（乘法运算量比累积模式多0.5倍）
if True:
    N=100
    x=np.random.randint(-10,10,N).astype(float) # 测试随机信号
    #x=np.zeros(N).astype(float)                # 测试冲击响应
    #x[N//2]=1

    # 基于“长”锯齿形卷积计算分段线性卷积
    # segment  |    1   |    2   |        3                  | 
    #  K       |    3   |    4   |        9                  |
    #  s       |   -1   |    2   |      -0.5                 |
    #  b       |-1      |-1      |4.5                        |
    h=np.array([-1,-2,-3,-1,1,3,5,4.5,4,3.5,3,2.5,2,1.5,1,0.5])
    
    K1,K2,K3= 3, 4, 9
    s1,s2,s3=-1, 2,-0.5
    b1,b2,b3=-1,-1, 4.5

    # 分段线性卷积
    y1=saw_ker_conv(x,K1,s1,b1)
    y2=saw_ker_conv(x,K2,s2,b2)
    y3=saw_ker_conv(x,K3,s3,b3)

    # 分段合成
    N1=np.size(y1)
    N2=np.size(y2)
    N3=np.size(y3)

    K=len(h)
    y=np.zeros(N-K+1)
    for n in range(N-K+1):
        y[n]+=y1[n+K2+K3]+\
              y2[n+   K3]+\
              y3[n      ]
    
    y_ref=np.convolve(x, h, mode='valid')   # 参考答案
    plt.plot(y,'o-')
    plt.plot(y_ref,'.--')
    plt.title('combined saw_ker_conv (seg mode)')
    plt.show()


# 绘制原始卷积核和近似卷积核图
# 代码清单5-30
if True:
    h_ori=np.array([  -1.2,  -1.7,  -2.7,  -1.4,   0.8,   3.2,   5.2, 4.3,   3.6, 3.4,   3.2, 2.7,   2.1, 1.7,   1.1, 0.3 ])
    h=np.array([  -1,  -2,  -3,  -1,   1,   3,   5, 4.5,   4, 3.5,   3, 2.5,   2, 1.5,   1, 0.5 ])
    plt.plot(h_ori,'o--k')
    plt.plot(h,color='k')
    plt.legend(['kernel', 'approximated kernel'])
    plt.show()


    h=np.array([  -1,  -2,  -3,  -1,   1,   3,   5, 4.5,   4, 3.5,   3, 2.5,   2, 1.5,   1, 0.5 ])
    # segment  |        1     |           2       |                     3                       | 
    #  b       |  -1   -1   -1|  -1   -1   -1   -1|  -1   -1   -1  -1   -1   -1   -1   -1   -1  |
    #  s1=-1   |   0   -1   -2|  -3   -4   -5   -6|  -7   -8   -9  -10  -11  -12  -13  -14  -15 |
    #  s2= 3   |             0|   3    6    9   12|  15   18   21   24   27   30   33   36   39 |
    #  s3=-2.5 |              |                  0|-2.5   -5 -7.5  -10 -12.5 -15 -17.5 -20 -22.5|     
    #          |     K1=16   -+-      K2=13+1    -+-   K3=9+1                                   |
    K1,K2,K3= 16, 14, 10
    s1,s2,s3=-1, 3,-2.5
    b1,b2,b3=-1, 0,   0
    K=len(h)
    
    plt.clf()
    plt.subplot(3,1,1); plt.stem(np.hstack(([0]*(K-K1),np.arange(K1)*s1+b1)),linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-40,40])
    plt.subplot(3,1,2); plt.stem(np.hstack(([0]*(K-K2),np.arange(K2)*s2+b2)),linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-40,40])
    plt.subplot(3,1,3); plt.stem(np.hstack(([0]*(K-K3),np.arange(K3)*s3+b3)),linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-40,40])
    plt.show()
    

# 绘制梯形卷积核
if True:
    K,b,s=7,5,4
    ker1=np.arange(K)*s
    ker2=np.ones(K)*b
    
    plt.clf()
    plt.subplot(3,1,1); plt.stem(ker1+ker2,linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-1,40])
    plt.subplot(3,1,2); plt.stem(ker1     ,linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-1,40])
    plt.subplot(3,1,3); plt.stem(ker2     ,linefmt='k',markerfmt='ok',basefmt='k');plt.ylim([-1,40])
    plt.show()
    

# 矩形卷积和自身进行多重卷积
if True:
    #N=20
    #y_ref=np.exp(-(np.arange(N)/float(N)-0.5)**2/0.026)
    #y_ref/=np.max(y_ref)
    N=8
    h_ref=np.exp(-(np.arange(N+1)/float(N)-0.5)**2/0.066)
    h_ref/=np.max(h_ref)
    # 上面N=8的情况可以直接写成：h_ref=np.exp(-(np.arange(N+1)-4)**2/4.224)
    
    h=np.array([1,1])
    h_con=h.copy()
    for k in range(1,N):
        h_con=np.convolve(h_con, h, mode='full')
    s=np.max(h_con)
    h_con=h_con/s
    plt.plot(h_con,'ok',linewidth=2)
    plt.plot(h_ref,'r')
    plt.legend(['conv. result','Gaussian'])
    plt.show()
    
    x=np.random.randint(-10,10,100)
    y=x.copy()
    for k in range(N):
        y=np.hstack((y[0],y[1:]+y[0:-1],y[-1]))
    y=y/s
    print('s:',s)
    
    y_ref=np.convolve(x, h_ref, mode='full')
    plt.plot(y,'o-k')
    plt.plot(y_ref,'--r')
    plt.legend(['repeat conv.','Gaussian conv.'])
    plt.show()
    
'''
alpha=0.99
b=[1.0-alpha]
a=[1,-alpha]
#y=lfilter(b,a,x)
y=filtfilt(b,a,x)

plt.plot(x)
plt.plot(y)
plt.show()
'''
####################
# 高斯滤波的IIR近似
####################

# 计算sos IIR滤波器滤波输出
#             b0+b1*z^(-1)+b2*z^(-2)
# H(z) = s * ------------------------
#             1+a1*z^(-1)+a2*z^(-2)
def sos_iir(param,x,calc_r=False,bidir=False):
    if bidir:
        y,*_=sos_iir(param,x[::-1],calc_r,bidir=False)
        return sos_iir(param,y[::-1],calc_r,bidir=False)

    s,b0,b1,b2,a1,a2=param
    y=lfilter([b0,b1,b2],[1,a1,a2],x)
    if not calc_r: return y*s
    
    # 计算极点到复平面原点距离
    if a2==0:
        r=abs(a1)
    else:
        d=a1**2-4*a2
        if d>0:
            p1=2.0*a2/abs(-a1+np.sqrt(d))
            p2=2.0*a2/abs(-a1-np.sqrt(d))
            r=max(p1,p2)
        else:
            r=2.0*a2/np.sqrt(a1**2+abs(d))

    return y*s,r


## 计算普通iir滤波器输出
def iir(param,x,bidir=False):
    if bidir:
        y,*_=iir(param,x[::-1],bidir=False)
        return iir(param,y[::-1],bidir=False)

    s=param[0]
    num=len(param)-1
    b=param[1:1+num//2]
    a=param[1+num//2:]
    y=lfilter(b,a,x)
    return y*s


# 计算并联有损积分器IIR滤波器滤波输出
def multi_lossy_iir(param,x,calc_r=False):
    # 提取滤波器参数
    s=param[0::2]   # 偶数序号元素，对应每个滤波器输出幅度缩放因子
    a=param[1::2]   # 奇数序号元素，对应一阶IIR滤波器的“遗忘因子”
    
    s_forward =s[0::2]
    s_backward=s[1::2]
    
    a_forward =a[0::2]
    a_backward=a[1::2]

    y=np.zeros_like(x)
    for scale,alpha in zip(s_forward,a_forward): 
        y+=scale*lfilter([1],[1.,-alpha],x)
    for scale,alpha in zip(s_backward,a_backward): 
        y+=scale*lfilter([1],[1.,-alpha],x[::-1])[::-1]
    
    # 如果不需要计算极点到原点距离的话，直接输出结果
    if not calc_r: return y
    
    # 计算计算极点到原点最大距离，输出滤波结果和计算得到的距离
    r=np.max(np.abs(a))
    return y,r


def calc_multi_lossy_iir(param,x):
    # 提取滤波器参数
    s=param[0::2]   # 偶数序号元素，对应每个滤波器输出幅度缩放因子
    a=param[1::2]   # 奇数序号元素，对应一阶IIR滤波器的“遗忘因子”
    
    s_forward =s[0::2]
    s_backward=s[1::2]
    
    a_forward =a[0::2]
    a_backward=a[1::2]

    y=np.zeros_like(x)
    for scale,alpha in zip(s_forward,a_forward):
        yf=x.copy()
        for n in range(1,len(x)): yf[n]=yf[n-1]*alpha+yf[n]
        y+=yf*scale
    for scale,alpha in zip(s_backward,a_backward): 
        yf=x[::-1].copy()
        for n in range(1,len(x)): yf[n]=yf[n-1]*alpha+yf[n]
        y+=yf[::-1]*scale
    return y
    
# 代码清单5-33
def calc_sos_iir(param,x,bidir):
    if bidir:
        y=calc_sos_iir(param,x[::-1],bidir=False)
        y=calc_sos_iir(param,y[::-1],bidir=False)
        return y

    s,b0,b1,b2,a1,a2=param
    y=np.zeros_like(x)
    for n in range(2,len(x)): 
        y[n]=b0*x[n]+b1*x[n-1]+b2*x[n-2]-a1*y[n-1]-a2*y[n-2]  
    return y*s

# 代码清单5-34（经过了扩展）
IIR_TYPE    = 'lossy int'#'sos' # 'iir', 'lossy int'

SYMM        = False      # 使用对称滤波器结果？
INIT_NOISE  = True      # 加入初值噪声
NUM_REP     = 10#50        # 重复次数
CALC_R      = True
N           = 40        # FIR滤波器长度

# 构造滤波器冲击响应
t=np.arange(N)/float(N)

if SYMM:    # 生成对称的高斯滤波器
    y_ref=np.exp(-(t-0.5)**2/0.01)
    y_ref/=np.sum(y_ref)
    if False:
        plt.plot(y_ref,'.-k')
        plt.show()
else:       # 生成对非称的滤波器
    y_ref=np.exp(-(t-0.5)**2/0.01)
    y_ref*=np.sin(t*2.0*np.pi*2.3)
    y_ref/=np.sum(y_ref)
    if False:
        plt.plot(y_ref,'.-k')
        plt.show()

# 构造单位冲击序列
x=np.zeros(N)
if SYMM:
    x[N//2]=1
else:
    x[18]=1   # 考虑因果性

def cost_func(param):
    global x
    if IIR_TYPE=='sos':
        if CALC_R:
            y,r=sos_iir(param,x,calc_r=True,bidir=SYMM)
            return np.mean((y-y_ref)**2)+max(r-1.0,0)*1000
        else:
            y=sos_iir(param,x,bidir=SYMM)
            return np.mean((y-y_ref)**2)
    elif IIR_TYPE=='lossy int':
        if CALC_R:
            y,r=multi_lossy_iir(param,x,calc_r=True)
            return np.mean((y-y_ref)**2)+max(r-1.0,0)*1000
        else:
            y=multi_lossy_iir(param,x)
            return np.mean((y-y_ref)**2)
    else:
        y=iir(param,x,bidir=SYMM)
        return np.mean((y-y_ref)**2)

if True:
    f_opt=p_opt=None
    for rep in range(NUM_REP):
        if IIR_TYPE=='sos': 
            p0=np.ones(20)*0.1
        elif IIR_TYPE=='lossy int': 
            p0=np.array([1.5,0.65,-1.2,0.55,-1.5,0.5,1.2,0.5,0,0,0,0])
            p0=np.array([ 15.27511046,0.45406408,
                         -35.24457492,0.14269444,
                         -47.54674489,0.3362914 ,
                          36.33437906,0.1669472 ,
                          34.47884458,0.26702922,
                          -3.24704866,0.32677227])
            
        # 加入初值噪声
        if INIT_NOISE: p0+=np.random.randn(len(p0))*0.1
        
        # 优化
        res = minimize(cost_func, p0, 
                    method='powell',#'nelder-mead', # 'powell',#
                    options={'xtol': 1e-8, 'disp': False})
        
        # 记录历次优化最优结果
        if f_opt is None:
            p_opt=res.x
            f_opt=res.fun    
        elif res.fun<f_opt:
            p_opt=res.x
            f_opt=res.fun
        print('[%d] f_opt:%f'%(rep,f_opt))
    
    # 计算优化结果对应的滤波器冲击响应
    r=None
    if IIR_TYPE=='sos':
        y_opt,r=sos_iir(p_opt,x,calc_r=True,bidir=SYMM)
        y_calc=calc_sos_iir(p_opt,x,bidir=SYMM)
    elif IIR_TYPE=='lossy int':
        y_opt,r=multi_lossy_iir(p_opt,x,calc_r=True)
        y_calc=calc_multi_lossy_iir(p_opt,x)
    else:
        y_opt=iir(p_opt,x,bidir=SYMM)
        y_calc=y_opt
    
    print('p_opt:',p_opt)
    print('f_opt:',f_opt)
    print('r:',r)
    
    plt.plot(y_ref,'r')#,linewidth=3)
    plt.plot(y_opt, 'g--')
    plt.plot(y_calc, 'k--')
    plt.legend(['ideal', 'approximated'])
    plt.show()
    
    # 测试对随机信号滤波
    d=np.random.randint(-10,10,200).astype(float)
    d_calc=calc_multi_lossy_iir(p_opt,d)
    d_ref =np.convolve(d, y_ref, mode='full')
    plt.plot(d_calc,'k')
    plt.plot(d_ref[18:],'r--')
    plt.show()
    

## 手动设计一阶IIR
if True:
    N=40
    x=np.zeros(N)
    x[18]=1

    t=np.arange(N)/float(N)
    y_ref=np.exp(-(t-0.5)**2/0.01)
    y_ref*=np.sin(t*2.0*np.pi*2.3)
    y_ref/=np.sum(y_ref)
    
    y=np.zeros_like(x)
    s,a=  1.5,0.65; y+=s*lfilter([1],[1.,-a],x)
    s,a= -1.5,0.5 ; y+=s*lfilter([1],[1.,-a],x)
    s,a= -1.2,0.55; y+=s*lfilter([1],[1.,-a],x[::-1])[::-1]
    s,a=  1.2,0.5 ; y+=s*lfilter([1],[1.,-a],x[::-1])[::-1]
    
    
    plt.plot(y_ref,'r')
    plt.plot(y,'k')
    #plt.plot(x,'g')
    plt.grid(True)
    plt.show()
    
    p0=np.array([1.5,0.65,-1.2,0.55,-1.5,0.5,1.2,0.5,0,0,0,0])
    y0=calc_multi_lossy_iir(p0,x)
    plt.plot(y_ref,'r')
    plt.plot(y0,'k')
    plt.grid(True)
    plt.show()

    
# 用分段SVD近似表示长卷积核
# 将卷积核h分解成P段，每段长L
# 用R个长度L的序列（基）的线性组合近似表示h的各段值
# 其中R个基存放于列表UR中
# 线性之和系数存放于VR中
# 代码清单5-31
def appr_seg_filter(h,L,P,R,x,UR=None,VR=None,out_URVR=False):
    # 计算卷积核的分段近似
    if UR is None or VR is None:            
        H=h.reshape(P,L).T      # 卷积核分成P段，每段为一个列向量构成矩阵H
        U,S,Vh=np.linalg.svd(H) # SVD分解
        UR=[(U[:,r]*S[r]).ravel() for r in range(R)]
        VR=[(Vh.T[:,r]).ravel()   for r in range(R)]
    
    # 用SVD分解得到的R个卷积核(存放于UR中)对原信号分别滤波
    # y_appr_seg=[lfilter(UR[r],[1],x) for r in range(R)]
    y_appr_seg=[]
    for r in range(R):
        y_appr_seg.append(lfilter(UR[r],[1],x))
        
    # 通通过线性组合R个卷积结果构造出P个滤波结果的近似
    y_mix=[]
    for p in range(P):
        tmp=y_appr_seg[0]*VR[0][p]
        for r in range(1,R):
            tmp+=y_appr_seg[r]*VR[r][p]
        y_mix.append(tmp)
    
    # 延迟相加合并得到最中的近似卷积
    y_comb=np.zeros(N)
    for p in range(P):
        y_comb[p*L:N]+=y_mix[p][:N-p*L]
    
    return y_comb if not out_URVR else (y_comb,UR,VR)

if True:
    L,P=8,10    # 滤波器分段长度和分段数量
    R=2         # 用R个子滤波器近似表示所有P个分段滤波器
    N=200       # 数据长度
    x=np.random.randint(-10,10,N).astype(float)
    
    K=L*P
    h=np.cos(np.linspace(0,np.pi*14.7,K))*np.sin(np.linspace(0,np.pi*2.3,K))**2
    plt.plot(h,'k')
    plt.show()
    
    # 分段滤波
    y_seg=[]
    for p in range(P):
        y_seg.append(lfilter(h[p*L:(p+1)*L],[1],x))
    
    # 合并
    y=np.zeros(N)
    for p in range(P):
        y[p*L:N]+=y_seg[p][:N-p*L]
    
    y_ref=lfilter(h,[1],x)
    plt.plot(y,'.-k')
    plt.plot(y_ref,'--k')
    plt.legend(['y','y_ref'])
    plt.show()
    
    # 利用卷积核的分段近似计算卷积
    # 代码清单5-32
    x=np.random.randint(-10,10,N).astype(float)
    y_appr,UR,VR=appr_seg_filter(h,L,P,R,x,out_URVR=True)
    
    y_ref=lfilter(h,[1],x)
    plt.plot(y_appr,'.-k')
    plt.plot(y_ref,'--k')
    plt.title('conv. with kernel approximation')
    plt.legend(['y_appr','y_ref'])
    plt.show()
    
    # 利用卷积核的分段近似计算冲击响应
    x=np.zeros(N).astype(float)
    x[N//2]=1
    y_appr=appr_seg_filter(h,L,P,R,x,UR,VR)
    y_ref=lfilter(h,[1],x)
    plt.plot(y_appr,'.-k')
    plt.plot(y_ref,'--k')
    plt.title('impulse response')
    plt.legend(['y_appr','y_ref'])
    plt.show()
    
    
