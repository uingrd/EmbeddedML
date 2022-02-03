#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# Strassen快速矩阵乘法
#
# smul  代码清单6-1
# cwmul 式(6-10)
####################

## Strassen矩阵乘法
# 对2x2矩阵乘法需要7次乘法和18次加减法
# 输入：
#   X=[A B]  Y=[E F]
#     [C D]    [G H]
# 输出：  
#    [Z00 Z01]=X*Y
#    [Z10 Z11]
def smul(X,Y):
    if X.size==1:   # 标量元素
        return float(X)*float(Y)
    
    # 可进一步拆分为子矩阵
    N=X.shape[0]//2 # 子矩阵尺寸
    A,B,C,D=X[0:N,0:N],X[0:N,N:],X[N:,0:N],X[N:,N:]
    E,F,G,H=Y[0:N,0:N],Y[0:N,N:],Y[N:,0:N],Y[N:,N:]
    
    # 分块计算
    P1=smul(A,F-H)
    P2=smul(A+B,H)
    P3=smul(C+D,E)
    P4=smul(D,G-E)
    P5=smul(A+D,E+H)
    P6=smul(B-D,G+H)
    P7=smul(A-C,E+F)
    
    Z00=P5+P4-P2+P6
    Z01=P1+P2
    Z10=P3+P4
    Z11=P1+P5-P3-P7
    
    # 子矩阵合并得到结果
    return np.vstack((np.hstack((Z00,Z01)),
                      np.hstack((Z10,Z11))))


# Coppersmith–Winograd算法
# 对2x2矩阵乘法需要7次乘法和15次加减法
def cwmul(X,Y):
    if X.size==1:   # 标量元素
        return float(X)*float(Y)
    
    # 可进一步拆分为子矩阵
    N=X.shape[0]//2 # 子矩阵尺寸
    A,B,C,D=X[0:N,0:N],X[0:N,N:],X[N:,0:N],X[N:,N:]
    E,F,G,H=Y[0:N,0:N],Y[0:N,N:],Y[N:,0:N],Y[N:,N:]
    
    # 分块计算
    S1=C+D  
    S2=S1-A 
    S3=A-C  
    S4=B-S2 
    
    T1=F-E
    T2=H-T1
    T3=H-F
    T4=T2-G
    
    M1=cwmul(A,E)
    M2=cwmul(B,G)
    M3=cwmul(S4,H)
    M4=cwmul(D,T4)
    M5=cwmul(S1,T1)
    M6=cwmul(S2,T2)
    M7=cwmul(S3,T3)
    
    U1=M1+M2
    U2=M1+M6
    U3=U2+M7
    U4=U2+M5
    U5=U4+M3
    U6=U3-M4
    U7=U3+M5
    
    # 子矩阵合并得到结果
    return np.vstack((np.hstack((U1,U5)),
                      np.hstack((U6,U7))))

def extend_row(m): return np.vstack((m,np.zeros((1,m.shape[1]))))
def extend_col(m): return np.hstack((m,np.zeros((m.shape[0],1))))
## Strassen矩阵乘法
# 可应用于非2^n的方阵乘法
def smul_nsqr(X,Y):
    if X.size==1: return float(X)*Y
    if Y.size==1: return X*float(Y)
    
    Nr,Nc=X.shape # 子矩阵尺寸
    Mr,Mc=Y.shape # 子矩阵尺寸
    
    # 处理行列奇数问题
    if Nr==1 or Nc==1 or Mc==1: return X.dot(Y)
    if Nr%2: return smul_nsqr(extend_row(X),Y)[:-1,:]
    if Nc%2: return smul_nsqr(extend_col(X),extend_row(Y))
    if Mc%2: return smul_nsqr(X,extend_col(Y))[:,:-1]    
    
    Nr0,Nc0=Nr//2,Nc//2
    Mr0,Mc0=Mr//2,Mc//2

    # 拆分为子矩阵处理
    A,B,C,D=X[0:Nr0,0:Nc0],X[0:Nr0,Nc0:],X[Nr0:,0:Nc0],X[Nr0:,Nc0:]
    E,F,G,H=Y[0:Mr0,0:Mc0],Y[0:Mr0,Mc0:],Y[Mr0:,0:Mc0],Y[Mr0:,Mc0:]
    
    # 分块计算
    P1=smul_nsqr(A,F-H)
    P2=smul_nsqr(A+B,H)
    P3=smul_nsqr(C+D,E)
    P4=smul_nsqr(D,G-E)
    P5=smul_nsqr(A+D,E+H)
    P6=smul_nsqr(B-D,G+H)
    P7=smul_nsqr(A-C,E+F)
    
    Z00=P5+P4-P2+P6
    Z01=P1+P2
    Z10=P3+P4
    Z11=P1+P5-P3-P7
    
    # 子矩阵合并得到结果
    return np.vstack((np.hstack((Z00,Z01)),
                      np.hstack((Z10,Z11))))


# Coppersmith–Winograd算法
# 可应用于非2^n的方阵乘法
def cwmul_nsqr(X,Y):
    if X.size==1: return float(X)*Y
    if Y.size==1: return X*float(Y)
    
    Nr,Nc=X.shape # 子矩阵尺寸
    Mr,Mc=Y.shape # 子矩阵尺寸
    
    # 处理行列奇数问题
    if Nr==1 or Nc==1 or Mc==1: return X.dot(Y)
    if Nr%2: return cwmul_nsqr(extend_row(X),Y)[:-1,:]
    if Nc%2: return cwmul_nsqr(extend_col(X),extend_row(Y))
    if Mc%2: return cwmul_nsqr(X,extend_col(Y))[:,:-1]   

    Nr0,Nc0=Nr//2,Nc//2
    Mr0,Mc0=Mr//2,Mc//2

    # 拆分为子矩阵处理
    A,B,C,D=X[0:Nr0,0:Nc0],X[0:Nr0,Nc0:],X[Nr0:,0:Nc0],X[Nr0:,Nc0:]
    E,F,G,H=Y[0:Mr0,0:Mc0],Y[0:Mr0,Mc0:],Y[Mr0:,0:Mc0],Y[Mr0:,Mc0:]
    
    # 分块计算
    S1=C+D  
    S2=S1-A 
    S3=A-C  
    S4=B-S2 
    
    T1=F-E
    T2=H-T1
    T3=H-F
    T4=T2-G
    
    M1=cwmul_nsqr(A,E)
    M2=cwmul_nsqr(B,G)
    M3=cwmul_nsqr(S4,H)
    M4=cwmul_nsqr(D,T4)
    M5=cwmul_nsqr(S1,T1)
    M6=cwmul_nsqr(S2,T2)
    M7=cwmul_nsqr(S3,T3)
    
    U1=M1+M2
    U2=M1+M6
    U3=U2+M7
    U4=U2+M5
    U5=U4+M3
    U6=U3-M4
    U7=U3+M5
    
    # 子矩阵合并得到结果
    return np.vstack((np.hstack((U1,U5)),
                      np.hstack((U6,U7))))


###########
# 单元测试
###########
if __name__=='__main__':
    np.random.seed(4321)
    
    print('[INF] test on square matrices')
    for n in range(7):
        N=2**n
        print('[INF] ---- test:%d ----'%n)
        print('[INF] size: %d'%N)
        
        x=np.random.randint(-100,100,(N,N))
        y=np.random.randint(-100,100,(N,N))
        z0=x.dot(y)
        
        print('[INF] testing smul...  ',end='')
        z1=smul(x,y)
        print('error:',np.max(np.abs(z0-z1).ravel()))

        print('[INF] testing cwmul... ',end='')
        z2=cwmul(x,y)
        print('error:',np.max(np.abs(z0-z2).ravel()))

    print('[INF] test on non-square matrices')
    for n in range(10):
        print('[INF] ---- test:%d ----'%n)
        Nr,Nc,Mc=np.random.randint(1,100,3)
        print('[INF] size: %d,%d,%d'%(Nr,Nc,Mc))
        x=np.random.randint(-100,100,(Nr,Nc))
        y=np.random.randint(-100,100,(Nc,Mc))
        z0=x.dot(y)
        
        print('[INF] testing smul_nsqr...  ',end='')
        z1=smul_nsqr(x,y)
        print('error:',np.max(np.abs(z0-z1).ravel()))

        print('[INF] testing cwmul_nsqr... ',end='')
        z2=cwmul_nsqr(x,y)
        print('error:',np.max(np.abs(z0-z2).ravel()))
