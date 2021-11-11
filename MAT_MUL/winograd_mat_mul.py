#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# Winograd快速矩阵乘法
#
# mul  代码清单 6-2
#
####################

## 矩阵右端扩展一列0
def col_zero_ext(M): return np.hstack((M,np.zeros((M.shape[0],1))))

## 矩阵底端扩展一行0
def row_zero_ext(M): return np.vstack((M,np.zeros((1,M.shape[1]))))

## Winograd矩阵乘法
def mul(A,B):
    N ,L=A.shape
    L1,M=B.shape
    if L1 != L: 
        print('[ERR] size mismatch')
        return None # 尺寸错误
    
    if L%2==1:  # 通过插补0是的尺寸L改成偶数，然后再调用乘法 
        return mul(col_zero_ext(A),row_zero_ext(B))
    
    # An=[np.sum(A[n,::2]*A[n,1::2]) for n in range(N)]
    An=np.zeros(N)
    for n in range(N):
        for k in range(L//2):
            An[n]+=A[n,2*k]*A[n,2*k+1]
    
    # Bm=[np.sum(B[::2,m]*B[1::2,m]) for m in range(M)]
    Bm=np.zeros(M)
    for m in range(M):
        for k in range(L//2):
            Bm[m]+=B[2*k,m]*B[2*k+1,m]
    
    
    C=np.zeros((N,M))
    for n in range(N):
        for m in range(M):
            # C[n,m]=np.sum((A[n,::2]+B[1::2,m].T)*\
            #               (A[n,1::2]+B[::2,m].T))-An[n]-Bm[m]   
            C[n,m] = -An[n]-Bm[m]         
            for k in range(L//2):
                C[n,m]+=(A[n,k*2]+B[k*2+1,m])*(A[n,k*2+1]+B[k*2,m])

    return C
    
## 生成随机矩阵
def random_mat(row=2,col=2,low=-100,high=100):
    return np.random.randint(low=low,high=high,size=(row,col)).astype(np.float32)

##########
# 单元测试
##########

if __name__ == '__main__':
    np.random.seed(1234)
    
    # 生成测试矩阵
    N,M,L=53,31,81  # 矩阵尺寸
    A=random_mat(N,L)
    B=random_mat(L,M)

    # 矩阵相乘
    C=mul(A,B)  
    # 打印误差
    print('[INF] error:',np.max(np.abs(C-np.dot(A,B))))   

    # 显示MNL和运算次数减少量(图6-7)
    import pylab as plt
    N=np.arange(3,50)
    M=np.arange(3,50)
    r=(N+M+M*N)/(2*M*N)
    plt.plot(N,r,'.-k')
    plt.grid(True)
    plt.xlabel('N')
    plt.ylabel('ratio of # multiplications')
    plt.title('ratio of # multiplications')
    plt.show()
