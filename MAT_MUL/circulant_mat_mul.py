#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 通过DFT计算循环矩阵和向量乘积
#
# circulant_mat_mul_FFT 代码清单6-4
# circulant_mat_mul_DFT 式(6-26)
# DFT_mat               式(6-25)
#
####################

# 共厄转置
def getH(mat): return np.conj(mat.T)

# 得到DFT矩阵
def DFT_mat(N):
    m,n = np.meshgrid(np.arange(N),np.arange(N))
    w0 = np.exp(-2j*np.pi/N)
    W  = np.power(w0,m*n)/np.sqrt(N)
    return W

# 基于numpy的FFT的循环矩阵快速乘法
def circulant_mat_mul_FFT(a,x):    
    fa=np.fft.fft(a)
    fx=np.fft.fft(x)
    fy=fa*fx
    y=np.fft.ifft(fy)
    return np.real(y)

# 基于DFT矩阵的循环矩阵快速乘法
def circulant_mat_mul_DFT(a,x,W=None):
    N=len(a)
    if W is None: W=DFT_mat(N)
    
    L=np.diag(np.fft.fft(a))        
    Wx=W.dot(x)
    y=np.conj(W).dot(L).dot(Wx)
    return np.real(y)
    
####################
# 单元测试
####################
if __name__ == '__main__':
    np.random.seed(4567)
    from scipy.linalg import circulant

    # 生成随机循环阵A和测试数据
    N=5 # 矩阵尺寸
    x=np.random.randint(-10,10,N).astype(float) 
    a=np.random.randint(-10,10,N).astype(float) # 矩阵第一行
    A=circulant(a)  # 构造整个矩阵
    print('[INF] A:\n',A)

    ####################
    # 式(6-24)的验证
    W=DFT_mat(N)    # 构造DFT矩阵W
    # 验证W正交性 
    print('[INF] verify W*W.H==eye(N), error:',np.max(np.abs(np.dot(W,getH(W))-np.eye(N)).ravel()))

    # 循环阵A第一行原始的傅立叶变换
    fa=W.dot(a.reshape(N,1))*np.sqrt(N)
    D=np.diag(fa.ravel())

    # A矩阵的傅立叶矩阵分解形式验证
    A1=getH(W).dot(D).dot(W)
    print('[INF] verify A==W.H*D*W, error:',np.max(np.abs((A1-A).ravel())))

    ####################
    # 基于DFT的循环矩阵和向量乘法验证
    y=circulant_mat_mul_FFT(a,x)
    print('[INF] circulant_mat_mul_FFT')
    print('[INF] verify y==A.dot(x):',np.max(np.abs(y-A.dot(x)).ravel()))

    print('[INF] circulant_mat_mul_DFT')
    y=circulant_mat_mul_DFT(a,x)
    print('[INF] verify y==A.dot(x):',np.max(np.abs(y-A.dot(x)).ravel()))



