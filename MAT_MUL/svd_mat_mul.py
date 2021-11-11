#!/usr/bin/python3
# coding=utf-8

########################
# 使用SVD实现矩阵低秩近似和近似乘法
#
# svd_mat_mul: 代码清单6-3
#
########################

import numpy as np

def random_mat(row=2,col=2,low=-100,high=100):
    return np.random.randint(low=low,high=high,size=(row,col)).astype(float)

def gen_random_low_rank_mat(N,L,R,Pn=0):
    return np.dot(random_mat(N,R),random_mat(R,L))+Pn*np.random.randn(N,L)

# 基于矩阵的SVD低秩表示计算近似矩阵乘法
def svd_mat_mul(X,UR=None,VR=None,A=None,R=None):
    if UR is None or VR is None:        
        # 从A的SVD分解得到UR、VR
        U,S,Vh=np.linalg.svd(A)
        UR=[U[:,r]*S[r] for r in range(R)]
        VR=[Vh.T[:,r] for r in range(R)]

        # 验证A的分解式
        Ar=np.zeros_like(A)
        for r in range(R):
            Ar+=UR[r].reshape(-1,1).dot(VR[r].reshape(1,-1))
        print('[INF] rank R approx. error:', np.max(np.abs(A-Ar))/np.mean(np.abs(A)))  # 舍入误差
        
    # 计算A和LxM矩阵X的乘积: Y=AX
    Y=np.zeros((N,M))
    for m in range(M):
        xm=X[:,m].flatten()
        for r in range(R):
            Y[:,m]+=UR[r]*np.sum(VR[r]*xm)
    return Y

if __name__=='__main__':
    np.random.seed(1234)
    
    # 生成低秩矩阵A，尺寸NxL，秩R
    N,L=30,20       # A的尺寸
    R=3             # A的秩
    A=gen_random_low_rank_mat(N,L,R,Pn=0.1)

    # 生成LxM随机矩阵X
    M=40
    X=random_mat(L,M)

    # 测试SVD分解近似矩阵计算
    Y=svd_mat_mul(X,A=A,R=R)
    print('[INF] error:',np.max(np.abs(Y-np.dot(A,X)))/np.mean(np.abs(Y)))

