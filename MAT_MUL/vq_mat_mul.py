#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pylab as plt

####################
# 基于矢量量化的快速(近似)矩阵乘法
####################

np.random.seed(4567)

## 聚类（矢量量化）
from sklearn.cluster import KMeans
def vq(v,K):
    cs=KMeans(n_clusters=K, n_init='auto')
    cs.fit(v)
    return cs.cluster_centers_, cs.labels_ 

## 测试上述聚类算法
if False:
    # 生成测试数据
    K=3     # 中心数量
    L=4     # 向量尺寸
    N=100   # 模拟数据数量
    
    # 生成聚类中心
    vec_cent=[np.random.randn(L) for _ in range(K)]

    # 围绕聚类中心生成数据集
    idx=np.random.randint(0,K,N)
    vec=[vec_cent[k]+np.random.randn(L)*0.1 for k in idx]

    cs_cent,cs_idx=vq(vec,K)
    
    # 显示聚类效果
    color=['r','g','b']
    for k in range(K):
        # 聚类中心
        cx,cy = cs_cent[k][0],cs_cent[k][1] 
        
        # 原始数据
        x = [vec[n][0] for n in range(N) if cs_idx[n]==k]
        y = [vec[n][1] for n in range(N) if cs_idx[n]==k]
    
        plt.plot( x, y,'.'+color[k],markersize=2 )
        plt.plot(cx,cy,'x'+color[k],markersize=12)
    plt.show()

##############################
# 基于矢量量化的矩阵乘法
# 代码清单 6-7
##############################

## 生成测试数据
N=200           # 矩阵行数
L=2             # 矩阵列数
K=20            # 矢量量化码本尺寸

# 待矢量量化矩阵（生成随机矩阵）
A=np.random.randn(N,L)

# 从mat中分离出L个行向量构成集合
vec=[A[n,:].ravel() for n in range(N)]

# N个行向量聚类，得到K个聚类中心（作为这N个行向量的矢量量化中心）
vec_cent,vec_idx=vq(vec,K)

# Lx1测试向量
x=np.random.randn(L,1)

# 计算x和K个矢量量化中心的内积
x_vec_cent=[np.sum(x.ravel()*vec_cent[k]) for k in range(K)]
y_hat=np.array([x_vec_cent[vec_idx[n]] for n in range(N)])

# 精确矩阵乘法
y=np.dot(A,x).ravel()

# 计算误差
err=y-y_hat
SNR=10*np.log10(np.sum(y**2)/np.sum(err**2))

plt.plot(y_hat,'b')
plt.plot(y,'r')
plt.plot(err,'k')
plt.legend(['error','y','y_hat'])
plt.title('SNR: %.2fdB'%SNR)
plt.show()

##############################
# 基于矢量量化的分块矩阵乘法
# 代码清单 6-8
##############################

N=200   # 矩阵行数
L=2     # 子矩阵列数
K=20    # 子矩阵矢量量化码本尺寸
P=5     # 子矩阵分块数

# 待矢量量化矩阵（生成随机矩阵）
A=np.random.randn(N,L*P)

# 分段矢量量化
A_vq=[] # 存放分段矢量量化信息
for b in range(P):
    # 从矩阵A选出第b段
    A_blk=A[:,b*L:(b+1)*L]
    
    # 提取其中的各个行向量
    vec=[A_blk[n,:].ravel() for n in range(N)]
    
    # 矢量量化，结果保存在A_vq
    A_vq.append(vq(vec,K))


# Lx1测试向量
x=np.random.randn(L*P,1)

# 基于分段矢量量化计算矩阵乘法
# 需要：P*L*K次乘法
y_hat=None  # 存放结果
for b in range(P):
    # 提取第b段矩阵矢量量化的量化中心和矢量编号
    vec_cent,vec_idx=A_vq[b]
    
    # 提取x分段x_blk
    x_blk=x[b*L:(b+1)*L].ravel()
    
    # 计算x分段和矢量量化中心的内积（BLK_SZ*K次乘法）
    x_blk_vec_cent=[np.sum(x_blk*vec_cent[k]) for k in range(K)]
    
    # 构成y分段计算结果
    y_blk =[x_blk_vec_cent[vec_idx[n]] for n in range(N)]
    
    # 拼接到完整结果
    y_hat=np.array(y_blk) if b==0 else y_hat+np.array(y_blk)


# 精确结果
# 需要：P*L*N次乘法
y=np.dot(A,x).ravel()

# 计算误差
err=y-y_hat
SNR=10*np.log10(np.sum(y**2)/np.sum(err**2))

plt.plot(y_hat,'b')
plt.plot(y,'r')
plt.plot(err,'k')
plt.legend(['error','y','y_hat'])
plt.title('SNR: %.2fdB'%SNR)
plt.show()







