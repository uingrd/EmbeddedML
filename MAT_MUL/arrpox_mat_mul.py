#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 基于数据低秩近似的算法
#
# 代码清单6-5
#
####################

## 近似矩阵乘法的数据准备
# 用于计算：y=Ax，利用x的统计相关特性近似
# 该函数计算近似运算所需要的向量集{w1,w2,...}和
# 矩阵A经预处理得到向量集{a1,a2,...}
# 输入
#   A       常数矩阵
#   x_all   矩阵，每一列存放一个训练用的输入向量
#   R       表明算法使用x的秩R近似
# 输出
#   ar_all  矩阵，每列分别对应{a1,a2,...aR}
#   wr_all  矩阵，每列分别对应{w1,w2,...wR}
def calc_ar_wr(A,x_all,R):
    Cx=x.dot(x.T)
    e,w=np.linalg.eig(Cx)

    # 验证x低秩
    if False:        
        plt.plot(np.sort(e))
        plt.title('singular value')
        plt.show()
        
    # 特征值分解，计算x的秩空间，w是降维矩阵，w的各列是单位长度正交向量
    idx=np.argsort(e)[-R:]
    
    wr_all=w[:,idx]
    ar_all=A.dot(wr_all)
    
    return ar_all,wr_all

## 计算近似矩阵乘法
# 使用预先计算并保存的ar_all={a1,a2,...,aR}和wr_all={w1,w2,...,wR}
def mat_mul_approx(ar_all,wr_all,x):
    return ar_all.dot(wr_all.T.dot(x))

##########
# 单元测试
##########
if __name__ == '__main__':
    import pylab as plt
    np.random.seed(4321)

    L=20    # x向量长度
    N=40    # 矩阵A行数
    R=8     # 降维后向量长度

    print('[INF] L*N:',L*N)         # 降维前运算量
    print('[INF] (L+N)*R:',(L+N)*R) # 降维后运算量

    K=2000  # 训练用向量数目

    # 构造测试矩阵
    A=np.random.randn(N,L)

    # 构造低秩x
    x1=np.random.randn(L*K)
    for _ in range(40): x1=np.convolve(x1,np.ones(100)/100.0,mode='same')
    x2=np.random.randn(L*K)/40
    for _ in range(10): x2=np.convolve(x2,np.ones(10)/10.0,mode='same')
    x_all=x1+x2
    x_all.shape=L,K
    x=x_all

    # 基于训练数据x_all计算近似矩阵乘法需要的数据
    # 数据x_all实矩阵，每一列对应一个L行的数据样本
    # A是用于计算的(常数)矩阵，R表示数据x按R维近似
    ar_all,wr_all=calc_ar_wr(A,x_all,R)

    # 对输入x计算近似矩阵乘法
    y_approx=mat_mul_approx(ar_all,wr_all,x)

    # 精确矩阵乘法
    y =A.dot(x )

    # 检验近似误差
    if True:
        plt.clf()
        plt.plot(y_approx.ravel(),'b')
        plt.plot(y.ravel(),'r')
        plt.plot(y_approx.ravel()-y.ravel(),'k')
        plt.legend(['approx.', 'reference', 'error'])
        plt.title('comparison')
        plt.show()

        err=np.linalg.norm(y_approx-y)/np.linalg.norm(y)
        print('[INF] relative error: %0.2f'%err)






