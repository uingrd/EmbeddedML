#!/usr/bin/python3
# coding=utf-8

import numpy as np
import sympy as sp

################################
# AlphaTensor快速矩阵计算表达式提取
# 参考论文：Fawzi, A. et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature 610 (2022)
# 代码参考：https://github.com/deepmind/alphatensor/tree/main/algorithms
################################

NUM_EQU_VERIFY=10   # 验证的矩阵乘法分解式数目，如果想验证所有分解式的话，可以设置为np.inf
PRINT_EQU=True      # 是否打印矩阵计算表达式

# 根据原始矩阵乘法构建张量表示
def get_tensor(a: int, b: int, c: int) -> np.ndarray:
    result = np.full((a*b, b*c, c*a), 0, dtype=np.int32)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                result[i * b  + j][j * c + k][k * a + i] = 1
    return result

# 基于张量低秩表示(u,v,w)计算矩阵(x,y)的乘积
def calc_mat_mul(x,y,u,v,w):
    xf,yf=x.ravel(),y.ravel()
    zf=np.zeros(x.shape[0]*y.shape[1])
    for u_,v_,w_ in zip(u.T,v.T,w.T):
        p=(xf*u_).sum()*(yf*v_).sum()
        zf+=p*w_
    return zf.reshape(y.shape[1],x.shape[0]).T

# 打印矩阵乘法的符号分解
def sym_mat_mul(a,b,c,u,v,w):
    calc_str=[]
    
    xf=sp.symbols('x[0:%d_0:%d]'%(a,b))
    yf=sp.symbols('y[0:%d_0:%d]'%(b,c))
    zf=np.zeros(c*a)
    pr=sp.symbols('p[0:%s]'%u.shape[1])
    
    for r,(u_,v_,w_) in enumerate(zip(u.T,v.T,w.T)):
        px,py=(xf*u_).sum(),(yf*v_).sum()
        sx,sy=str(px).replace('_',','),str(py).replace('_',',')
        if (u_!=0).sum()>1: sx='(%s)'%sx
        if (v_!=0).sum()>1: sy='(%s)'%sy
        calc_str.append('p[%d] = %s * %s'%(r,sx,sy))
        
        zf=zf+w_*pr[r]
    
    for i in range(a):
        for k in range(c):
            calc_str.append('z[%d,%d] = %s'%(i,k,str(zf[k*a+i])))
    
    return calc_str
    

# 测试快速算法
np.random.seed(1234)
EPS=1.0e-11

# 加载所有快速矩阵乘法的张量低秩分解
f=np.load('factorizations_r.npz',allow_pickle=True)
for n,k in enumerate(f.keys()):
    if n>NUM_EQU_VERIFY: break
    
    # 张量分解的正确性验证
    a,b,c=(eval(s) for s in k.split(','))
    t0= get_tensor(a, b, c)                 # 原始矩阵乘法对应的张量表示

    u, v, w = f[k]                          # 提取分解结果
    r=u.shape[-1]                           # 分解的秩（对应乘法次数）
    t1=np.einsum('ir,jr,kr->ijk', u, v, w)  # 重构矩阵乘法构建张量表示
    
    check='ERROR' if not np.array_equal(t0,t1) else ''
    print(f'size:({k}), rank:{r}, ({a*b*c}, {r/a/b/c*100}%) {check}')

    # 数值矩阵的测试
    x=np.random.randn(a,b)
    y=np.random.randn(b,c)
    z0=x.dot(y)

    if not PRINT_EQU:
        # 基于数值计算
        z=calc_mat_mul(x,y,u,v,w)
    else:
        # 先生成计算表达式，在进行数值计算
        z,p=np.zeros((a,c)),[0]*r
        calc_str=sym_mat_mul(a,b,c,u,v,w)
        print('------------------------')
        for s in calc_str: 
            if True: print(s)
            exec(s)
    
    e=np.abs(z0-z).max()
    check='ERROR' if e>EPS else ''
    if check:
        print('value test error:%.2e %s'%(e,check))
    print()
    
    

