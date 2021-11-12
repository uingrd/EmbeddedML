#!/usr/bin/python3.5
# coding=utf-8

import numpy as np

#######################
# 演示使用反射变换实现矩阵
# 乘法的过程
#######################

QMIN,QMAX=0,255 # 量化表示的最大最小值

## 根据numpy矩阵data分析并提取量化参数
def calc_quant_param(data):
    vmin,vmax=np.min(data.ravel()),np.max(data.ravel())
    s=float(vmax-vmin)/float(QMAX-QMIN)
    z=float(QMIN)-float(vmin)/s
    z=int(round(np.clip(z,QMIN,QMAX)))
    return s,z

## 数据量化，计算：dq=round(d/s+z)
def calc_quant_data(d,s,z):
    dq=d/s+z
    dq=np.round(np.clip(dq,QMIN,QMAX)).astype(int)
    return dq

## 数据反量化，计算：d=s*(dq-z)
def calc_dequant_data(dq,s,z):
    return s*(dq-z).astype(float)

## 量化矩阵乘法
# 从Aq、Bq计算C=A*B的量化表示Cq
def quant_matmul(Aq, sa, za, 
                 Bq, sb, zb,
                     sc, zc):
    # 整数乘法
    Cq=np.dot(Aq-za,
              Bq-zb)    
    # 乘以常数系数(浮点数)，可以用整数乘法近似，但这里为演示简单，简单使用了浮点乘法
    Cq=(sa*sb/sc)*Cq.astype(float)  
    Cq=np.round(Cq).astype(int)+zc
    return Cq

####################
# 单元测试           
####################
if __name__ == '__main__':
    np.random.seed(1234)

    # 生成2个随机矩阵
    A=np.random.randn(2,3)      
    B=np.random.randn(3,3)

    # 用浮点运算计算参考答案
    C_ref=np.dot(A,B)

    # 计算A的量化参数sa,za和量化矩阵Aq
    # A=sa*(Aq-za)
    sa,za=calc_quant_param(A)   
    Aq=calc_quant_data(A,sa,za)

    #显示量化结果
    print('sa:%f, za:%f'%(sa,za))
    print('Aq:\n',Aq)
    print('A:\n',A)
    print('recovered A:\n',calc_dequant_data(Aq,sa,za))

    # 计算A的量化参数sa,za和量化矩阵Aq
    # A=sa*(Aq-za)
    sb,zb=calc_quant_param(B)   
    Bq=calc_quant_data(B,sb,zb)

    print('sb:%f, zb:%f'%(sb,zb))
    print('Bq:\n',Bq)
    print('B:\n',B)
    print('recovered B:\n',calc_dequant_data(Bq,sb,zb))

    # 计算C的量化参数sc,zc
    # 注意，实际运算时sc,zc是通过数据统计
    # 事先指定的，不会像这里从答案计算得到
    sc,zc=calc_quant_param(C_ref)   

    ## 使用量化形式计算乘法
    Cq=quant_matmul(Aq,sa,za,
                    Bq,sb,zb,
                       sc,zc)
                       
    ## 比较计算误差，我们将Cq反量化后和参考答案比较
    C=calc_dequant_data(Cq,sc,zc)

    print('sc:%f, zc:%f'%(sc,zc))
    print('Cq:\n',Cq)
    print('C:\n',C)
    print('reference C:\n',C_ref)
    print('relative err:',np.linalg.norm(C-C_ref)/np.linalg.norm(C))


