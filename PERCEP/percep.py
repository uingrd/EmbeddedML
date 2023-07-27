#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import datasets

# 设置当前工作目录
import os,sys
os.chdir(sys.path[0])

# 加载数据集
iris = datasets.load_iris()

# 数据拆分为新联合测试集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    test_size = 0.3, 
                                                    random_state = 1, 
                                                    stratify = iris.target )
# 构建感知机分类器
model =  Perceptron()

# 利用训练数据集训练分类器
model.fit(x_train, y_train)

# 在测试数据机上运行分类器推理
y_pred = model.predict(x_test)

# 相似错误率
print("[INF] test error: %d/%d"%
      ((y_test != y_pred).sum(),len(y_test)))

# 手动验证算法的运算步骤
import numpy as np
b=model.intercept_.reshape(3,1)
W=model.coef_

f=W.dot(iris.data.reshape(-1,4).T)+b
t=np.argmax(f,axis=0)

e=np.sum(t.ravel().astype(int)!=iris.target.ravel().astype(int))
print('[INF] manually verify, error: %d/%d'%(e,len(t)))
if False:
    import IPython
    IPython.embed()

num_cls,num_dim=W.shape
num_dat=len(iris.target)

# 生成C语言头文件
print('[INF] generating C header file...')
fp=open('export_code/percep_test.h','wt')

fp.write('#ifndef __PERCEP_TEST_H__\n')
fp.write('#define __PERCEP_TEST_H__\n')
fp.write('\n')
fp.write('#ifdef USE_CMSIS\n')
fp.write('#include "arm_math.h"\n')
fp.write('#else\n')
fp.write('#include "../mat_f32.h"\n')
fp.write('#endif\n')
fp.write('\n')


fp.write('#define NUM_CLS %d\n'%num_cls)
fp.write('#define NUM_DIM %d\n'%num_dim)
fp.write('#define NUM_DAT %d\n'%num_dat)
fp.write('\n')
fp.write('#endif\n')

fp.close()

## 生成C源代码
print('[INF] generating C source...')
fp=open('export_code/percep_test.c','wt')
fp.write('#include "percep_test.h"\n')
fp.write('\n')
fp.write('const float32_t W_f32[NUM_DIM*NUM_CLS] = \n{')
for w in W:
    fp.write('\n    ')
    for w_ in w: 
        fp.write('%e, '%w_)
fp.write('\n};\n')   

fp.write('const float32_t b_f32[NUM_CLS] = \n{\n    ')
for b_ in b:
    fp.write('%e, '%b_[0])
fp.write('\n};\n')    

fp.write('const float32_t percep_test_in[NUM_DAT*NUM_DIM] = \n{')
for d_ in iris.data:
    fp.write('\n    ')
    for d in d_: fp.write('%e, '%d)
fp.write('\n};\n') 
fp.write('const uint32_t percep_test_out[] = \n{')
for n,t in enumerate(iris.target):
    if n%16==0: fp.write('\n    ')
    fp.write('%d, '%t)
fp.write('\n};\n')

fp.write('                                                              \n')
fp.write('int32_t argmax(float32_t *p, int32_t len)                     \n')
fp.write('{                                                             \n')
fp.write('    float32_t v=p[0];                                         \n')
fp.write('    int32_t idx=0;                                            \n')
fp.write('    for (int32_t n=1; n<len; n++)                             \n')
fp.write('        if (p[n]>v)                                           \n')
fp.write('        {                                                     \n')
fp.write('            v=p[n];                                           \n')
fp.write('            idx=n;                                            \n')
fp.write('        }                                                     \n')
fp.write('  return idx;                                                 \n')
fp.write('}                                                             \n')
fp.write('                                                              \n')
fp.write('int32_t percep_test()                                         \n')
fp.write('{                                                             \n')
fp.write('    uint32_t err=0;                                           \n')
fp.write('    float32_t t_f32[NUM_DIM];                                 \n')
fp.write('    float32_t a_f32[NUM_CLS];                                 \n')
fp.write('    float32_t c_f32[NUM_CLS];                                 \n')
fp.write('                                                              \n')
fp.write('    arm_matrix_instance_f32 W,a,b,c,t;                        \n')
fp.write('                                                              \n')
fp.write('    arm_mat_init_f32(&W, NUM_CLS, NUM_DIM,(float32_t *)W_f32);\n')
fp.write('    arm_mat_init_f32(&a, NUM_CLS, 1,              a_f32);     \n')
fp.write('    arm_mat_init_f32(&b, NUM_CLS, 1, (float32_t *)b_f32);     \n')
fp.write('    arm_mat_init_f32(&c, NUM_CLS, 1,              c_f32);     \n')
fp.write('    arm_mat_init_f32(&t, NUM_DIM, 1,              t_f32);     \n')
fp.write('                                                              \n')
fp.write('    err=0;                                                    \n')
fp.write('    for (int n=0; n<NUM_DAT; n++)                             \n')
fp.write('    {                                                         \n')
fp.write('        for (int m=0; m<NUM_DIM; m++)                         \n')
fp.write('            t_f32[m]=percep_test_in[n*NUM_DIM+m];             \n')
fp.write('        arm_mat_mult_f32(&W, &t, &a);                         \n')
fp.write('        arm_mat_add_f32 (&a, &b, &c);                         \n')
fp.write('        if (percep_test_out[n]!=argmax(c_f32,NUM_CLS))        \n')
fp.write('            err++;                                            \n')
fp.write('    }                                                         \n')
fp.write('    return err;                                               \n')
fp.write('}                                                             \n')
fp.write('\n')
fp.write('#include <stdio.h>\n')
fp.write('int main()\n')
fp.write('{\n')
fp.write('    int32_t num_err=percep_test();\n')
fp.write('    printf("[INF] test error: %d/%d\\n",num_err,NUM_DAT);\n')
fp.write('}\n')
fp.close()

