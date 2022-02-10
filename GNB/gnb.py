#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 高斯朴素贝叶斯模型

from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(iris.data, iris.target)
y_pred = model.predict(iris.data)

print('[INF] test error: %d/%d'%
      ((iris.target != y_pred).sum(),iris.data.shape[0]))

#num_cls, num_dim = model.sigma_.shape
num_cls, num_dim = model.var_.shape
num_dat=len(iris['target'])

import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

# 生成C头文件
print('[INF] generating C header...')
fp=open('export_code/gnb_test.h','wt')
fp.write('#ifndef __GNB_TEST_H__\n')
fp.write('#define __GNB_TEST_H__\n')
fp.write('\n')
fp.write('#ifdef USE_CMSIS\n')
fp.write('#include "arm_math.h"\n')
fp.write('#else\n')
fp.write('#include "../gnb.h"\n')
fp.write('#endif\n')
fp.write('\n')
fp.write('#define NUM_CLS %d\n'%num_cls)
fp.write('#define NUM_DIM %d\n'%num_dim)
fp.write('#define NUM_DAT %d\n'%num_dat)
fp.write('#endif\n')
fp.close()

## 生成C源代码
print('[INF] generating C source...')
fp=open('export_code/gnb_test.c','wt')
fp.write('#ifdef USE_CMSIS\n')
fp.write('#include "arm_math.h"\n')
fp.write('#else\n')
fp.write('#include "../gnb.h"\n')
fp.write('#endif\n')
fp.write('\n')
fp.write('#include "gnb_test.h"\n')
fp.write('\n')
fp.write('const float32_t model_theta[NUM_CLS*NUM_DIM] = \n{')
for t_ in model.theta_:
    fp.write('\n    ')
    for t in t_: fp.write('%e, '%t)
fp.write('\n};\n')   

fp.write('const float32_t model_sigma[NUM_CLS*NUM_DIM] = \n{')
for s_ in model.var_:
    fp.write('\n    ')
    for s in s_: fp.write('%e, '%s)
fp.write('\n};\n')    

fp.write('const float32_t model_priors[NUM_CLS] = \n{\n    ')
if model.class_prior_ is not None:
    for p in model.class_prior_: fp.write('%e, '%p)
else:
    for _ in range(num_cls): fp.write('%e, '%(1.0/float(num_cls)))
fp.write('\n};\n')
fp.write('\n')

fp.write('const float32_t test_in[NUM_DAT*NUM_DIM] = \n{')
for d_ in iris['data']:
    fp.write('\n    ')
    for d in d_: fp.write('%e, '%d)
fp.write('\n};\n') 
fp.write('const uint32_t test_out[] = \n{')
for n,t in enumerate(iris['target']):
    if n%16==0: fp.write('\n    ')
    fp.write('%d, '%t)
fp.write('\n};\n')

fp.write('\n')
fp.write('int32_t gnb_test()\n')
fp.write('{\n')
fp.write('    uint32_t err=0;\n')
fp.write('    float32_t prob[NUM_CLS];\n')
fp.write('\n')
fp.write('    arm_gaussian_naive_bayes_instance_f32 model;\n')
fp.write('    model.vectorDimension=NUM_DIM;\n')
fp.write('    model.numberOfClasses=NUM_CLS;\n')              
fp.write('    model.theta=model_theta;\n')
fp.write('    model.sigma=model_sigma;\n')
fp.write('    model.classPriors=model_priors;\n')

fp.write('    model.epsilon=0.0;\n')
#fp.write('    model.epsilon=%e;\n'%model.epsilon_)

fp.write('\n')
fp.write('    err=0;\n')
fp.write('    for (int n=0; n<NUM_DAT; n++)\n')
fp.write('        if (test_out[n]!=arm_gaussian_naive_bayes_predict_f32(&model,test_in+n*NUM_DIM,prob))\n')
fp.write('            err++;\n')
fp.write('    return err;\n')
fp.write('}\n')

fp.write('\n')
fp.write('#include <stdio.h>\n')
fp.write('int main()\n')
fp.write('{\n')
fp.write('    int32_t num_err=gnb_test();\n')
fp.write('    printf("[INF] test error: %d/%d\\n",num_err,NUM_DAT);\n')
fp.write('}\n')

fp.close()


