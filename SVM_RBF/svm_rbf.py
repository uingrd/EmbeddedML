#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(1234)

ENABLE_PCA=True
PCA_DIM=3

from sklearn import datasets
print('[INF] loading data...')
data = datasets.load_breast_cancer()
x,y=data.data,data.target

# 训练/测试数据集分离
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,shuffle=True)

num_dat=len(y)
num_dim=x[1]

print('[INF] num training:',len(train_y))
print('[INF] num testing:' ,len(test_y))

# 数据降维
if ENABLE_PCA:
    from sklearn.decomposition import PCA
    print('[INF] PCA')
    pca = PCA(n_components=PCA_DIM)
    train_x_ori,test_x_ori=train_x.copy(),test_x.copy()
    train_x = pca.fit_transform(np.array(train_x))
    test_x  = pca.transform(test_x)
    
    num_dim=train_x.shape[1]
    pca_in_dim=train_x_ori.shape[1]
    print('[INF] data dim:',num_dim)

if False:
    import IPython
    IPython.embed()
    
# 训练SVM模型
print('[INF] training SVM model...')
model = svm.NuSVC(gamma=1.5e-4,kernel='rbf')
model.fit(train_x, train_y)

# 测试训练结果
print('[INF] testing model...')
y_pred = model.predict(test_x)
print('[INF] num err:%d'%np.sum(y_pred!=test_y))
print('[INF] ACC:%2f%%'%(100.0*np.mean(y_pred==test_y)))


num_sv,num_dim=model.support_vectors_.shape
num_dat=len(test_y)

print('[INF] num_sv:',num_sv)
print('[INF] num_dim:',num_dim)
print('[INF] num_dat:',num_dat)

# 生成C语言头文件
print('[INF] generating C header file...')
fp=open('./export_code/svm_rbf_test.h','wt')
fp.write('#ifndef __SVM_RBF_TEST_H__\n')
fp.write('#define __SVM_RBF_TEST_H__\n')
fp.write('\n')
fp.write('#define SVM_RBF_NUM_SV %d\n'%num_sv)
fp.write('#define SVM_RBF_NUM_DIM %d\n'%num_dim)
fp.write('#define SVM_RBF_NUM_DAT %d\n'%num_dat)
if ENABLE_PCA:
    fp.write('#define PCA_IN_DIM %d\n'%pca_in_dim)

fp.write('\n')
fp.write('#endif')
fp.close()

####################
# 生成C语言源代码
print('[INF] generating C source file...')
fp=open('./export_code/svm_rbf_test.c','wt')
fp.write('#include <stdint.h>\n')
fp.write('#include <stdio.h>\n')
fp.write('#include "svm_rbf_test.h"\n')
fp.write('#include "../svm_rbf.h"\n')
fp.write('\n')

# 权重系数
fp.write('const float model_dual_coef[SVM_RBF_NUM_SV] = \n{')
for n,t in enumerate(model.dual_coef_.flatten()):
    if n%16==0: fp.write('\n    ')
    fp.write('%e, '%t)
fp.write('\n};\n')

# 支持向量
fp.write('const float model_sv[SVM_RBF_NUM_SV*SVM_RBF_NUM_DIM] = \n{')
for sv_ in model.support_vectors_:
    fp.write('\n    ')
    for sv in sv_: fp.write('%e, '%sv)
fp.write('\n};\n')   

# 类别编码
fp.write('const int32_t model_cls[2]={0,1};\n')

# 测试数据输入
if ENABLE_PCA:
    fp.write('const float svm_rbf_test_in[SVM_RBF_NUM_DAT*PCA_IN_DIM] = \n{')
    for d_ in test_x_ori:
        fp.write('\n    ')
        for d in d_: fp.write('%e, '%d)
else:
    fp.write('const float svm_rbf_test_in[SVM_RBF_NUM_DAT*SVM_RBF_NUM_DIM] = \n{')
    for d_ in test_x:
        fp.write('\n    ')
        for d in d_: fp.write('%e, '%d)
fp.write('\n};\n') 
# 测试参考输出
fp.write('const int32_t svm_rbf_test_out[SVM_RBF_NUM_DAT] = \n{')
for n,t in enumerate(test_y):
    if n%16==0: fp.write('\n    ')
    fp.write('%d, '%t)
fp.write('\n};\n')

# PCA降维数据
if ENABLE_PCA:
    fp.write('\n')
    fp.write('const float pca_dat[SVM_RBF_NUM_DIM*PCA_IN_DIM] = \n{')
    for c_ in pca.components_:
        fp.write('\n    ')
        for c in c_: fp.write('%e, '%c)
    fp.write('\n};\n') 
    
    fp.write('const float pca_dat_m[PCA_IN_DIM] = \n{')
    for n,m in enumerate(pca.mean_):
        if n%16==0: fp.write('\n    ')
        fp.write('%e, '%m)
    fp.write('\n};\n') 
    

# 测试代码
fp.write('\n')
fp.write('int32_t main()\n')
fp.write('{\n')
fp.write('    uint32_t err=0;\n')
fp.write('    int32_t  res=0;\n')
fp.write('\n')
# PCA矩阵对象    
if ENABLE_PCA:
    fp.write('    float pca_out1[PCA_IN_DIM];\n')       # 存放输入数据去均值结果(未降维)的数组
    fp.write('    float pca_out[SVM_RBF_NUM_DIM];\n')   # 存放降维输出数据的数组
    
    fp.write('    arm_matrix_instance_f32 mat_pca;\n')      # 降维矩阵
    fp.write('    arm_matrix_instance_f32 mat_pca_m;\n')    # 数据去均值
    fp.write('    arm_matrix_instance_f32 mat_pca_in;\n')   # 降维前数据
    fp.write('    arm_matrix_instance_f32 mat_pca_out1;\n') # 输入数据去均值结果(未降维)
    fp.write('    arm_matrix_instance_f32 mat_pca_out;\n')  # 降维输出

# 构建SVM模型数据结构
fp.write('\n')    
fp.write('    arm_svm_rbf_instance_f32 model;\n')
fp.write('    model.nbOfSupportVectors= SVM_RBF_NUM_SV;\n')
fp.write('    model.vectorDimension   = SVM_RBF_NUM_DIM;\n')              
fp.write('    model.intercept         = %e;\n'%model.intercept_.flatten()[0])
fp.write('    model.dualCoefficients  = model_dual_coef;\n')
fp.write('    model.supportVectors    = model_sv;\n')
fp.write('    model.classes           = model_cls;\n')
fp.write('    model.gamma             = %e;\n'%model.gamma)

# PCA变换矩阵的数据结构填充
if ENABLE_PCA:
    fp.write('\n')
    fp.write('    arm_mat_init_f32(&mat_pca, SVM_RBF_NUM_DIM, PCA_IN_DIM, (float *)pca_dat);\n')
    fp.write('    arm_mat_init_f32(&mat_pca_m, PCA_IN_DIM, 1, (float *)pca_dat_m);\n')
    fp.write('    arm_mat_init_f32(&mat_pca_out1, PCA_IN_DIM, 1, pca_out1);\n')
    fp.write('    arm_mat_init_f32(&mat_pca_out, SVM_RBF_NUM_DIM, 1, pca_out);\n')    
    
fp.write('\n')
fp.write('    err=0;\n')
fp.write('    for (int n=0; n<SVM_RBF_NUM_DAT; n++)\n')
fp.write('    {\n')
# PCA降维计算
if ENABLE_PCA:
    fp.write('        arm_mat_init_f32(&mat_pca_in, PCA_IN_DIM, 1, (float *)svm_rbf_test_in+n*PCA_IN_DIM);\n')  
    fp.write('        arm_mat_sub_f32(&mat_pca_in, &mat_pca_m , &mat_pca_out1);\n')
    fp.write('        arm_mat_mult_f32(&mat_pca, &mat_pca_out1, &mat_pca_out);\n')
    # 计算分类
    fp.write('        arm_svm_rbf_predict_f32(&model,pca_out,&res);\n')
else:
    # 计算分类
    fp.write('        arm_svm_rbf_predict_f32(&model,svm_rbf_test_in+n*SVM_RBF_NUM_DIM,&res);\n')
fp.write('        if (svm_rbf_test_out[n]!=res) err++;\n')
fp.write('    }\n')
fp.write('    printf("[INF] num err:%d\\n",err);\n')
fp.write('    printf("[INF] ACC:%.6f%%\\n",100.0-100.0*(float)err/(float)SVM_RBF_NUM_DAT);\n')
fp.write('    return err;\n')
fp.write('}\n')
fp.close()

print('[INF] end')
if False:
    import IPython
    IPython.embed()
