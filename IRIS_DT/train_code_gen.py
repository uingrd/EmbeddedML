#!/usr/bin/python3
# -*- coding: utf-8 -*-

####################
# 决策树训练程序
# 及C代码生成器
####################

import numpy as np
import IPython

np.random.seed(1)

from sklearn.tree import DecisionTreeClassifier

## 自动代码生成（C语言）
# 输入：
#   tree            -- scikit-learn输出的决策树
#   fname           -- 文件名
def tree_to_c_code(tree):
    print('[INF] Generating code...')
    
    ## 生成C文件
    with open('tree.c', 'wt') as fout:
        def recurse(node=0, prefix='    '):
            if left[node]==-1 and right[node]==-1:  # leaf node
                for i,v in enumerate(value[node][0]):
                    if v==0: continue
                    fout.write(prefix+'target['+str(i)+']='+str(int(v))+';\n')
            else:
                fout.write(prefix + 'if (feature['+str(feature[node])+']<=' +str(threshold[node])+')\n')
                fout.write(prefix + '{\n')
                if t.children_left[node] != -1: recurse(left[node], prefix+'    ')
                fout.write(prefix + '}\n')
                fout.write(prefix + 'else\n')
                fout.write(prefix + '{\n')
                if t.children_right[node] != -1: recurse(right[node], prefix+'    ')
                fout.write(prefix + '}\n')
                
        t=tree.tree_
        left,right=t.children_left,t.children_right
        value,threshold,feature=t.value,t.threshold,t.feature
        
        fout.write('#include "tree.h"\n\n')
        fout.write('void tree(float *feature, int *target)\n')
        fout.write('{\n')
        fout.write('    for (int n=0; n<NUM_CLS; n++) target[n]=0;\n')
        recurse()
        fout.write('}\n\n')
        

# 加载测试数据
import sklearn.datasets as datasets
if True:
    data=datasets.load_iris()
    TREE_DEP=5
else:
    data=datasets.load_digits()
    TREE_DEP=9

x,y=data['data'],data['target']


# 训练/测试数据集分离
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,shuffle=True)

if True:
    print('[INF] x_train:',x_train)
    print('[INF] y_train:',y_train)
    
# 决策树训练
cls = DecisionTreeClassifier(max_depth=TREE_DEP)
cls.fit(x_train, y_train)

# 训练结果测试
y_pred=cls.predict(x_test)
print('[INF] ACC: %.4f%%'%(100.0*(y_pred==y_test).astype(float).mean()))

# 导出C代码
tree_to_c_code(cls)

# 生成C语言头文件
num_dat,num_dim=x_test.shape
num_cls=len(set(y))
print('[INF] num_cls:',num_cls)

with open('tree.h', 'wt') as fout:
    fout.write('#ifndef __TREE_H__\n#define __TREE_H__\n\n')
    fout.write('#define NUM_CLS %d\n'%int(num_cls))
    fout.write('#define NUM_DIM %d\n'%int(num_dim))
    fout.write('\n#endif\n\n')

# 生成测试数据文件
with open('test_tree_dat.h', 'wt') as fout:
    fout.write('#ifndef __TEST_TREE_DAT_H__\n#define __TEST_TREE_DAT_H__\n\n')
    fout.write('#include "tree.h"\n\n')
    fout.write('#define NUM_DAT %d\n'%int(num_dat))
    fout.write('\n#endif\n\n')
    fout.close()

with open('test_tree_dat.c', 'wt') as fout:
    fout.write('#include "test_tree_dat.h"\n\n')

    fout.write('const float dt_tree_test_in[NUM_DAT*NUM_DIM] = \n{')
    for d_ in x_test:               # 测试输入
        fout.write('\n    ')
        for d in d_: fout.write('(float)%e, '%d)
    fout.write('\n};\n\n') 

    fout.write('const int dt_tree_test_out[NUM_DAT] = \n{')
    for n,t in enumerate(y_test):   # 参考输出
        if n%16==0: fout.write('\n    ')
        fout.write('%d, '%t)
    fout.write('\n};\n\n')

# 显示训练结果
if False:
    from sklearn.tree import export_text
    print(export_text(cls,[str(i) for i in range(num_feature)]))

# 编译
if False:
    import os
    os.system('gcc -c tree.c')
    os.system('gcc -c test_tree.c')
    os.system('gcc -c test_tree_dat.c')
    os.system('gcc -o test_tree.exe tree.o test_tree.o test_tree_dat.o')
    os.system('test_tree.exe')
    
    
    
    
