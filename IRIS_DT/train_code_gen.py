#!/usr/bin/python3
# -*- coding: utf-8 -*-

####################
# 决策树训练程序
# 及C代码生成器
####################

import numpy as np
from sklearn.tree import DecisionTreeClassifier

## 自动代码生成（C语言）
# 输入：
#   tree            -- scikit-learn输出的决策树
#   fname           -- 文件名
def tree_to_c_code(tree, code_path=''):
    print('[INF] Generating code...')
    
    ## 生成C文件
    with open(code_path+'tree.c', 'wt') as fout:
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
        

####################
# 本地测试(应用演示)
####################

if __name__=='__main__':
    
    import IPython
    np.random.seed(1)

    EXPORT_CODE='export_code/'

    # 加载测试数据
    print('[INF] loading data...')
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

    # 决策树训练
    print('[INF] fitting decision tree...')
    cls = DecisionTreeClassifier(max_depth=TREE_DEP)
    cls.fit(x_train, y_train)

    # 训练结果测试
    print('[INF] testing...')
    y_pred=cls.predict(x_test)
    print('[INF] ACC: %.4f%%'%(100.0*(y_pred==y_test).astype(float).mean()))

    # 导出C代码
    print('[INF] generating C code (folder %s)...'%EXPORT_CODE)
    tree_to_c_code(cls,EXPORT_CODE)

    # 生成C语言头文件
    num_dat,num_dim=x_test.shape
    num_cls=len(set(y))
    
    print('[INF] generating tree.h...')
    with open(EXPORT_CODE+'tree.h', 'wt') as fout:
        fout.write('#ifndef __TREE_H__\n#define __TREE_H__\n\n')
        fout.write('#define NUM_CLS %d\n'%int(num_cls))
        fout.write('#define NUM_DIM %d\n'%int(num_dim))
        fout.write('\n#endif\n\n')

    # 生成测试数据文件
    print('[INF] generating test_tree_dat.h...')
    with open(EXPORT_CODE+'test_tree_dat.h', 'wt') as fout:
        fout.write('#ifndef __TEST_TREE_DAT_H__\n#define __TEST_TREE_DAT_H__\n\n')
        fout.write('#include "tree.h"\n\n')
        fout.write('#define NUM_DAT %d\n'%int(num_dat))
        fout.write('\n#endif\n\n')
        fout.close()

    print('[INF] generating test_tree_dat.c...')
    with open(EXPORT_CODE+'test_tree_dat.c', 'wt') as fout:
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

    # 编译生成C代码并测试运行
    if True:
        print('[INF] compile generated codes and execute...')
        import os,sys
        def exec_cmd(cmd):
            print('[INF]',cmd)
            os.system(cmd)
        exec_cmd('gcc -std=c99 -c %stree.c -o %stree.o'%(EXPORT_CODE,EXPORT_CODE))
        exec_cmd('gcc -std=c99 -c test_tree.c -I%s -o %stest_tree.o'%(EXPORT_CODE,EXPORT_CODE))
        exec_cmd('gcc -std=c99 -c %stest_tree_dat.c -o %stest_tree_dat.o'%(EXPORT_CODE,EXPORT_CODE))
        exec_cmd('gcc -o %stest_tree.exe %stree.o %stest_tree.o %stest_tree_dat.o'%(EXPORT_CODE,EXPORT_CODE,EXPORT_CODE,EXPORT_CODE))
        if sys.platform=='win32':
            print('[INF] windows')
            exec_cmd('.\\%s\\test_tree.exe'%EXPORT_CODE[:-1])
        else:
            print('[INF] linux')
            exec_cmd('./%stest_tree.exe'%EXPORT_CODE)
        
        
        
        
