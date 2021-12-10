# 该例程演示使用scikit-learn训练决策树分类器，并自动生成C语言代码(数据文件)实现分类器推理运算

1. 该代码训练的决策树实现IRIS花卉分类任务

2. train_code_gen.py完成决策树训练和代码生成
  生成C语言测试代码和参数文件存放于export_code目录，分别为：
    - tree.c          决策树分类器
    - tree.h     
    - test_tree_dat.c 测试数据
    - test_tree_dat.h

3. 代码tree.c实现基于决策树的分类推理，它连同train_code_gen.py生成的4个C文件一起编译得到决策树推理的测试程序
   编译命令是：gcc -o test_tree -Iexport_code export_code/tree.c test_tree.c export_code/test_tree_dat.c -std=c99，编译生成test_tree(.exe)。
   运行该程序执行决策树推理

4. 备注：
- C语言例程不依赖第三方机器学习框架
- 测试用的C程序可以直接改用嵌入式处理器的编译器编译并运行
