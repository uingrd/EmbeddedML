# 基于scikit-learn的感知器分类器演示
1. percep.py使用scikit-learn中的感知器模块训练二分类器，训练结果自动生成C测试源代码（export_code目录下）
2. 编译脚本build.py编译自动生成的C代码，得到测试程序
3. 文件说明:
    - percep.py: 训练分类器并自动生成测试C代码
    - build.py: 编译测试C代码，生成可执行测试程序并运行测试
    - mat_f32.c, mat_f32.h:  矩阵计算函数，不使用CMSIS，替代其中的矩阵运算函数
    - export_code/percep_test.c, export_code/percep_test.h: 测试代码和数据文件(percep.py自动生成)
4. 备注:
    - build.py目前仅测试了在Ubuntu和macOS上的执行
    - 文件mat_f32.c, mat_f32.h的功能在ARM的CMSIS库中有对应的实现，可以通过编译选项 -DUSE_CMSIS 选择使用CMSIS库的实现
    - 虽然C测试代码在PC上运行，该程序能够直接移植到嵌入式系统实现
    - 测试代码的数据验证精度是150个分类样本错6个（和训练结果一致）

