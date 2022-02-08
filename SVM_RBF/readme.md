# 基于scikit-learn的支持向量机二分类器演示
1. 例程svm_rbf.py使用scikit-learn中的SVM模块训练二分类器，训练结果自动生成C测试源代码（export_code目录下）
2. 编译脚本build.py编译自动生成的SVM的C代码，得到测试程序
3. 文件说明:
    - svm_rbf.py: 训练支持向量机并自动生成测试C代码
    - build.py: 编译测试C代码，生成可执行测试程序并运行测试
    - svm_rbf.c, svm_rbf.h: SVM的底层运算实现
    - mat_f32.c, mat_f32.h: 矩阵运算代码
    - export_code/svm_rbf_test.c, export_code/svm_rbf_test.h: 测试数据文件(svm_rbf.py自动生成)
4. 备注:
    - build.py目前仅测试了在Ubuntu和macOS上的执行
    - 文件svm_rbf.c, svm_rbf.h, mat_f32.c, mat_f32.h在ARM的CMSIS库中有对应的实现，可以通过编译选项 -DUSE_CMSIS 选择使用CMSIS库的实现
    - 虽然C测试代码在PC上运行，该程序能够直接移植到嵌入式系统实现
    - 测试代码的数据验证精度是93.567251%

