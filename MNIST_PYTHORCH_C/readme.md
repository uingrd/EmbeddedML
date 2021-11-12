# 该例程演示使用pytorch训练MNIST手写数字识别神经网络，并自动生成C语言代码(数据文件)实现神经网络推理运算

1. main.py执行的步骤
  - 训练神经网络，训练结果保存于目录：export_model/mnist_cnn.pth
  - 生成C语言测试代码和参数文件，即
    - export_code/param.c     网络参数
    - export_code/param.h     
    - export_code/test_data.c 测试数据
    - export_code/test_data.h

2. 代码main.c实现MNIST CNN神经网络推理
   编译命令是：
   gcc -o mnist mnist.c export_code/param.c export_code/test_data.c -std=c99
   (由于数据文件较大，编译需要一些时间完成),
   编译生成mnist(.exe)。
   运行该程序执行神经网络推理，精度大约98%

备注：
- C语言例程不依赖第三方机器学习框架
- 虽然测试用的C程序由VC编译运行，它也可以直接改成用嵌入式处理器的编译器编译并运行
