# 基于CSD格式的常数乘法表达式生成和C代码生成
- 将给定整数常数转成CSD格式并自动生成测试代码实现快速整数乘法
- 主要API说明：
  - int16_to_code(v): 将16-bit整数v转成CSD格式并输出使用移位加减合成该数据的公式
  - int16_to_c_code(v,fname): 将16-bit整数v转成CSD格式并自动生成C程序实现快速乘法运算
- 使用说明
  - 运行CSD.py测试不同数据的CSD生成，并生成测试代码：export_code/auto_code.c
  - 运行build.py编译生成的export_code/auto_code.c得到可执行程序export_code/test，运行test输出测试结果(如果正确的话输出："[OK]")
