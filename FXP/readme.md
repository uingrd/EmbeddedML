# 数据定点化以及基于定点运算的FIR滤波演示
- fxp_filter.py 演示FIR滤波器的定点化过程，并自动生成C测试代码fxp_filter_data.c
- fxp_filter.c 通过定点数运算实现FIR滤波器的C代码，
    - 该代码读取fxp_filter_data.c文件内的待滤波数据和滤波器抽头(该文件由fxp_filter.py自动生成)
    - 通过整数运算得到FIR滤波结果，并保存于文件y_int.bin
    - 该文件和python程序配套，用于比对python执行定点数运算结果和C执行定点数运算结果的一致性
    - 它的编译命令是：
    gcc -o fxp_filter fxp_filter_data.c fxp_filter.c -std=c99
    编译结果是：fxp_filter(.exe)
