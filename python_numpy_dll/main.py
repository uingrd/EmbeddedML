#encoding:utf-8

import numpy as np
from ctypes import *

import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

api = CDLL('./libAdd.so')

# 输入输出参数都为整数简单C函数调用
print('---- api.add:%d (3)'% api.add(1, 2))
 
# 入参为整数指针，输出为整数的C函数调用
a=(c_int*2)()
a[0]=2
a[1]=10
print('---- api.add_point: %d (12)'%api.add_point(a))

# 输入输出都为单精度浮点数的C函数调用
api.add_f32.argtypes = (c_float, c_float)
api.add_f32.restype = c_float
print('---- api.add_f32: %.2f (4.6)'%api.add_f32(1.2, 3.4)) 

# 通过传递给C函数的(ctypes)指针存放计算结果的例子
b=(c_int*2)()
api.add_point_io(a, b)
print('---- api.add_point_io: b=[%d,%d] ([12,-8])'%(b[0],b[1]))

# numpy单精度浮点数组指针作为输入参数的C函数调用，返回单精度浮点数
x=np.arange(10).astype(np.float32)/7.0
api.sum_f32_point.argtypes = [POINTER(c_float), c_long]
api.sum_f32_point.restype = c_float
print('---- api.sum_f32_point: sum(x)=%.2f (%.2f)' % (api.sum_f32_point(x.ctypes.data_as(POINTER(c_float)),len(x)),
                                                      np.sum(x)))
# 输入为numpy单精度浮点数组指针，并且输出存放于numpy单精度浮点数组指针的C程序调用例子
x=np.arange(10).astype(np.float64)
y=np.zeros_like(x).astype(np.float64)
api.cumsum_f32_point(x.ctypes.data_as(POINTER(c_float)),\
                     y.ctypes.data_as(POINTER(c_float)),\
                     len(x))
print('---- api.cumsum_f32_point:')
print(y)
print('reference: [  0.   1.   3.   6.  10.  15.  21.  28.  36.  45.]')


# 输入为两个numpy单精度浮点数组指针，并且输出存放于numpy单精度浮点数组指针的C程序调用例子
x1=np.arange(10).astype(np.float32)
x2=np.arange(10).astype(np.float32)*100
y =np.zeros_like(x1).astype(np.float32)
api.add_f32_point(x1.ctypes.data_as(POINTER(c_float)),\
                  x2.ctypes.data_as(POINTER(c_float)),\
                   y.ctypes.data_as(POINTER(c_float)),\
                  len(x))
print('---- api.add_f32_point:')
print(y)
print('reference:',x1+x2)

