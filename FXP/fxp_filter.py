#coding:utf-8
#!/usr/bin/python

import os, sys, platform, IPython
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz

import matplotlib.pylab as plt
from data_format import *

np.random.seed(1234)

# 设置当前运行目录
os.chdir(sys.path[0])

####################
# 演示对通过定点数运算实现FIR近似滤波
####################

## 定点化参数
NUM_BITS =8             # 定点数的总量化bit数
TAP_SHIFT=2
TAP_SCALE=2.0**TAP_SHIFT# 滤波器系数缩放比例因子

## 生成演示用的FIR滤波器参数
FS   = 1000.0                                   # 采样率
numtaps, beta = kaiserord(20.0, 20.0/FS)        # 滤波器参数估算
taps = firwin(numtaps, [0.04,0.06], window=('kaiser', beta), pass_zero=False)

## 生成滤波器输入数据(带噪声测试信号)
N = 1000
t = np.arange(N)/FS
x = np.cos(np.pi*50.0*t)+np.random.randn(N)

## 计算数据的定点化格式参数
# 返回
#   n: 定点数整数位宽
#   m: 定点数小数位宽
# 对应格式参数Sn.m
def find_fxp_fmt(v,num_bits):
    # 计算整数位宽
    n=int(np.ceil(np.log2(np.max(np.abs(v)))))
    n=max(0,min(num_bits-1,n))
    m=num_bits-n-1
    return n,m

## 计算精确滤波结果
y = lfilter(taps, 1.0, x)
if True:
    plt.plot(taps,'k-')
    plt.title('filter taps (float)')
    plt.grid(True)
    plt.show()
    
    plt.plot(t,x,'b-')
    plt.plot(t,y,'r-')
    plt.title('filter IO (float)')
    plt.legend(['in', 'out'])
    plt.xlabel('t')
    plt.grid(True)
    plt.show()

## 计算滤波器系数的定点数格式(注意：经过了尺度变化TAP_SCALE)
nbits_taps,mbits_taps=find_fxp_fmt(taps*TAP_SCALE,NUM_BITS)
print('[INF] np.max(abs(taps*scale))=%f (scale=%f)'%(np.max(abs(taps*TAP_SCALE)),TAP_SCALE))
print('[INF] filter taps fmt.: S%d.%d (scale=%f)'%(nbits_taps,mbits_taps,TAP_SCALE))

## 滤波器系数定点化转换(注意：考虑了尺度变化TAP_SCALE)
fxp_taps_int=np.round(taps*TAP_SCALE*(2**mbits_taps)).astype(int)   # 定点数的符号表示(整数格式)
fxp_taps_f32=fxp_taps_int/(2.0**mbits_taps)                         # 定点数对应的值(浮点格式)
plt.clf()                                                           # 显示权重系数量化误差
plt.plot(fxp_taps_f32/TAP_SCALE, 'b')
plt.plot(taps, 'r')
plt.plot(taps-fxp_taps_f32/TAP_SCALE, 'k')
plt.title('filter taps quant. err.')
plt.legend(['taps (fxp)', 'taps (float)','quant. err.'])
plt.show()

## 计算滤波器输入的定点数格式
nbits_x,mbits_x=find_fxp_fmt(x,NUM_BITS)
print('[INF] np.max(abs(x)):',np.max(abs(x)))
print('[INF] filter input fmt.: S%d.%d'%(nbits_x,mbits_x))

## 滤波器输入的定点化转换
fxp_x_int=np.round(x*(2**mbits_x)).astype(int)                      # 定点数的符号表示(整数格式)
fxp_x_f32=fxp_x_int/(2.0**mbits_x)                                  # 定点数对应的值(浮点格式)
plt.clf()   # 显示输入量化误差
plt.plot(fxp_x_f32,'b')
plt.plot(x, 'r')
plt.plot(x-fxp_x_f32, 'k')
plt.title('input quant. err.')
plt.legend(['x (fxp)','x (float)','quant. err.'])
plt.show()

## 计算滤波器输出的定点数格式(考虑滤波器抽头的缩放TAP_SCALE)
nbits_y,mbits_y=find_fxp_fmt(y*TAP_SCALE,NUM_BITS)
print('[INF] np.max(abs(y)):',np.max(abs(y)))
print('[INF] filter output fmt.: S%d.%d'%(nbits_y,mbits_y))

## 模拟定点数运算(整数乘加)，输出小数有mbits_x+mbits_taps位
fxp_y_int=lfilter(fxp_taps_int, 1.0, fxp_x_int).astype(int)

## 模拟输出格式转换，定点数小数mbits_y位
# 注意：
#   输入x的定点化表示有mbits_x位小数
#   滤波器抽头经过放大后定点化表示有mbits_taps位小数
#   定点数直接计算输出有mbits_x+mbits_taps位小数
fxp_y_int>>=TAP_SHIFT                                               # 补偿滤波器权系数的放大
fxp_y_int>>=mbits_x+mbits_taps-mbits_y                              # 格式调整，满足mbits_y位小数 
fxp_y_int=np.clip(-2**(NUM_BITS-1),2**(NUM_BITS-1)-1,fxp_y_int)     # 模拟定点数受限位宽发生饱和

## 滤波结果转回浮点
fxp_y_f32=fxp_y_int.astype(float)/(2.0**mbits_y)                    # 转回浮点数的结果

# 显示量化误差对应的SNR
SNR=sum(y**2)/sum((fxp_y_f32-y)**2)
print('[INF] SNR:',10*np.log10(SNR),'dB')

plt.clf()   # 显示量化误差
plt.plot(fxp_y_f32,'b')
plt.plot(y,'r')
plt.plot(fxp_y_f32-y,'k')
plt.title('output quant. err.')
plt.legend(['y (fxp)','y (float)','quant. err.'])
plt.show()

## 生成C测试程序的代码并测试
if True:
    with open('fxp_filter_data.c','wt') as f:
        
        f.write('const int Y_SHIFT = %d;\n'%(TAP_SHIFT+mbits_x+mbits_taps-mbits_y))  # 输出的右移量
        f.write('const int NUM_TAPS = %d;\n'%len(taps))
        f.write('const int NUM_X = %d;\n'%len(x))
        f.write('const signed char x_int[] = \n{')
        for n,v in enumerate(fxp_x_int):
            if n%16==0: f.write('\n    ')
            f.write('%d, '%v)
        f.write('\n};\n\n')

        f.write('const signed char taps_int[] = \n{')
        for n,v in enumerate(fxp_taps_int):
            if n%16==0: f.write('\n    ')
            f.write('%d, '%v)
        f.write('\n};\n\n')

    # 编译并运行C测试程序
    if platform.system()=='Darwin': # MacOS
        os.system('clang -o test fxp_filter_data.c fxp_filter.c -DCLANG_MACOS -std=c99')
        os.system('./test')
    else:
        os.system('gcc -o test fxp_filter_data.c fxp_filter.c -std=c99')
        os.system('./test')
    
    # 读回C测试程序输出的定点数
    fxp_y_f32_c=np.fromfile('y_int.bin', dtype=np.int8).astype(float)/(2.0**mbits_y)

    plt.clf()   # 显示量化误差
    plt.plot(fxp_y_f32_c,'b')
    plt.plot(fxp_y_f32,'r')
    plt.plot(fxp_y_f32_c-fxp_y_f32,'k')
    plt.title('fxp comparison, C vs. python')
    plt.legend(['C output','fxp_y', 'err.'])
    plt.show()

