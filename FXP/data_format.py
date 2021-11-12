#!/usr/bin/python3
# coding=utf-8

import numpy as np
import struct,math

########################################
## float16/float32/float64和hex的格式转换
########################################

def to_fp16(x): return np.float16(x)
def to_fp32(x): return np.float32(x)
def to_fp64(x): return np.float64(x)

def fp16_to_bytes(x):   return x.tobytes()
def fp32_to_bytes(x):   return x.tobytes()
def fp64_to_bytes(x):   return x.tobytes()

def fp16_to_hex4_str (x): return fp16_to_bytes(x).hex()
def fp32_to_hex8_str (x): return fp32_to_bytes(x).hex()
def fp64_to_hex16_str(x): return fp64_to_bytes(x).hex()

def bytes_to_fp16(b): return np.fromstring(b, dtype=np.float16)[0]
def bytes_to_fp32(b): return np.fromstring(b, dtype=np.float32)[0]
def bytes_to_fp64(b): return np.fromstring(b, dtype=np.float64)[0]

def hex_str_to_bytes(s,n): return bytes([eval('0x'+s[n*2:n*2+2]) for n in range(n)])
def hex4_str_to_bytes (s): return hex_str_to_bytes(s,2)
def hex8_str_to_bytes (s): return hex_str_to_bytes(s,4)
def hex16_str_to_bytes(s): return hex_str_to_bytes(s,8)

def hex4_str_to_fp16 (s): return bytes_to_fp16(hex4_str_to_bytes (s))
def hex8_str_to_fp32 (s): return bytes_to_fp32(hex8_str_to_bytes (s))
def hex16_str_to_fp64(s): return bytes_to_fp64(hex16_str_to_bytes(s))

## 直接定点数转换
def double_to_fxp(v,n=0,m=15):
        i=round(v*2.**m)
        i=min( 2**(n+m)-1,i)
        i=max(-2**(n+m)  ,i)
        return i/2.**m

## 有符号整数和二进制之间的格式转换
def int_to_bin(v,n=None):
    if n is None: n=64
    return bin((1<<n)+v)[2:].zfill(n) if v<0 else bin(v)[2:].zfill(n)
def int8_to_bin (v): return int_to_bin(v,8)
def int16_to_bin(v): return int_to_bin(v,16)
def int32_to_bin(v): return int_to_bin(v,32)
def int64_to_bin(v): return int_to_bin(v,64)

## 二进制bit转成整数，输入字串是如下格式：
# 0b1000_11_0_11000
# 字串开头的0b可以不加，另外字串中的_被自动忽略
def bin_to_int(s,n=None): 
    s=s.replace('0b','').replace('_','')
    if n is None: n=len(s)
    return int(s,2) if s[0]=='0' else int(s,2)-(1<<n)
def bin_to_int8 (s): return bin_to_int(s,8 )
def bin_to_int16(s): return bin_to_int(s,16)
def bin_to_int32(s): return bin_to_int(s,32)
def bin_to_int64(s): return bin_to_int(s,64)

def print_bin(v,n=8):
    s=int_to_bin(v,n)
    num=len(s)
    for n,c in zip(range(num),s):
        if n>0 and (num-n)%4==0: print(' ',end='')
        print(c,end='')
    print('')
    return s
    
####################
# 定点数类
class fxp_c:
    def __init__(self, v=0, n=0, m=15):
        self.set_param(n,m)     # 格式相关的参数设置
        self.set(v)             # 值设置
    
    def set_param(self,n,m):
        self.n=n                # 整数位宽（不考虑符号位）
        self.m=m                # 小数位宽
        self.k=m+n+1            # 总位宽
        
        self.imax= 2**(m+n)-1   # 最大可能值
        self.imin=-2**(m+n)     # 最小可能值
        self.scale=2**self.m    # 小数对应的倍乘因子
    
    # 修改小数位数
    def change_m(self,m):
        if m==self.m: return self.v
        
        if m>self.m:
            i=self.i*2**(m-self.m)  # 左移  
        else:
            i=self.i+2**(self.m-m-1)# 舍入
            i//=2**(self.m-m)       # 右移
        self.set_param(self.n,m)    # 设置新的格式参数
        return self.seti(i)
    
    # 修改整数位数
    def change_n(self,n):
        if n==self.n: return self.v
        
        self.set_param(n,self.m)    # 设置新的格式参数
        return self.seti(self.i)    # 考虑可能溢出，需要重新设置
        
    # 以整数形式设置内容
    def seti(self,i,sat=True):
        self.i=min(self.imax,max(i,self.imin)) if sat else i  # 饱和
        if self.i != i: print('saturation in seti()')
            
        self.v=float(self.i)/float(self.scale) # 计算值
        return self

    # 以bit串形式设置内容
    def setb(self,s): return self.seti(bin_to_int(s,self.k))
    
    # 设置实数值
    def set(self,v): return self.seti(int(round(v*float(self.scale))))
    
    # 以bit串形式返回内容
    def to_bits(self,has_point=True): 
        s=int_to_bin(self.i,self.k)
        return s[:self.n+1]+'.'+s[self.n+1:] if has_point else s

    # 打印信息
    def dump(self):
        print('val:',self.v)
        print('int:',self.i)
        print('bit:',self.to_bits())
        print('  m:',self.m)
        print('  n:',self.n)
        print('  k:',self.k)

    def check(self):
        if self.i> 2**(self.k-1)-1      : print('error')
        if self.i<-2**(self.k-1)        : print('error')
        if self.k!=self.m+self.n+1      : print('error')
        if self.v*(2.**self.m)-float(self.i) > 0.5*2.**(-self.m): print('error')
        if self.scale!=2**self.m        : print('error')
        if self.imax!=2**(self.k-1)-1   : print('error')
        if self.imin!=-2**(self.k-1)    : print('error')


## 定点数复制
def fxp_copy(s): return fxp_c(s.v,s.n,s.m)
    
## 定点数格式转换
def fxp_conv(s,n,m):
    s_new=fxp_copy(s)
    s_new.change_n(n)
    s_new.change_m(m)
    return s_new

## 定点数相加
def fxp_add(s1,s2):
    # 获取格式参数
    n,m = max(s1.n,s2.n), max(s1.m,s2.m)
    
    s1_new=fxp_conv(s1,n,m)
    s2_new=fxp_conv(s2,n,m)
    s1_new.seti(s1_new.i+s2_new.i)
    return s1_new

## 定点数减
def fxp_sub(s1,s2):
    # 获取格式参数
    n,m = max(s1.n,s2.n), max(s1.m,s2.m)
    
    s1_new=fxp_conv(s1,n,m)
    s2_new=fxp_conv(s2,n,m)
    s1_new.seti(s1_new.i-s2_new.i)
    return s1_new

## 定点数乘
def fxp_mul(s1,s2):
    # 获取格式参数
    k = s1.k+s2.k
    m = s1.m+s2.m
    return fxp_c(s1.i*s2.i,k-m-1,m)


####################
# 单元测试
####################

if __name__=='__main__':
    ## 测试float16/float32/float64格和hex的格式转换
    if True:
        print()
        x=to_fp16(-30.875)
        print('fp16 value: %.4f'% x)
        print('hex:', fp16_to_hex4_str(x))
        print('%.4f'%hex4_str_to_fp16(fp16_to_hex4_str(x)))

        x=to_fp32(-24.75)
        print('fp32 value:', x)
        print('hex:', fp32_to_hex8_str(x))
        print(hex8_str_to_fp32(fp32_to_hex8_str(x)))

        x=to_fp64(3.14)
        print('fp64 value:', x)
        print('hex:', fp64_to_hex16_str(x))
        print(hex16_str_to_fp64(fp64_to_hex16_str(x)))

    if False:
        import tensorflow as tf
        import math

        g=tf.Graph()
        with g.as_default():
            with tf.Session(graph=g) as sess:
                pi_bfloat16 = sess.run(tf.to_bfloat16(math.pi))
                print()
                print('pi_bfloat16:',pi_bfloat16)
                print('err:',pi_bfloat16.astype(np.float32)-np.float32(math.pi))

    if True:
        print()
        print(bin_to_int('11111111',8))
        print(int_to_bin(bin_to_int('11111111',8),8))

        print(bin_to_int('11111110',8))
        print(int_to_bin(bin_to_int('11111110',8),8))

        print(bin_to_int('10000000',8))
        print(int_to_bin(bin_to_int('10000000',8),8))

        print(bin_to_int('01111111',8))
        print(int_to_bin(bin_to_int('01111111',8),8))

        print(bin_to_int('00000001',8))
        print(int_to_bin(bin_to_int('00000001',8),8))

        print(bin_to_int('00000000',8))
        print(int_to_bin(bin_to_int('00000000',8),8))

    ########################################
    ## 双精度浮点数和long类型之间转换
    ########################################
    long2bits   = lambda L: ("".join([str(int(1 << i & L > 0)) for i in range(64)]))[::-1]
    double2long = lambda d: struct.unpack("Q",struct.pack("d",d))[0]
    double2bits = lambda d: long2bits(double2long(d))
    long2double = lambda L: struct.unpack('d',struct.pack('Q',L))[0]
    bits2double = lambda b: long2double(bits2long(b))
    bits2long   = lambda z:sum([bool(z[i] == '1')*2**(len(z)-i-1) for i in range(len(z))[::-1]])

    ## 测试
    if True:
        print()
        print(math.pi)
        print('to long:', double2long(math.pi))
        print('back to double:', long2double(double2long(math.pi)))

        print('to bits:', double2bits(math.pi))
        print('back to double:', bits2double(double2bits(math.pi)))

        print('double->long->bits->long->double:', long2double(bits2long(long2bits(double2long(math.pi)))))


    ####################
    # 打印定点数范围
    if True:
        print()
        K=8
        for n in range(K):
            m=K-1-n
            step=1./2.**m
            imax=2**(K-1)-1
            imin=-imax-1
            vmax,vmin=imax*step,imin*step
            print('S%d.%d: '%(n,m))
            print('    step: %.8f'%step)
            print('    vmax: %.8f'%vmax)
            print('    vmin: %.8f'%vmin)
            print('S%d.%d, '%(n,m), end='')
            #print('%.8f, %.8f, %.8f'%(vmin,vmax,step))

    if True:
        print()
        print(math.pi)
        v10_5_pi=double_to_fxp(math.pi,10,5)
        print(v10_5_pi)
        print(v10_5_pi-math.pi)

        print()
        print(math.e)
        v10_5_e=double_to_fxp(math.e,10,5)
        print(v10_5_e)
        print(v10_5_e-math.e)

    ##################
    # 测试定点运算
    if True:
        s10_5_pi=fxp_c(math.pi,10,5)
        print()
        print(s10_5_pi.v)
        print(s10_5_pi.to_bits())

        s10_5_e=fxp_c(math.e,10,5)
        print()
        print(s10_5_e.v)
        print(s10_5_e.to_bits())

        s10_5_e_pi=fxp_c(0,10,5)
        i=s10_5_e.i-s10_5_pi.i
        s10_5_e_pi.seti(i)
        print()
        print('s10_5_e.i:',s10_5_e.i)
        print('s10_5_e.to_bits():',s10_5_e.to_bits())
        print('s10_5_e.v:',s10_5_e.v)

        print()
        print('s10_5_pi.i:',s10_5_pi.i)
        print('s10_5_pi.to_bits():',s10_5_pi.to_bits())
        print('s10_5_pi.v:',s10_5_pi.v)

        print()
        print('s10_5_e_pi.i:',s10_5_e_pi.i)
        print('s10_5_e_pi.to_bits():',s10_5_e_pi.to_bits())
        print('s10_5_e_pi.v:',s10_5_e_pi.v)

        print()
        print(s10_5_e.v-s10_5_pi.v)
        print(s10_5_e.v-s10_5_pi.v-s10_5_e_pi.v)
        print(math.e-math.pi)


        s5_10_pi=fxp_c(3.15625,5,10)
        print()
        print('s5_10_pi.v:',s5_10_pi.v)
        print('s5_10_pi.i:'    ,s5_10_pi.i)
        print('s5_10_pi.to_bits():',s5_10_pi.to_bits())


        print()
        s=fxp_c(20.25,4,3)
        s.dump()

    ##################
    # 测试C++程序下的定点运算


    print('\n----- test1 -----')
    a=bin_to_int16("00000000010_10111")
    b=bin_to_int16("00000000011_00101")

    print(a,b)
    print(hex(a),hex(b))

    res=a-b
    print(res/32.)
    print_bin(res,16)
    print('\n-----test 2 -----')

    a=bin_to_int16("00000000010_10111")
    b=bin_to_int16("11110_01101100000")
    c=b>>5

    print('a:',a)
    print_bin(a,16)
    print(hex(a))

    print('b:',b)
    print_bin(b,16)
    print(hex(b))

    print('c:',c)
    print_bin(c,16)
    print(hex(c))

    d=a+c
    print('d:',d)
    print_bin(d,16)
    print(hex(b))

    #####################
    if False:
        import os, platform

        if(platform.system()=='Windows'):  
            os.system('del /Q test_fxp.elf')
            os.system('gcc test_fxp.cpp -o test_fxp.elf')
            print('------ running test_fxp.elf -----')
            os.system('test_fxp.elf')
        else:        
            os.system('rm test.elf')
            os.system('gcc test_fxp.cpp -o test_fxp.elf')
            print('------ running test_fxp.elf -----')
            os.system('./test_fxp.elf')

