#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 生成二进制数的CSD表示
####################

## 将二进制字串(整数，无'0b'前缀)转成字典
# 代码清单 4-20
def bin_str_to_dict(bv): return {n:int(v=='1') for n,v in enumerate(bv[::-1])}

## 将二进制字串(整数)转成CSD表示
def bin_str_to_csd(bv):    
    w=len(bv)
    b=bin_str_to_dict(bv)
    b[-1]=0
    b[w]=b[w-1]
    gamma={-1:0}
    a={}
    theta={}
    for i in range(w):
        theta[i]=b[i]^b[i-1]
        gamma[i]=(1-gamma[i-1])*theta[i]
        a[i]=(1-2*b[i+1])*gamma[i]
    return a

## 将16-bit整数表示为二进制字串，用补码表示负数
def int16_to_bin_str(v):
    if v<0: v+=65536
    bv=bin(v)[2:]
    return '0'*(16-len(bv))+bv

## 将16-bit有符号整数转成CSD表示
def int16_to_csd(v):
    bv=int16_to_bin_str(v)
    csd=bin_str_to_csd(bv)
    return csd

## 将字典转回16-bit整数
def dict_to_int16(d): 
    v=sum([(2**k)*v for k,v in d.items()])
    if v>32767: v-=65536
    return v 

## 将字典内容打印成二进制字串
def dict_to_str(d):
    s=''
    for n in range(max(d.keys())+1):
        if n in d:
            s={-1:'n',0:'0',1:'1'}[d[n]]+s
        else:
            s='x'+s
    return s[::-1]

## 将CSD内容转成移位运算指令字符串
# 代码清单 4-21
def csd_to_code(csd):
    s=''
    for n,v in csd.items():
        if v==0: 
            continue
        elif n==0:
            s+='+x' if v>0 else '-x'
        else:
            s+='+(x<<%d)'%n if v>0 else '-(x<<%d)'%n
    return  s[1:] if s[0]=='+' else s

## 将16-bit有符号整数转成移位运算指令字符串
def int16_to_code(v):
    return csd_to_code(bin_str_to_csd(int16_to_bin_str(v)))
    
####################
# 单元测试
####################
if __name__=='__main__':
    print('\n[INF] ---- test 1 ----')
    bv=int16_to_bin_str(-141)
    print('[INF] bv:\n    ',bv)

    d=bin_str_to_dict(bv)
    print('[INF] d=bin_str_to_dict(bv):\n    ',d)

    v=dict_to_int16(d)
    print('[INF] v=dict_to_int16(d):\n    ',v)

    csd=bin_str_to_csd(bv)
    print('[INF] csd                :\n    ',csd                )
    print('[INF] dict_to_str(csd)   :\n    ',dict_to_str(csd)   )
    print('[INF] dict_to_int16(csd) :\n    ',dict_to_int16(csd) )
    print('[INF] csd_to_code(csd)   :\n    ',csd_to_code(csd)   )
    print('[INF] int16_to_code(-141):\n    ',int16_to_code(-141))
    

    # test2
    print('\n[INF] ---- test 2 ----')
    bv=int16_to_bin_str(141)
    print('[INF] bv:\n    ',bv)

    d=bin_str_to_dict(bv)
    print('[INF] d=bin_str_to_dict(bv):\n    ',d)

    v=dict_to_int16(d)
    print('[INF] v=dict_to_int16(d):\n    ',v)

    csd=bin_str_to_csd(bv)
    print('[INF] csd               :\n    ',csd               )
    print('[INF] dict_to_str(csd)  :\n    ',dict_to_str(csd)  )
    print('[INF] dict_to_int16(csd):\n    ',dict_to_int16(csd))
    print('[INF] csd_to_code(csd)  :\n    ',csd_to_code(csd)  )
    print('[INF] int16_to_code(141):\n    ',int16_to_code(141))

    # test3
    print('\n[INF] ---- test 3 ----')
    for v in range(-32768,32768):
        csd=int16_to_csd(v)
        v1=dict_to_int16(csd)
        if v%10000==0:
            print('[INF] %d'%v)
        if v!=v1: 
            print('[ERR] %d,%d'%(v,v1))
            print('[ERR]    ',bv)
            print('[ERR]    ',dict_to_str(csd))
            
