#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

####################
# 多常数乘法搜索算法
# 注意：只合成“正奇数“集合
#
# 代码清单 4-23
####################

###################
## 全局设置
###################

N=12            # 合成数据的bit宽度
GMAX=N          # 合成期间考虑的最大移位量
VMAX=1<<GMAX    # 合成的最大数

###################
# API
###################

## 从a和b生成所有形如pa*(a<<ga)+pb*(b<<gb)的元素s，返回它们的集合S
# 要求生成元素s满足：
#   1) 0<s<VMAX; 
#   2) s不在集合P内
# 注意：
#   输入a和b的次序无关，即
#   synthesize_value_all(a,b,P)==synthesize_value_all(b,a,P)
def synthesize_value_all(a,b,P):
    S=[]
    for ga in range(GMAX-int(math.log2(a))):
        for gb in range(GMAX-int(math.log2(b))):
            for pa,pb in [(1,1),(1,-1)]:
                s=abs(pa*(a<<ga)+pb*(b<<gb))
                if s in P: continue     # 排除集合P内的元素
                if 0<s<VMAX: S.append(s)# 排除0和过大元素
                    
    return list(set(S)) # 排除重复元素


## 搜索用a和b生成t的表达式，即：
#   t=pa*(a<<ga)+pb*(b<<gb)
# 输出：
#   生成t的代码字符串
#   ret_param==True时，额外输出(ga,gb,pa,pb,a,b)
# 注意：
#   这里仅仅根据搜索次序得到最早生成的那个
def find_equation(t=1,a=1,b=1,ret_param=False):
    # 最简单的情形
    if t==1: 
        equ='c1=x'
        return equ if not ret_param else (equ,(0,0,1,1,1,0))
    
    # 简单移位就能生成的情形
    if t in [a*(1<<g) for g in range(GMAX)]:
        g=int(round(np.log2(t/a))) 
        equ='c%d=c%d<<%d'%(t,a,g)
        return equ if not ret_param else (equ,(g,0,1,1,a,0))
    if t in [b*(1<<g) for g in range(GMAX)]:
        g=int(round(np.log2(t/b))) 
        equ='c%d=c%d<<%d'%(t,b,g)
        return equ if not ret_param else (equ,(0,g,1,1,0,b))

    # 遍历搜索
    for ga in range(GMAX-int(math.log2(a))):
        for gb in range(GMAX-int(math.log2(b))):
            for pa,pb in [(1,1),(1,-1)]:
                c=pa*(a<<ga)+pb*(b<<gb)
                if c<0: c,pa,pb=-c,-pa,-pb  # 负数的生成表达式转成正数
                if t==c:                    # 找到答案
                    equ_a=('(c%d<<%d)'%(a,ga)) if ga>0 else ('c%d'%a)
                    if pa<0: equ_a='-'+equ_a
                    equ_b=('(c%d<<%d)'%(b,gb)) if gb>0 else ('c%d'%b)
                    equ_b=('+'+equ_b) if pb>0 else ('-'+equ_b)
                    equ='c%d='%t+equ_a+equ_b
                    return equ if not ret_param else (equ,(ga,gb,pa,pb,a,b)) 
    print('[ERR] ********** cannot find equ. for %d by (%d,%d)!'%(t,a,b,))
    equ='c%d=?'%t
    return equ if not ret_param else (equ,tuple([None]*6)) 


## 用集合R内的元素生成R之外的新元素，结果保存在S中
def synthesize_from_set(R):
    S={}
    for a in R:
        for b in R:
            if b>a: continue    # 减少重复
            for s in synthesize_value_all(a,b,R):
                if s in S: continue
                S[s]=(a,b)
    return S

## 启发式地选择最接近T的s
def H(R,S,T):
    # 选最接近T的s
    diff=np.inf
    for s in S:
        for t in T:
            d=min([abs(s-t) for t in T])
            if d<diff: diff,s_sel=d,s
    return s_sel

## 验证列表内的表达式序列
def verify_equations(equ_str_list):
    x,flag=1,True
    for equ_str in equ_str_list:
        ver_equ_str=equ_str+';\nif %s!=%s: print("    error ********!");\n'%(equ_str[:equ_str.find('=')],equ_str[1:equ_str.find('=')])
        ver_equ_str+='flag=False if %s!=%s else flag\n'%(equ_str[:equ_str.find('=')],equ_str[1:equ_str.find('=')])
        try:
            exec(ver_equ_str)
        except:
            return False
    return flag
                

## 生成的表达式代码，并通过本地执行验证
# 输入
#   E   每个数的两个生成元
#   C   待输出表达式的数集 
# 输出
#   数据合成表达式列表
def gen_equations(E,R,verbose=False):
    # 根据R0中数据合成的“依赖顺序”，提取其中个元素加入C0列表
    C0=[1]
    R0=R.copy()
    R0.remove(1)
    while R0:
        for r in R0:
            a,b=E[r]
            if a in C0 and b in C0:
                C0.append(r)
                R0.remove(r)
                break
    print('[INF] C0(%d):'%len(C0),C0)
    
    # 生成所有表达式字符串
    equ_str_list=['c1=x' if c==1 else find_equation(c,*E[c])\
                    for n,c in enumerate(C0)]    # 存放所有生成代码
    if verbose: 
        for n,equ_str in enumerate(equ_str_list): 
            print('[INF]    %d: %s'%(n,equ_str))
        print()
    return equ_str_list

## MCM搜索
# 输出
#   R   中间数集
#   E   每个数对应的两个生成数
#   生成表达式字符串数组
def mcm_search(C,verbose=False):
    # 搜索期间的中间数据存储
    T=C.copy()  # 待合成的数集(集合内容在运行期间不断减少)
    R,W=[],[1]  # R保存每一轮合成计算中“有用”合成数据，W保存下一轮需要增补到R的数据
    E={}        # 存放合成参数

    # 核心搜索代码:
    #   从数字{1}开始，尝试合成T中的所有数据，
    #   R存放当前为止所有“有用”的合成数据
    # 执行步骤:
    #   1. 每轮搜索前，R用上一轮的W扩充
    #   2. 每轮搜索从R计算生成集S，
    #   3. 从S中找出目标集T中元素w，并从T移走(表明已成功合成了)，元素w会加入集合W (W是下一轮合成运算对集合R的增补)
    #   4. T未空则启发式地从S中挑选一个新增目标数加入W，返回步骤1
    while len(T)>0:
        while len(W)>0:
            R=set(list(R)+list(W))          # 上一轮新增的已合成数据融入R集合
            S=synthesize_from_set(R)        # 计算从R能够合成所有数据
            
            # 下面代码运行后得到W是R的扩充集，即下一轮循环新增的数
            W=[s for s in S if s in T]      # W = S和T交集，即“有用的”合成数据
            for w in W:                     # 从T移走已合成数据，加入E(合成完毕数据集)
                T.remove(w)
                if w not in E: E[w]=S[w]    # 保存合成参数
            
        if len(T)>0:                        # 还有待合成的数据?
            s=H(R,S,T)                      # 需要扩充目标数据集，启发式地从S中挑选目标集(T)之外的数s
            W.append(s)
            if s not in E: E[s]=S[s]        # 保存合成参数

    if verbose: 
        print('[INF] R(%d):'%len(R),R)
        print('[INF] E:',E)

    return R,E,gen_equations(E,R,verbose)

## 生成C语言测试代码
def gen_ccode(equ_str_list,fname='export_code/auto_code.c'):
    with open(fname,'wt') as f:
        f.write('#include <stdint.h>\n')
        f.write('#include <stdio.h>\n')
        f.write('\n')
        f.write('#include "../float_shift.h"\n')
        f.write('\n')
        f.write('void mcm(int32_t x, int32_t *output)\n')
        f.write('{\n')
        for equ_str in equ_str_list:
            f.write('    int32_t '+equ_str+';\n')
        f.write('\n')
        for n,equ_str in enumerate(equ_str_list):
            f.write('    output[%d]='%n+equ_str[:equ_str.find('=')]+';\n')
        f.write('}\n')
        f.write('\n')
        f.write('int main()\n')
        f.write('{\n')
        x=np.random.randint(-10000,10000)
        f.write('    int32_t x=%d;\n'%x)
        f.write('    int32_t output[%d];\n'%len(equ_str_list))
        f.write('    int32_t reference[%d]={'%len(equ_str_list))
        for equ_str in equ_str_list: f.write(equ_str[1:equ_str.find('=')]+', ')
        f.write('};\n')
        
        f.write('    printf("[INF] start of test\\n");\n')
        f.write('    mcm(x,output);\n')
        f.write('    for (int n=0; n<%d; n++)\n'%len(equ_str_list))
        f.write('        if (output[n]!=reference[n]*x)\n')
        f.write('            printf("[ERR] x*%d\\n",reference[n]);\n')
        f.write('        else\n')
        f.write('            printf("[INF] x*%d \tPASS\\n",reference[n]);\n')
        f.write('    printf("[INF] end of test\\n");\n')
        f.write('    return 0;\n')
        f.write('}\n')

####################
# 单元测试
####################
if __name__ == '__main__':
    import platform,os,sys
    # 设置当前运行目录
    os.chdir(sys.path[0])

    np.random.seed(1234)
    
    # MCM搜索并生成C程序的测试
    if True:
        C=[29,183,161,7,47] # 可选测试集 [29,183,161,7,47],[11,137,7],[89,101,3],[11,13,29,43],list(range(3,256,2)),list(range(3,32,2))
        R,E,equ_str_list=mcm_search(C,verbose=True)
        gen_ccode(equ_str_list)
    
    if False:
        TEST_NUM=100    # 测试循环轮次
        NUM     =20     # 测试数集尺寸
        
        # 自动测试
        equ_size=[]
        for it in range(TEST_NUM):
            # 生成随机测试集
            C=list(set(np.clip(0,VMAX-1,np.random.randint(3,VMAX/2,NUM)*2+1)))
            print('[INF] C(%d):'%len(C),C)  # C是待搜索数据集
            
            R,E,equ_str_list=mcm_search(C,verbose=True)
            print('[INF] R(%d):'%len(R),R)

            equ_size.append(len(R))
            if not verify_equations(equ_str_list):
                print('\n[ERR] **********')
                break

        print('[INF] average # equation:',np.mean(equ_size))    # 35.94

