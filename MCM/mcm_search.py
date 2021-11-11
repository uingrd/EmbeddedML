#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

####################
# 多常数乘法搜索算法
# 代码清单 4-23
####################

###################
## 全局设置
###################

N=8             # 合成数据的bit宽度
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
def synthesize_value_all(a,b,P=None):
    S=[]
    for ga in range(GMAX-int(math.log2(a))):
        for gb in range(GMAX-int(math.log2(b))):
            for pa,pb in [(1,1),(1,-1)]:
                s=pa*(a<<ga)+pb*(b<<gb)
                if P is not None:
                    if s in P: continue     # 排除集合P内的元素
                if abs(s) in S: continue    # 排除已生成过的元素
                if 0<abs(s)<VMAX:           # 排除0和过大元素
                    S.append(abs(s))
    return S


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
            for s in synthesize_value_all(a,b,R):
                if s in R: continue
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

## 本地执行并验证生成的表达式代码
# (动态生成python代码片段并执行)
def verify_equation_string(E,C0):
    exec('x=1') # 用于验证表达式正确性
    for n,c in enumerate(C0):
        equ_str='c1=x' if c==1 else find_equation(c,*E[c])
        print('[INF]    %d: %s'%(n,equ_str), end='')
        # 验证表达式正确性
        ver_equ_str='%s; print("    error ********!" if %s!=%s else "")'%(equ_str, equ_str[:equ_str.find('=')],equ_str[1:equ_str.find('=')])
        exec(ver_equ_str)

####################
# 单元测试
####################

# C是待搜索数据集
# C=[29,183,161,7,47]
# C=[11,137,7]
# C=[89,101,3]
# C=[11,13,29,43]
# C=[n for n in range(3,256) if n%2==1]
# C=[n for n in range(3,32) if n%2==1]

# 自动测试
for it in range(100):
    # 生成随机测试集
    C=[]
    for _ in range(20):
        v=np.random.randint(3,256,1).ravel()[0]
        if v%2==0: v+=1
        C.append(v)
    C=list(set(C))
    print('[INF] C(%d):'%len(C),C)  # C是待搜索数据集

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

    print('[INF] R(%d):'%len(R),R)
    if False: print('[INF] E:',E)

    # 输出计算表达式
    C0=[1]
    R0=R.copy()     # 所有合成目标(是原始地“目标数集”并上合成期间额外生成地“中间数集”)
    R0.remove(1)    
    # 根据R0中数据合成的“依赖顺序”，提取其中个元素加入C0列表
    while len(R0)>0:
        for r in R0:
            a,b=E[r]# r由a、b合成
            if a in C0 and b in C0: # 只有r所依赖的a和b已经生成，r才会被加入C0
                C0.append(r)
                R0.remove(r)
                break
    print('[INF] C0(%d):'%len(C0),C0)
    # C0中元素按次序被合成(考虑了合成次序的依赖性)

    # 依次合成C0中元素，打印表达式序列并验证其正确性
    verify_equation_string(E,C0)

# 下面自动生成图片，需要安装graphviz才能运行
if False:
    from graphviz import Digraph
    dot = Digraph(comment='MCM graph')
    for c in C0:
        dot.node(name='v'+str(c),label=str(c))

    for c in C0:
        if c==1: continue
        _,(ga,gb,pa,pb,a,b)=find_equation(c,*E[c],ret_param=True)
        if a>0:
            e=str(1<<ga) if pa>0 else ('-'+str(1<<ga))
            dot.edge('v'+str(a), 'v'+str(c), e)
        if b>0:
            e=str(1<<gb) if pb>0 else ('-'+str(1<<gb))
            dot.edge('v'+str(b), 'v'+str(c), e)

    dot.view()
    dot.render('MCM-graph.gv', view=True)

    # dot MCM-graph.gv  -Kcirco -Tpdf -o img.pdf
    # dot MCM-graph.gv  -Kneato -Tpdf -o img.pdf
    # dot MCM-graph.gv  -Kdot -Tpdf -o img.pdf
    # dot MCM-graph.gv  -Ktwopi -Tpdf -o img.pdf
    # dot MCM-graph.gv  -Kfdp -Tpdf -o img.pdf
    # dot MCM-graph.gv  -Ksfdp -Tpdf -o img.pdf
    
