#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

####################
# 单常数乘法的搜索算法
# 代码清单 4-22
####################

N=10             # 合成数据的bit宽度
GMAX=N          # 合成期间考虑的最大移位量
VMAX=1<<GMAX    # 合成的最大数


# 从a和b生成元素s=pa*(a<<ga)+b*(b<<gb)
def synthesize_value(ga,gb,pa,pb,a,b):
    return pa*(a<<ga)+pb*(b<<gb)


# 从a和b生成所有形如pa*(a<<ga)+b*(b<<gb)的元素，返回它们的集合
# 要求生成的值满足：
#   为正
#   小于VMAX
#   不在集合P内
# 注意：
#   输入a和b的次序无关，即
#   synthesize_value_all(a,b,P)==synthesize_value_all(b,a,P)
def synthesize_value_all(a,b,P):
    S=[]
    for ga in range(GMAX-int(math.log2(a))):
        for gb in range(GMAX-int(math.log2(b))):
            for pa,pb in [(1,1),(1,-1)]:
                s=synthesize_value(ga,gb,pa,pb,a,b) # 合成元素s=pa*(a<<ga)+pb*(b<<gb)

                if s in P: continue                 # 排除集合P内的元素
                if abs(s) in S: continue            # 排除已生成过的元素
                if s==0 or abs(s)>=VMAX: continue   # 排除0和过大元素
                
                S.append(abs(s))
    return S


# 计算集合A、B和C的并集
def merge_set(A,B,C=()):
    return set(list(A)+list(B)+list(C))


# 得到生成t的两个数a和b(有可能a==b)
def get_parent(t,T):
    _,ab=T[t]
    return ab if len(ab)==2 else tuple(list(ab)*2)


# 计算用a和b生成t的表达式，输出字符串
# 注意：这里仅仅根据搜索次序得到最早生成的那个
def find_equation(t,a,b,ret_param=False):
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
                c=synthesize_value(ga,gb,pa,pb,a,b)   # =pa*(a<<ga)+b*(b<<gb)
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


# 得到T中所有元素的生成表达式(字符串)
def find_all_equations(T): 
    equ_all=['']
    for t in range(1,1<<N):
        a,b=get_parent(t,T)
        equ_all.append(find_equation(t,a,b))
    return equ_all


# 显示数据t的所有生成表达式(包括其相关元素)
#   T: 提供元素t的依赖关系
#   P: 保存已经生成了元素
#   equ_all: 预先生成的T内每个元素的表达式
#   ret:    保存所有生成表达式
def find_equations_t(t,T,equ_all=None,P=None,ret=None):
    if equ_all is None: equ_all=find_all_equations(T)
    if P       is None: P=[]
    if ret     is None: ret=[]
    
    # 递归，先生成t的父节点元素的表达式，在生成t的表达式
    if t not in P:
        if t==1: 
            ret.append(equ_all[1])
        else:
            a,b=get_parent(t,T)
            find_equations_t(a,T,equ_all,P,ret); P.append(a)
            find_equations_t(b,T,equ_all,P,ret); P.append(b)
            ret.append(equ_all[t])
            P.append(t)
    return ret


# 用A、B内的元素生成P集合外的新元素，结果保存在T中
def synthesize_from_set(A,B,P,T,check_order=False):
    print('[INF] syn_from_set()')
    print('[INF]    len(A):',len(A))
    print('[INF]    len(B):',len(B))
    print('[INF]    len(P):',len(P))
    for a in A:
        for b in B:
            if check_order and a>b: continue    # 用于A==B的情况下减少重复
            S=synthesize_value_all(a,b,P)
            for s in S:
                # 得到c的生成集c_set
                a_set,b_set=T[a][0],T[b][0]
                s_set=merge_set(a_set,b_set,[a,b])

                if len(T[s][0])==0:     # 第一次合成s，记录其生成集
                    T[s]=(s_set,(a,b))
                elif len(s_set)<len(T[s][0]):
                    T[s]=(s_set,(a,b))  # s之前合成过，保存最小的生成集
    return


# 找出T中生成集尺寸为n的元素
def find_set_by_size(n,T,P=None):
    return [t for t,(t_set,_) in enumerate(T) if len(t_set)==n] if P is None else\
           [t for t,(t_set,_) in enumerate(T) if len(t_set)==n and t not in P]


# 验证等式正确性
def verify_t(t,equ_all,T):
    equ_t_all=find_equations_t(t,T,equ_all)
    if False: print('[INF] %d, len(equ_t_all):'%t,len(equ_t_all))
    equ='x=1;'
    for e in equ_t_all:
        equ+=e+';'
    equ+='print(["[ERR] %d","."][c%d==%d],end="")'%(t,t,t)
    if False: print(equ)
    exec(equ)
    return equ_t_all


####################
# 入口
####################

# 初始化
T=[((),()) for n in range(1<<N)]    # T[t]=(s_set,(a,b)), t=syn(a,b)
F=[]                                # 存放生成模式固定了的数

# 生成集大小为1的数据
for a in 2**np.arange(N):
    T[a]=(set([1]),set([1]))
    F.append(a)                     # 数据1<<n的生成模式固定

# (F,F)内元素进行合成，输出保存在T中(得到最多1次加减运算能生成的数)
synthesize_from_set(F,F,F,T,check_order=True)

# 逐一生成不同尺寸生成集的数据
k=1
while True:
    print('[INF] k: %d, len(F):%d'%(k,len(F)))
    W=find_set_by_size(k,T,F)
    P=merge_set(F,W)
    if len(P)>=(1<<N)-1:
        F=P
        break
    synthesize_from_set(F,W,P,T,check_order=False) # 合成新元素
    synthesize_from_set(W,W,P,T,check_order=True )
    F=P
    k+=1

if False:
    print(T)
    print('[INF] len(F):',len(F))

# 生成T中所有元素的生成表达式
equ_all=find_all_equations(T)
if True:
    for e in equ_all:
        print('[INF] '+e)

# 打印每个数的生成表达式集
if True:
    for t in range(1,1<<N):
        print('[INF] equations for',t)
        for e in find_equations_t(t,T,equ_all):
            print('[INF]    ',e)

# 验证
if True:
    for t in range(1,1<<N):
        verify_t(t,equ_all,T)
    print()
        
# 统计所有数据中，最多生成表达式数量(扣除了表达式'V1=1'计数)
if True:
    max_equ_num=max([len(find_equations_t(t,T,equ_all))-1 for t in range(1,1<<N)])
    print('[INF] max_equ_num:',max_equ_num)
    
# 统计所有数据中，最大生成集的尺寸
if True:
    max_syn_set_size=max([len(t_set) for t_set,_ in T])
    print('[INF] max_syn_set_size:',max_syn_set_size)

# 生成计算图文件，注意，需要安装python的graphviz包
if False:
    from graphviz import Digraph
    dot = Digraph(comment='SCM graph')
    for t in range(1,VMAX):
        dot.node(name='v'+str(t),label=str(t))
        
    for t in range(2,VMAX):
        a,b=get_parent(t,T)
        _,(ga,gb,pa,pb,a,b)=find_equation(t,a,b,ret_param=True)
        if a>0:
            e=str(1<<ga) if pa>0 else ('-'+str(1<<ga))
            dot.edge('v'+str(a), 'v'+str(t), e)
        if b>0:
            e=str(1<<gb) if pb>0 else ('-'+str(1<<gb))
            dot.edge('v'+str(b), 'v'+str(t), e)

    dot.view()
    dot.render('SCM-graph.gv', view=True)

    # dot SCM-graph.gv  -Kcirco -Tpdf -o img.pdf
    # dot SCM-graph.gv  -Kneato -Tpdf -o img.pdf
    # dot SCM-graph.gv  -Kdot -Tpdf -o img.pdf
    # dot SCM-graph.gv  -Ktwopi -Tpdf -o img.pdf
    # dot SCM-graph.gv  -Kfdp -Tpdf -o img.pdf
    # dot SCM-graph.gv  -Ksfdp -Tpdf -o img.pdf
