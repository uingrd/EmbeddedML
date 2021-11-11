import math
import numpy as np
import matplotlib.pyplot as plt

########################
# 几种近似tanh计算方法
# 代码清单 2-12
########################

## 近似算法
def tanh_1(x):
    if x < -3: return -1
    if x >  3: return  1
    return x*(27+x*x)/(27+9*x*x )

## 近似算法
def tanh_2(x):
    if x> 3.4: return  1
    if x<-3.4: return -1
    
    x/=3.4
    x = (abs(x)-2.0)*x
    return (abs(x)-2.0)*x

## 近似算法
def tanh_3(x,a=3.4,b=2.0,c=2.0):
    if x> a: return  1
    if x<-a: return -1
    
    x/=a
    x = (abs(x)-b)*x
    return (abs(x)-c)*x

# 近似算法的参数自动搜索
def find_opt_param():

    xv=np.linspace(-3.5,3.5,10000)
    y =np.tanh(xv)
    
    def cost_func(param):
        nonlocal xv,y
        y0=np.array([tanh_3(x,*param) for x in xv])
        return np.max(np.abs(y-y0))
    
    init=[3.4,2,2]
    from scipy.optimize import minimize
    opt=minimize(cost_func, 
                 x0=init, 
                 bounds=((init[0]*0.9,init[0]*1.1),
                         (init[1]*0.9,init[1]*1.1),
                         (init[2]*0.9,init[2]*1.1)), 
                 method='powell',#'Nelder-Mead',#, 
                 options={'maxiter':10000})
    print('[INF] opt:',opt.x)
    print('[INF]    before:',cost_func((3.4,2.0,2.0)))
    print('[INF]     after:',cost_func(opt.x))
    return opt.x

####################
# 单元测试
####################
if __name__ == '__main__':
    xv=np.linspace(-3.5,3.5,80000)
    y = [math.tanh(x) for x in xv]

    if True:
        y0= [tanh_1(x) for x in xv]
        plt.plot(xv, y ,'b')
        plt.plot(xv, y0,'r')
        plt.legend(['approx. (tanh_1)','reference'])
        plt.grid(True)
        plt.show()

        plt.plot(xv, np.array(y)-np.array(y0),'k')
        plt.title('error (tanh_1)')
        plt.grid(True)
        plt.show()

    if True:
        y0= [tanh_2(x) for x in xv]
        plt.plot(xv, y ,'b')
        plt.plot(xv, y0,'r')
        plt.legend(['approx. (tanh_2)','reference'])
        plt.grid(True)
        plt.show()

        plt.plot(xv, np.array(y)-np.array(y0),'k')
        plt.title('error (tanh_2)')
        plt.grid(True)
        plt.show()

    # 基于最小误差搜索近似算法参数
    if True:
        opt_param=find_opt_param()        
        y0= [tanh_3(x,*opt_param) for x in xv]
        plt.plot(xv, y ,'b')
        plt.plot(xv, y0,'r')
        plt.legend(['approx. (tanh_3)','reference'])
        plt.grid(True)
        plt.show()

        plt.plot(xv, np.array(y)-np.array(y0),'k')
        plt.title('error (tanh_3)')
        plt.grid(True)
        plt.show()

