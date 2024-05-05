import numpy as np
import matplotlib.pyplot as plt

########################
# 近似sin计算方法
########################

def algo1(x): 
    y=x/np.pi
    return 4*y*(1-y)

def algo2(x): 
    if True:
        return -0.417698*x*x+1.312236*x-0.050465
    else:
        return (60*(np.pi**2-12)/np.pi**5)*x**2-\
               (60*(np.pi**2-12)/np.pi**4)*x+\
               (12*(np.pi**2-10)/np.pi**3)\

def algo3(x):
    return 16*x*(np.pi-x)/(5*np.pi**2-4*x*(np.pi-x))
    
def sin_approx(x,algo):
    x%=np.pi*2.0
    if x>np.pi:x-=np.pi*2.0
    return np.sign(x)*algo(np.abs(x))
    

####################
# 单元测试
####################
if __name__ == '__main__':
    N=10000
    x=np.linspace(-np.pi,np.pi,N+1)[:-1]
    s=np.sin(x)

    s1=np.array([sin_approx(x_,algo1) for x_ in x])
    s2=np.array([sin_approx(x_,algo2) for x_ in x])
    s3=np.array([sin_approx(x_,algo3) for x_ in x])
    
    print('algo1 err. %f (max: %f)'%(np.mean(np.abs(s-s1)),np.max(np.abs(s-s1))))
    print('algo2 err. %f (max: %f)'%(np.mean(np.abs(s-s2)),np.max(np.abs(s-s2))))
    print('algo3 err. %f (max: %f)'%(np.mean(np.abs(s-s3)),np.max(np.abs(s-s3))))
    
    plt.clf()
    plt.plot(s-s1)
    plt.plot(s-s2)
    plt.plot(s-s3)
    plt.legend(['algo. 1 err.', 'algo. 2 err.','algo. 3 err.'])
    plt.grid(True)
    plt.show()
