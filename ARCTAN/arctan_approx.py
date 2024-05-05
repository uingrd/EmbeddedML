import numpy as np
import matplotlib.pyplot as plt

########################
# 近似arctan2计算方法
########################

def algo1(i,q): return i*q/(i**2+0.28125*q**2)
def algo2(i,q): return i*q/(i**2+0.28086*q**2)
def algo3(i,q): 
    x=q/i
    return np.pi*0.25*x+0.285*x*(1-np.abs(x))
def algo4(i,q): 
    x=q/i
    return np.pi*0.25*x+0.273*x*(1-np.abs(x))
def algo5(i,q): 
    x=q/i
    return np.pi*0.25*x+x*(0.186982-0.191942*x**2)
def algo6(i,q): 
    x=q/i
    return np.pi*0.25*x-x*(np.abs(x)-1)*(0.2447+0.0663*np.abs(x))
    
def arctan2(i,q,algo):
    if np.abs(i)>np.abs(q): 
        return algo(i,q) if i>0 else np.sign(q)*np.pi+algo(i,q)
    else:
        return 0.5*np.pi-algo(q,i) if q>0 else -0.5*np.pi-algo(q,i)
    

####################
# 单元测试
####################
if __name__ == '__main__':
    N=10000
    t=np.linspace(-np.pi,np.pi,N+1)[:-1]

    t1=np.array([arctan2(i,q,algo1) for i,q in zip(np.cos(t),np.sin(t))])
    t2=np.array([arctan2(i,q,algo2) for i,q in zip(np.cos(t),np.sin(t))])
    t3=np.array([arctan2(i,q,algo3) for i,q in zip(np.cos(t),np.sin(t))])
    t4=np.array([arctan2(i,q,algo4) for i,q in zip(np.cos(t),np.sin(t))])
    t5=np.array([arctan2(i,q,algo5) for i,q in zip(np.cos(t),np.sin(t))])
    t6=np.array([arctan2(i,q,algo6) for i,q in zip(np.cos(t),np.sin(t))])

    print('algo1 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t1))),np.max(np.abs(np.rad2deg(t-t1)))))
    print('algo2 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t2))),np.max(np.abs(np.rad2deg(t-t2)))))
    print('algo3 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t3))),np.max(np.abs(np.rad2deg(t-t3)))))
    print('algo4 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t4))),np.max(np.abs(np.rad2deg(t-t4)))))
    print('algo5 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t5))),np.max(np.abs(np.rad2deg(t-t5)))))
    print('algo6 err. %f (max: %f)'%(np.mean(np.abs(np.rad2deg(t-t6))),np.max(np.abs(np.rad2deg(t-t6)))))
    
    plt.clf()
    plt.plot(np.rad2deg(t),np.rad2deg(t-t1))
    plt.plot(np.rad2deg(t),np.rad2deg(t-t2))
    plt.plot(np.rad2deg(t),np.rad2deg(t-t3))
    plt.plot(np.rad2deg(t),np.rad2deg(t-t4))
    plt.plot(np.rad2deg(t),np.rad2deg(t-t5))
    plt.plot(np.rad2deg(t),np.rad2deg(t-t6))

    plt.legend(['err. algo. 1','err. algo. 2','err. algo. 3','err. algo. 4','err. algo. 5','err. algo. 6'])
    plt.grid(True)
    plt.show()
