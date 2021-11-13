import IPython
import numpy as np

###################
# 循环卷积计算
#
# conv_1D_slow      对应式5-17
# conv_1D_DFT       对应式5-21
# conv_1D_numpy_FFT 对应代码清单5-1，5-2  (变量名有更新)
# conv_2D_slow      对应式5-12
# conv_2D_numpy_FFT 对应代码清单5-19      (变量名有更新)
# conv_2D_DFT       对应式5-67
# 
###################

# 按定义直接计算循环卷积
def conv_1D_slow(x,h):
    N,M=len(x),len(h)
    y=np.array([sum([x[(n-m)%N]*h[m] for m in range(M)]) for n in range(N)])
    return y

# 用numpy的线性卷积计算循环卷积
def conv_1D_numpy(x,h):
    N,M=len(x),len(h)
    if N!=M:
        print('[ERR] x/h size not match!')
        return None

    # 通过序列延拓用numpy的线性卷积计算循环卷积
    y=np.convolve(np.tile(x,2), h)[N:-N+1] 
    return y

# 用numpy的FFT计算循环卷积
# 注意：要求h尺寸不大于x
def conv_1D_numpy_FFT(x,h):
    N,M=len(x),len(h)
    
    # 序列不等长处理
    if N>M:
        h=np.array(list(h)[:]+[0]*(N-M))
        return conv_1D_numpy_FFT(x,h)
    elif M>N:   # 要求h尺寸不大于x
        print('[ERR] x/h size not match!')
        return None

    fx=np.fft.fft(x)    # 时域转为频域
    fh=np.fft.fft(h)    
    fy=fx*fh            # 频域乘积
    y=np.fft.ifft(fy)   # 序列x和h的循环卷积结果
    return np.real(y)

# 用DFT矩阵计算循环卷积
def conv_1D_DFT(x,h,W=None):
    N,M=len(x),len(h)
    if N!=M:
        print('[ERR] x/h size not match!')
        return None

    if W is None:
        w0=np.exp(2.0j*np.pi/float(N))
        W=np.array([[w0**(-r*c) for c in range(N)] for r in range(N)])/np.sqrt(float(N))  # 生成DFT变换矩阵
        if False: print('error:',np.linalg.norm(np.eye(N)-W.dot(np.conj(W.T))))               # 验证矩阵正交性

    Wx=W.dot(x.reshape(N,1))
    Wh=W.dot(h.reshape(N,1))
    y=np.sqrt(N)*np.conj(W).dot(Wx*Wh)                    # 计算循环卷积
    return np.real(y).ravel()

# 根据定义计算2D循环卷积
# 注意：要求H尺寸不大于X
def conv_2D_slow(X,H):
    N,M=X.shape
    K,R=H.shape
    if K>N or R>M:   # 要求H尺寸不大于X
        print('[ERR] X/H size not match!')
        return None
        
    Y=np.zeros((N,M))
    for n in range(N):
        for m in range(M):
            Y[n,m]+=sum([X[(n-k)%N,(m-r)%M]*H[k,r] for k in range(K) for r in range(R)])
    return Y

#
# 用矩阵1D-DFT实现 (式 5-67)
#
# 用DFT矩阵计算2D循环卷积
# 注意：X和H是相同尺寸的实数方阵
# W是DFT变换矩阵，如果需要提高运行速度，需要事先计算好W并传给该API
def conv_2D_DFT(X,H,W=None):
    N,M=X.shape
    K,R=H.shape
    if N!=M or K!=R or N!=K:
        print('[ERR] X/H size not match!')
        return None

    if W is None:
        w0=np.exp(2.0j*np.pi/float(N))
        W=np.array([[w0**(-r*c) for c in range(N)] for r in range(N)])/np.sqrt(float(N))  # 生成DFT变换矩阵
    
    WXW=W.dot(X).dot(W)
    WHW=W.dot(H).dot(W)
    WYW=WXW*WHW 
    Y=N*np.conj(W).dot(WYW).dot(np.conj(W))
    return np.real(Y)

## 使用numpy内带的FFT模块计算2D循环卷积
# 注意：X和H是相同尺寸的实数方阵
# W是DFT变换矩阵，如果需要提高运行速度，需要事先计算好W并传给该API
# 代码清单5-19
def conv_2D_numpy_FFT(X,H):
    N,M=X.shape
    K,R=H.shape
    if N!=M or K!=R or N!=K:
        print('[ERR] X/H size not match!')
        return None

    WXW=np.array([np.fft.fft( X [:,n]).ravel() for n in range(N)]).T    # 逐列 DFT
    WXW=np.array([np.fft.fft(WXW[n,:]).ravel() for n in range(N)])      # 逐行 DFT
    WHW=np.array([np.fft.fft( H [:,n]).ravel() for n in range(N)]).T    # 逐列 DFT
    WHW=np.array([np.fft.fft(WHW[n,:]).ravel() for n in range(N)])      # 逐行 DFT
    WYW=WXW*WHW     
    Y=np.array([np.fft.ifft(WYW[:,n]).ravel() for n in range(N)]).T     # 逐列逆 DFT
    Y=np.array([np.fft.ifft( Y [n,:]).ravel() for n in range(N)])       # 逐行逆 DFT
    return np.real(Y)

# 使用numpy内带的FFT模块计算2D循环卷积
def conv_2D_numpy_FFT2(X,H):
    from numpy.fft import fft2, ifft2
    Y=ifft2(fft2(X)*fft2(H, s=X.shape))
    return np.real(Y)

##########
# 单元测试
##########

if __name__ == '__main__':
    np.random.seed(1234)

    # 测试数据生成
    N,M=10,10    # 卷积序列长度
    x=np.random.randint(-10,10,N).astype(float)
    h=np.random.randint(-10,10,M).astype(float)

    y =conv_1D_slow(x,h)
    y1=conv_1D_DFT(x,h)
    y2=conv_1D_numpy(x,h)
    y3=conv_1D_numpy_FFT(x,h)

    # 计算误差
    print('[INF] y:\n',y)
    if y is not None:
        if y1 is not None: print('[INF] error (y1,y):',np.max(np.abs(y1-y))) 
        if y2 is not None: print('[INF] error (y2,y):',np.max(np.abs(y2-y)))
        if y3 is not None: print('[INF] error (y3,y):',np.max(np.abs(y3-y)))

    # 测试数据生成
    N,M,K,R=10,10,10,10    # 数据尺寸
    X=np.random.randint(-10,10,(N,M)).astype(float)
    H=np.random.randint(-10,10,(K,R)).astype(float)

    Y =conv_2D_slow(X,H)
    Y1=conv_2D_DFT(X,H)
    Y2=conv_2D_numpy_FFT(X,H)
    Y3=conv_2D_numpy_FFT2(X,H)

    # 计算误差
    print('[INF] Y:\n',Y )
    if Y is not None:
        if Y1 is not None: print('[INF] error:',np.max(np.abs(Y-Y1).ravel()))
        if Y2 is not None: print('[INF] error:',np.max(np.abs(Y-Y2).ravel()))
        if Y3 is not None: print('[INF] error:',np.max(np.abs(Y-Y3).ravel()))

