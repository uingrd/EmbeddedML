#!/usr/bin/python3
# coding=utf-8

import IPython

import cv2
import numpy as np
import scipy as sci
from scipy import signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####################
# 近似2D卷积例程
####################


np.random.seed(1234)

def sim_img(wid=400,hgt=400):
    imgR,imgG,imgB=np.random.rand(3,hgt,wid)-0.5
    img=np.stack((signal.convolve2d(imgR,np.ones((5,5)),mode='same'),
                  signal.convolve2d(imgG,np.ones((5,5)),mode='same'),
                  signal.convolve2d(imgB,np.ones((5,5)),mode='same')))
    return img.transpose((1,2,0))

## 图像沿水平方向卷积
def conv_row(img,ker_row):
    hgt,wid=img.shape
    img_out=np.zeros((hgt,wid+len(ker_row)-1))
    for n in range(hgt): 
        img_out[n,:]=np.convolve(img[n,:],ker_row)
    return img_out


## 图像沿垂直方向卷积
def conv_col(img,ker_col):
    hgt,wid=img.shape
    img_out=np.zeros((hgt+len(ker_col)-1,wid))
    for n in range(wid): 
        img_out[:,n]=np.convolve(img[:,n],ker_col)
    return img_out


## 图像分别沿水平和垂直方向卷积
def conv_2d_sep(img, ker_col, ker_row):
    img_tmp=conv_col(img ,ker_col) 
    return conv_row(img_tmp,ker_row)


## 计算相对误差
def calc_img_err(img1,img2):
    return np.max(np.abs((img1-img2)/(img1+img2)/2))

    
    
if True:
    # 测试图像
    img=plt.imread('test_img.jpg') if False else sim_img()
    img=img[:,:,0].astype(float)
    img=cv2.resize(img, (100, 50)).astype(float)

    ker_col=np.array([4,-1,3]).astype(float)
    ker_row=np.array([1, 2,3]).astype(float)
    ker=np.dot(ker_col.reshape(3,1),ker_row.reshape(1,3))
    
    img_ref=signal.convolve2d(img,ker)
    img_out=conv_2d_sep(img,ker_col,ker_row)

    print('err:',np.max(np.abs(img_ref-img_out)))
    print('calc_img_err():',calc_img_err(img_ref,img_out))
    
    plt.subplot(3,1,1)
    plt.imshow(img_ref)
    plt.title('img_ref')
    
    plt.subplot(3,1,2)
    plt.imshow(img_out)
    plt.title('img_out')
    
    plt.subplot(3,1,3)
    plt.imshow(img_ref-img_out)
    plt.title('img_ref-img_out')
    plt.show()


def approx_conv_2d(H,R,img,UR=None,VR=None,ret_SVD=False):
    # H的SVD分解
    if UR is None or VR is None:
        U,S,Vh=np.linalg.svd(H)
        UR=[(U[:,r]*S[r]).ravel() for r in range(R)]
        VR=[(Vh.T[:,r]).ravel()   for r in range(R)]
    
    # 计算近似2D卷积
    img_out=np.zeros((np.size(img,0)+np.size(H,0)-1,\
            np.size(img,1)+np.size(H,1)-1))
    for ker_col,ker_row in zip(UR,VR):
        img_out+=conv_2d_sep(img, ker_col, ker_row)
    return img_out if not ret_SVD else (img_out,UR,VR)


if True:
    K,R=5,3
    S=[7,6,5,0.001,0.001]
    ker=np.zeros((K,K))
    for s in S:
        u=np.random.randint(-10,10,K).astype(float)
        v=np.random.randint(-10,10,K).astype(float)
        ker+=s*np.dot(u.reshape(K,1),v.reshape(1,K))

    # 测试图像
    img=plt.imread('test_img.jpg') if False else sim_img()
    img=img[:,:,0].astype(float)
    img=cv2.resize(img, (100, 50)).astype(float)

    # 近似卷积
    img_out=approx_conv_2d(ker,R,img)
    
    # 参考答案
    img_ref=signal.convolve2d(img,ker)
    print('err:',np.max(np.abs(img_out-img_ref)))
    print('calc_img_err():',calc_img_err(img_ref,img_out))
    
    plt.subplot(3,1,1)
    plt.imshow(img_ref)
    plt.title('img_ref')
    
    plt.subplot(3,1,2)
    plt.imshow(img_out)
    plt.title('img_out')
    
    plt.subplot(3,1,3)
    plt.imshow(img_ref-img_out)
    plt.title('img_ref-img_out')
    plt.show()


## 2D矩形卷积
def box_conv_2D(X,K1,K2,c=1):
    M,N=X.shape

    D=np.cumsum(X,axis=0)
    D=np.cumsum(D,axis=1)
    #D=X.copy()
    #for m in range(M):
    #    for n in range(1,N):
    #        D[m,n]=D[m,n]+D[m,n-1]
    #for n in range(N):
    #    for m in range(1,M):
    #        D[m,n]=D[m,n]+D[m-1,n]
    
    Y=D[K1-1:M,K2-1:N].copy()
    Y[1:, :]-=D[0:M-K1,K2-1:N]
    Y[ :,1:]-=D[K1-1:M,0:N-K2]
    Y[1:,1:]+=D[0:M-K1,0:N-K2]
    
    return Y*c

if True:
    M,N,K1,K2=100,80,15,11

    X=np.random.randint(-10,10,(M,N)).astype(float)
    Y=box_conv_2D(X,K1,K2)

    H=np.ones((K1,K2))
    Y_ref=signal.convolve2d(X,H,'valid')
    print('err:',np.max(np.abs(Y_ref-Y)))
    plt.subplot(1,2,1)
    plt.imshow(Y)
    plt.subplot(1,2,2)
    plt.imshow(Y_ref)
    plt.show()

## 用2D盒形卷积核近似表达给定卷积核，并测试用矩形卷积计算近似卷积
if True:
    H=np.array([[1,2,2,2,1],
                [2,3,3,3,2],
                [2,3,3,3,2],
                [2,3,3,3,2],
                [1,2,2,2,1]]).astype(float)
    Hn=H+(np.random.rand(5,5)-0.5)*0.5
    print('Hn:',Hn)

    # 显示上述卷积核
    x,y = np.meshgrid(np.arange(5),np.arange(5))
    xx,yy = x.ravel(), y.ravel()

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(xx, yy, np.zeros_like(H.ravel()), 1, 1, H.ravel(), shade=True,color='gray')
    ax.set_title('3D conv. kernel')
    ax.set_xlim([-1,6])
    ax.set_ylim([-1,6])
    ax.set_zlim([0,5])
    plt.show()

    # 分解成3个矩形卷积核，用他们进行快速卷积
    M,N=80,80
    X=np.random.randint(-10,10,(M,N)).astype(float)
    Y1=box_conv_2D(X,5,5)
    Y2=box_conv_2D(X,5,3)
    Y3=box_conv_2D(X,3,5)
    
    Y=Y1
    Y+=Y2[:,1:-1]
    Y+=Y3[1:-1,:]
    
    Y_ref=signal.convolve2d(X,H,'valid')
    print('err:',np.max(np.abs(Y_ref-Y)))
    
    plt.subplot(1,2,1)
    plt.imshow(Y)
    plt.subplot(1,2,2)
    plt.imshow(Y_ref)
    plt.show()
    
    plt.imshow(Y-Y_ref)
    plt.show()    
    
