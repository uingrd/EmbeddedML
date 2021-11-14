#!/usr/bin/python3.5
# coding=utf-8

import os,argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.autograd import Variable
from   torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import IPython

# 运行记录
import logging
logging.basicConfig(filename='run.log',format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)
def print_log(info):
    print(info)
    logging.info(info)

DOWNLOAD_DATA=True

DATA_PATH='./data/'
EXPORT_MODEL_PATH='./export_model/'
EXPORT_CODE_PATH ='./export_code/'

EXTRA_LOSS_SCALE=0.1e-3
TH=0.03

########################################
# 演示torch训练MNIST模型并测试权重系数稀疏化
# 代码清单 7-18
# 代码清单 7-19
########################################

## 得到运行参数
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size'     , type=int  , default=32   , metavar='N' , help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int  , default=1000 , metavar='N' , help='input batch size for testing')
    parser.add_argument('--epochs'         , type=int  , default=2   , metavar='N' , help='number of epochs to train')
    parser.add_argument('--lr'             , type=float, default=1.0  , metavar='LR', help='learning rate')
    parser.add_argument('--gamma'          , type=float, default=0.8  , metavar='M' , help='Learning rate step gamma')
    parser.add_argument('--no-cuda'        ,             default=False, action='store_true',  help='disables CUDA training')
    parser.add_argument('--seed'            , type=int , default=1    , metavar='S' , help='random seed')
    
    ret=parser.parse_args()
    print_log('[INF] '+str(ret))
    return ret
    
## 加载数据，返回数据加载器
def load_data(batch_size, test_batch_size):
    print('[INF] Generating data loader...')
    
    # 加载numpy数据
    print('[INF] Loading mnist data from npy file...')
    if DOWNLOAD_DATA:
        from torchvision import datasets
        data_train = datasets.MNIST(root = DATA_PATH,train = True,download = True)
        data_test = datasets.MNIST(root=DATA_PATH, train = False)
        
        train_x_np=data_train.data.numpy().astype(np.float32)
        train_y_np=data_train.targets.numpy().astype(int)
        test_x_np=data_test.data.numpy().astype(np.float32)
        test_y_np=data_test.targets.numpy().astype(int)
        
        np.save(DATA_PATH+'train_x.npy', train_x_np)
        np.save(DATA_PATH+'train_y.npy', train_y_np)
        np.save(DATA_PATH+'test_x.npy' , test_x_np)
        np.save(DATA_PATH+'test_y.npy' , test_y_np)
    else:
        train_x_np=np.load(DATA_PATH+'train_x.npy')
        train_y_np=np.load(DATA_PATH+'train_y.npy')
        test_x_np =np.load(DATA_PATH+'test_x.npy')
        test_y_np =np.load(DATA_PATH+'test_y.npy')
    
    # 构建data loader
    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np).type(torch.LongTensor)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x,train_y), 
                                               batch_size=batch_size, shuffle=False)

    test_x = torch.from_numpy(test_x_np)
    test_y = torch.from_numpy(test_y_np).type(torch.LongTensor)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x,test_y),
                                              batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


## 随机数种子设置
def set_random_seed(args):
    np.random.seed(args.seed)    
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available(): 
        torch.manual_seed(args.seed)

def plot_weight_hist(model,block=True):
    w=np.concatenate([np.array(m.weight.data.cpu()).ravel() for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)])
    b,c=np.histogram(w, bins=101)
    
    plt.clf()
    plt.semilogy(c[:-1],b/max(b),'.-')
    plt.show(block=block)
    if not block:
        plt.pause(0.1)
    

## 网络
class mnist_c(nn.Module):
    def __init__(self):
        super(mnist_c, self).__init__()

        self.conv1 = nn.Conv2d( 1, 32, 5, 1) # CI= 1, CO=32, K=5, S=1, (N, 1,28,28)->(N,32,24,24)
        self.conv2 = nn.Conv2d(32, 32, 5, 1) # CI=32, CO=32, K=5, S=1, (N,32,24,24)->(N,32,20,20)
        self.dropout = nn.Dropout2d(0.4)  
        self.fc1 = nn.Linear(512, 1024)      # 512=32*4*4, (N,512)->(N,1024)
        self.fc2 = nn.Linear(1024,  10)
        
        self.extra_loss=None
        return
        
    # 训练和推理
    def forward(self,x):
        x = self.conv1(x)       # (N, 1,28,28)->(N,32,24,24)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (N,32,24,24)->(N,32,12,12)
        x = self.conv2(x)       # (N,32,12,12)->(N,32, 8, 8)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (N,32, 8, 8)->(N,32, 4, 4)
        x = torch.flatten(x, 1) # (N,32, 4, 4)->(N,512)
        x = self.fc1(x)         # (N,512)->(N,1024)
        x = F.relu(x)           
        x = self.dropout(x)
        x = self.fc2(x)         # (N,1024)->(N,10)
        
        self.extra_loss=0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.extra_loss+=torch.sum(torch.abs(m.weight))
        return x


## 训练
def train(args, model, device, train_loader, test_loader):
    model.to(device)

    print('[INF] Preparing for train...')
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data   = data.resize_(args.batch_size, 1, 28, 28)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss+=model.extra_loss*EXTRA_LOSS_SCALE
                
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()),
                    end='')
        
        if True: plot_weight_hist(model,False)
        evaluate(args, model, device, test_loader)
        scheduler.step()


## 评估
def evaluate(args, model, device, test_loader): 
    model.to(device)
    model.eval()
    test_loss=correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.resize_(args.test_batch_size, 1, 28, 28)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss/=len(test_loader.dataset)
    acc=correct/len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.*acc))
    return acc


def export_param_to_c(model,path):
    # 生成c文件
    fname=path+'param.c'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name,value in model.state_dict().items():
        value=value.numpy().astype(np.float32)  # 网络参数
        name=name.replace('.','_')              # 网络参数名称

        # 输出权重形状数据(整数数组)
        if name[-7:]=='_weight':
            fp.write('const int %s[]={'%(name+'_shape'))
            for n in value.shape:
                fp.write('%d, '%n)
            fp.write('};\n')
        
        # 输出网络参数(浮点数组)
        fp.write('const float %s[]={'%name)
        for n,v in enumerate(value.flatten()):
            if n%10==0: fp.write('\n    ')
            fp.write('(float)%+0.8e, '%v)
        fp.write('\n};\n')
    fp.close()

    # 生成头文件(C程序里的数组申明)
    fname=path+'param.h'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name in model.state_dict():
        name=name.replace('.','_')
        fp.write('extern const float %s[];\n'%name)
        if name[-7:]=='_weight':
            fp.write('extern const int %s[];\n'%(name+'_shape'))
    fp.close()


## 生成C代码，8-bit量化,-128~127，v=q/128
def export_param_to_c_q8(model,path):
    # 生成c文件
    fname=path+'param.c'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name,value in model.state_dict().items():
        value=value.numpy().astype(np.float32)  # 网络参数
        name=name.replace('.','_')              # 网络参数名称

        # 输出权重形状数据(整数数组)
        if name[-7:]=='_weight':
            fp.write('const int %s[]={'%(name+'_shape'))
            for n in value.shape:
                fp.write('%d, '%n)
            fp.write('};\n')
        
        # 输出网络参数(浮点数组)
        if name[-7:]=='_weight':
            fp.write('const signed char %s[]={'%name)
            for n,v in enumerate(value.flatten()):
                if n%40==0: fp.write('\n    ')
                q=int(round(v*128))
                q=max(min(q,127),-128)
                fp.write('%d, '%q)
            fp.write('\n};\n')        
        else:
            fp.write('const float %s[]={'%name)
            for n,v in enumerate(value.flatten()):
                if n%10==0: fp.write('\n    ')
                fp.write('(float)%+0.8e, '%v)
            fp.write('\n};\n')
    fp.close()

    # 生成头文件(C程序里的数组申明)
    fname=path+'param.h'
    print('[INF] generating %s...'%fname)
    fp=open(fname,'wt')
    for name in model.state_dict():
        name=name.replace('.','_')
        if name[-7:]=='_weight':
            fp.write('extern const signed char %s[];\n'%name)
            fp.write('extern const int %s[];\n'%(name+'_shape'))
        else:
            fp.write('extern const float %s[];\n'%name)
    fp.close()


def export_test_data_to_c(test_x,test_y,num,path):
        # 生成头文件
        fp=open(path+'test_data.h','wt')
        fp.write('#define TEST_DATA_NUM %d\n'%num)
        fp.write('#define TEST_DATA_SIZE %d\n'%test_x[0].size)
        fp.write('extern const float test_x[TEST_DATA_NUM][TEST_DATA_SIZE];\n')
        fp.write('extern const unsigned char test_y[TEST_DATA_NUM];\n')
        fp.close()
        
        # 生成C文件
        fp=open(path+'test_data.c','wt')
        fp.write('#include "test_data.h"\n')
        # 导出图像数据到C程序
        fp.write('const float test_x[TEST_DATA_NUM][TEST_DATA_SIZE]={\n')
        for n in range(num):
            fp.write('    {')
            for v in test_x[n]: fp.write('(float)%+0.8e, '%v)
            fp.write('},\n')
        fp.write('};\n')
        # 导出参考答案到C程序
        fp.write('const unsigned char test_y[TEST_DATA_NUM]={\n    ')
        for n in range(num):
            fp.write('%u, '%test_y[n])
        fp.write('\n};\n')
        fp.close()


## 主入口
if __name__ == '__main__':
    args = get_args()
    if not args.no_cuda and torch.cuda.is_available():
        print('[INF] using CUDA...')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('[INF] Generating data loader...')
    train_loader, test_loader=load_data(args.batch_size,args.test_batch_size)
    
    ## 训练pytorch模型，测试并保存
    if True:
        set_random_seed(args)
        print('[INF] Constructing model...')
        model = mnist_c()
        
        print('[INF] Training...')
        train(args, model, device, train_loader,test_loader)
        torch.save(model, EXPORT_MODEL_PATH+'mnist_cnn.pth')
    
    ## 以CPU模式加载训练好的pytorch模型，过小的权重置零，然后测试
    if True:
        print('[INF] Loading model...')
        model=torch.load(EXPORT_MODEL_PATH+'mnist_cnn.pth').cpu()
        
        print('[INF] Original weights...')
        evaluate(args, model, torch.device('cpu'), test_loader)
        if True: plot_weight_hist(model,False)
        
        cnt1=cnt2=0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data[torch.le(torch.abs(m.weight),TH)]=0
                
                cnt1+=np.array(m.weight.data).size
                cnt2+=np.sum(np.abs(np.array(m.weight.data))>TH)
        
        print('[INF] non-zero weights: %.2f%%'%(float(cnt2)/float(cnt1)*100.0))
        evaluate(args, model, torch.device('cpu'), test_loader)
        if True: plot_weight_hist(model,False)
        torch.save(model, EXPORT_MODEL_PATH+'mnist_cnn_zero.pth')
        
    ## 导出模型参数到c文件
    if True:
        print('[INF] generating C source file for network parameters...')
        model=torch.load(EXPORT_MODEL_PATH+'mnist_cnn_zero.pth').cpu()
        if False:
            export_param_to_c(model,EXPORT_CODE_PATH)
        else:
            export_param_to_c_q8(model,EXPORT_CODE_PATH)

    # 导出测试数据到c文件
    if True:
        print('[INF] generating C source file for test data...')
        test_x=np.array([x.reshape(28,28).flatten() for x in np.load(DATA_PATH+'test_x.npy')]).astype(np.float32)
        test_y=np.load(DATA_PATH+'test_y.npy').flatten().astype(np.uint8)
        export_test_data_to_c(test_x,test_y,    # 测试数据和参考答案
                              20,               # 导出前500个数据
                              EXPORT_CODE_PATH) # 路径
        
