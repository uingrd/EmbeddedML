#!/usr/bin/python3
# coding=utf-8

import os,argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.autograd import Variable
from   torch.optim.lr_scheduler import StepLR

np.random.seed(1234)
torch.manual_seed(1234)

import matplotlib.pyplot as plt
import IPython

DOWNLOAD_DATA     = True

DATA_PATH='./data/'
EXPORT_MODEL_PATH='./export_model/'

########################################
# 演示torch训练MNIST模型并用额外的loss量化
# 通过整数推理和浮点累积
# 代码清单 7-16
########################################

QUANT=True
QUANT_SCALE=10.0
QUANT_CLIP=1.0

## 得到运行参数
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size'     , type=int  , default=100  , metavar='N'         , help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int  , default=1000 , metavar='N'         , help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs'         , type=int  , default=10   , metavar='N'         , help='number of epochs to train (default: 14)')
    parser.add_argument('--lr'             , type=float, default=1.0  , metavar='LR'        , help='learning rate (default: 1.0)')
    parser.add_argument('--gamma'          , type=float, default=0.8  , metavar='M'         , help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda'        ,             default=False, action='store_true' , help='disables CUDA training')
    parser.add_argument('--log-interval'   , type=int  , default=200  , metavar='N'         , help='how many batches to wait before logging training status')
    parser.add_argument('--save-model'     ,             default=True , action='store_true' , help='For Saving the current Model')    
    return parser.parse_args()


def plot_weight_hist(model,block=True):
    w=np.concatenate([np.array(m.weight.data.cpu()).ravel() for m in model.modules() 
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)])
    b,c=np.histogram(w, bins=101)
    
    plt.clf()
    plt.semilogy(c[:-1],b/max(b),'.-')
    plt.show(block=block)
    if not block:
        plt.pause(0.1)


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


## 网络
class model_c(nn.Module):
    def __init__(self):
        super(model_c, self).__init__()

        self.conv1  = nn.Conv2d( 1, 32, 5, 1) # CI= 1, CO=32, K=5, S=1, (N, 1,28,28)->(N,32,24,24)
        self.conv2  = nn.Conv2d(32, 32, 5, 1) # CI=32, CO=32, K=5, S=1, (N,32,24,24)->(N,32,20,20)
        self.dropout= nn.Dropout2d(0.4)  
        self.fc1    = nn.Linear(512, 1024)    # 512=32*4*4, (N,512)->(N,1024)
        self.fc2    = nn.Linear(1024,  10)
        
        self.param_quant=None
    
    # 量化参数的学习更新
    def update_param_quant(self,device):
        if self.param_quant is None:
            self.param_float={name:0 for name in self.state_dict().keys() if name.find('.weight')>=0}
            self.param_quant={name:0 for name in self.state_dict().keys() if name.find('.weight')>=0}
            
        for name,value in self.state_dict().items():
            if name.find('.weight')>=0:
                param_readback=np.array(self.state_dict()[name][:].cpu())       # 读回模型参数
                param_change  =param_readback-self.param_quant[name]            # 提取参数更新量
                self.param_float[name]+=param_change                            # 更新量加入浮点版本
                self.param_quant[name]=self.quant_clip(self.param_float[name])  # 模型参数重新量化
                self.state_dict()[name][:]=torch.from_numpy(self.param_quant[name]).to(device)  # 参数写回模型
    
    # 参数量化和饱和运算
    def quant_clip(self,t,scale=QUANT_SCALE,clip=QUANT_CLIP):
        return np.round(np.clip(t,-QUANT_CLIP,QUANT_CLIP)*scale)/scale

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
            loss.backward()
            optimizer.step()
            
            if epoch>=3:
                model.update_param_quant(device)
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        evaluate(args, model, device, test_loader)
        scheduler.step()
        
        # 调试，打印参数
        if True:
            plot_weight_hist(model, block=False)

## 评估
def evaluate(args, model, device, test_loader): 
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data   = data.resize_(args.test_batch_size, 1, 28, 28)
        
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
        print('[INF] Constructing model...')
        model = model_c()
        
        print('[INF] Trianing...')
        train(args, model, device, train_loader,test_loader)
        torch.save(model, EXPORT_MODEL_PATH+'mnist_cnn.pth')
    
    ## 加载训练好的pytorch模型并测试
    if True:
        print('[INF] Loading model...')
        model=torch.load(EXPORT_MODEL_PATH+'mnist_cnn.pth').cpu()
        
        print('[INF] Testing after quant...')
        evaluate(args, model, torch.device('cpu'), test_loader)
        plot_weight_hist(model, False)
