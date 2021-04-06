# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:09:07 2021

@author: admin
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


data=np.load("data.npz")
traindata_x,traindata_y,testdata_x,testdata_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
print(traindata_x.shape,traindata_y.shape,testdata_x.shape,testdata_y.shape)
print(traindata_x.dtype,traindata_y.dtype,testdata_x.dtype,testdata_y.dtype)

#求正常样本的均值和方差，并以此为基准对所有数据做标准化
normal_samples=np.vstack((traindata_x[:500,:],testdata_x[:960,:]))
normal_labels=np.hstack((traindata_y[:500],testdata_y[:960]))
print(np.unique(normal_labels))
print(normal_samples.shape)

mean=np.mean(normal_samples,axis=0)
print(mean.shape,mean)
std=np.std(normal_samples,axis=0)
print(std.shape,std)

#print('origan:',traindata_x[:2,:])
traindata_x2=(traindata_x-mean)/std
testdata_x2=(testdata_x-mean)/std
#print('processed',traindata_x2[:2,:])

#求每个类别标准化后的norm2，故障和正常数据的norm2有较大差异
norm_0=np.mean(traindata_x2[:500,:]**2)
print(norm_0)
norm=[norm_0]
for i in range(1,22):
    norm_i=np.mean(traindata_x2[500+(i-1)*480:500+i*480,:]**2)
    norm.append(norm_i)
print(norm) 

%matplotlib inline
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12,4))
plt.plot(norm)
plt.show()

#！！！猜想：norm2大的类别对应的测试准确率要高一些，是否是正相关的？

#训练集中的故障数据前填充20个零向量，使得训练集为(22, 500, 52)

traindata_x_pad=[traindata_x2[:500,:]]
traindata_y_pad=[traindata_y[:500]]
for i in range(1,22):
    data_i=traindata_x2[500+(i-1)*480:500+i*480,:]
    #print(data_i.shape)
    data_i=np.vstack((np.zeros((20,52)),data_i))
    #print(i,data_i.shape)
    label_i=traindata_y[500+(i-1)*480:500+i*480]
    label_i=np.hstack((np.zeros(20,dtype=np.int32),label_i))
    
    traindata_x_pad.append(data_i)
    traindata_y_pad.append(label_i)

traindata_x_pad=np.array(traindata_x_pad)
traindata_y_pad=np.array(traindata_y_pad)
traindata_y_pad=traindata_y_pad.reshape(traindata_y_pad.shape[0],traindata_y_pad.shape[1],-1)
print(traindata_x_pad.shape,traindata_y_pad.shape)

dataset1=np.concatenate((traindata_x_pad,traindata_y_pad),axis=2)
print(dataset1.shape)

#reshape测试集，使得训练集为(22, 960, 52)
print(testdata_x2.shape,testdata_y.shape)

testdata_x_pad=testdata_x2.reshape(22,-1,testdata_x2.shape[1])
testdata_y_pad=testdata_y.reshape(22,-1,1)
print(testdata_x_pad.shape,testdata_y_pad.shape)

dataset2=np.concatenate((testdata_x_pad,testdata_y_pad),axis=2)
print(dataset2.shape)

def create_dataset(dataset, look_back=6):
    dataX, dataY = [],[]
    for j in range(dataset.shape[0]):
        for i in range(dataset.shape[1]-look_back):
            a = dataset[j,i:(i + look_back),0:-1]
            dataX.append(a)
            dataY.append(dataset[j,i:(i + look_back),-1])
    
    return np.array(dataX), np.array(dataY,dtype=np.int32)

# 创建好输入输出
train_X, train_Y = create_dataset(dataset1)
test_X, test_Y = create_dataset(dataset2)
print(train_X.shape,train_Y.shape,train_X.dtype,train_Y.dtype)
print(test_X.shape,test_Y.shape,test_X.dtype,test_Y.dtype)

def onehot_encoder(y,cls=22):#y.shape=(n,l)
    
    y_hat=np.zeros((y.shape[0],y.shape[1],cls),dtype=np.int)
    print(y_hat.shape)
    for i in range(y.shape[0]):
        for j in range (y.shape[1]):
            index=int(y[i,j])
            #print(index)
            y_hat[i,j,index]=1
    return y_hat

#x=train_Y[(0,1000,5000),:]
#print(x)
#trainy_hat=onehot_encoder(x)
#print(trainy_hat)

data_all = np.concatenate((train_X,test_X),axis=0)
label_all = np.concatenate((train_Y,test_Y),axis=0)
print(data_all.shape,label_all.shape)

#重新划分训练集和测试集
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(data_all, label_all, test_size=0.2, random_state=99) 
print(train_X.shape,test_X.shape,train_Y.shape,test_Y.shape)

#转换成Tensor
train_x = (torch.from_numpy(train_X)).float()
train_y = torch.from_numpy(train_Y).long()
test_x = (torch.from_numpy(test_X)).float()
test_y = torch.from_numpy(test_Y).long()
print(train_x.dtype,train_x.size())
print(test_x.dtype,test_x.size())
print(train_y.dtype,train_y.size())
print(test_y.dtype,test_y.size())

# 先转换成 torch 能识别的 Dataset
import torch.utils.data as Data

train_dataset = Data.TensorDataset(train_x,train_y)
# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=64,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
test_dataset = Data.TensorDataset(test_x,test_y)
# 把 dataset 放入 DataLoader
test_loader = Data.DataLoader(
    dataset=test_dataset,      # torch TensorDataset format
    batch_size=64,      # mini batch size
    #shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

from collections import OrderedDict

class Sparse_autoencoder(nn.Module):
    def __init__(self, input_size=52, embedding_size=30, output_size=22, device=device):
        super(Sparse_autoencoder,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.output_size=output_size
        self.device=device
        
        self.dropout = nn.Dropout(p=0.2)  #1/(1-p)
        self.encoder=nn.Sequential(OrderedDict([
                    ('sparse1',nn.Linear(self.input_size, self.input_size)),
                    ('action1',nn.Tanh()),
                    ('sparse2',nn.Linear(self.input_size, self.embedding_size)),
                    ('action2',nn.Tanh())
                    ]))
        self.decoder=nn.Sequential(OrderedDict([
                    ('fc1',nn.Linear(self.embedding_size, self.input_size)),
                    ('action3',nn.Tanh()),
                    ('fc2',nn.Linear(self.input_size, self.input_size))
                    ]))
        self.reg = nn.Linear(self.embedding_size, self.output_size)
        
    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        self.encoder.sparse1.weight.data.uniform_(-initrange, initrange)
        self.encoder.sparse1.bias.data.uniform_(-initrange, initrange)
        self.encoder.sparse2.weight.data.uniform_(-initrange, initrange)
        self.encoder.sparse2.bias.data.uniform_(-initrange, initrange) 
        self.decoder.fc1.weight.data.uniform_(-initrange, initrange)
        self.decoder.fc1.bias.data.uniform_(-initrange, initrange)
        self.decoder.fc2.weight.data.uniform_(-initrange, initrange)
        self.decoder.fc2.bias.data.uniform_(-initrange, initrange)
        self.reg.weight.data.uniform_(-initrange, initrange)
        self.reg.bias.data.uniform_(-initrange, initrange)
        
    def forward(self, x): 
        b, l, h = x.shape  #(batch, seq, hidden)
        x = x.contiguous().view(l*b,-1) 
        e = self.encoder(self.dropout(x))
        x_bar = self.decoder(e)
        #out = self.reg(self.dropout(e))
        out = self.reg(e)
        x_bar = x_bar.contiguous().view(b,l,-1)
        
        return x_bar,e,out
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
        
Sparse_AE = Sparse_autoencoder(device=device)
Sparse_AE.to(device)
Loss_MSE = nn.MSELoss()
Loss_CE = nn.CrossEntropyLoss()
optimizer_SparseAE = torch.optim.SGD(Sparse_AE.parameters(), lr=1e-3,momentum=0.9)

#稀疏编码器加L1约束
def L1_penalty(param, debug=False):
    if isinstance(param, torch.Tensor):
        param= [param]
    total_L1_norm=0
    for p in filter(lambda p: p.data is not None, param):
        param_norm = p.data.norm(p=1) 
        if debug:print('param_norm',param_norm)
        total_L1_norm += param_norm
        if debug:print('L1',total_L1_norm)
        
    return total_L1_norm

test=torch.tensor([[[-1.0,-2.0],
                   [1.0,2.0]],
                   [[-1.0,-2.0],
                   [1.0,2.0]]])
print(test.shape)
print(L1_penalty(test,debug=True))

def val_AE(net,test_loader,lambd1,lambd2):
    Losses=[]
    Acces=[]
    Acc=[0]*22 
    Det=[0]*22
    Total=[0]*22
    #net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
          #if step<3:
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            batch_y=batch_y.view(-1)
            
            x_bar,embedding,out = net(batch_x)
            Loss_L1_penalty = lambd1 * L1_penalty(embedding)
            loss_L2 = Loss_MSE(x_bar,batch_x)            
            loss_CE = lambd2 * Loss_CE(out,batch_y)
            loss = loss_L2 + Loss_L1_penalty + loss_CE
            total = batch_y.size(0)
            Losses.append(loss.item()/total)
            
            out=F.softmax(out,1)
            _,index=out.max(dim=1)
            acc=(index==batch_y).sum().cpu().numpy() 
            Acces.append(acc/total)
            
              #按类别统计正确率
            for i in range(batch_y.shape[0]):
                Total[batch_y[i]]+=1
                if index[i]==batch_y[i]:
                    Acc[batch_y[i]]+=1
                if (index[i]>0)==(batch_y[i]>0):
                    Det[batch_y[i]]+=1
    for i in range(22):
        Acc[i]/=Total[i]
        Det[i]/=Total[i]
   
        #print(Losses)
    return np.mean(Losses),np.mean(Acces),Acc,Det

#Loss,ACC = val_AE(net,test_loader)   
#print(Loss)

#开始训练
train_Loss=[]
test_Loss=[]
train_Acc=[]
test_Acc=[]

allclass_acc=[]
alldet_acc=[]

def train(net, train_lodader, optimizer, lambd1=1e-4, lambd2=100, epoch=200):
    net.init_weights()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.5)
    #net.to(device)
    for e in range(epoch):
        Losses=[]
        Acces=[]
        for step, (batch_x, batch_y) in enumerate(train_loader):
          #if step<300:  
            scheduler.step()
            optimizer.zero_grad()
            batch_x=batch_x.to(device)    
            batch_y=batch_y.to(device)
            batch_y=batch_y.view(-1)
            
            x_bar,embedding,out = net(batch_x)            
            Loss_L1_penalty = lambd1* L1_penalty(embedding)
            loss_L2 = Loss_MSE(x_bar,batch_x)
            loss_CE = lambd2 * Loss_CE(out,batch_y)
            
            loss = loss_L2 + Loss_L1_penalty + loss_CE
            if step==0 and (e+1)%10==0:
                print(loss_L2,Loss_L1_penalty,loss_CE,loss)

            #反向传播      
            loss.backward()
            optimizer.step()

            total = batch_y.size(0)
            Losses.append(loss.item()/total)
            
            out=F.softmax(out,1)
            _,index=out.max(dim=1)
            acc=(index==batch_y).sum().cpu().numpy()       
            Acces.append(acc/total)
            
        #print(Losses,Acces)        
        if e==0: print('Epoch:0, TrainLoss:{:.5f},'.format(Losses[0]))
        Loss,Acc,ACC,Det=val_AE(Sparse_AE,test_loader,lambd1=lambd1,lambd2=lambd2)
        test_Loss.append(Loss)
        train_Loss.append(np.mean(Losses))
        test_Acc.append(Acc)
        train_Acc.append(np.mean(Acces))
        allclass_acc.append(ACC)
        alldet_acc.append(Det)
        print('Epoch:{}, TrainLoss:{:.5f}, TestLoss:{:.5f}, TrainAcc:{:.5f}, TestAcc:{:.5f},'.format(e+1, np.mean(Losses),Loss,np.mean(Acces), Acc ))     
    
train(Sparse_AE,train_loader,optimizer_SparseAE)

print("classication_acc",allclass_acc[-1])
print("det_acc",alldet_acc[-1])
k=[1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20]
det=[]
for i in k:
    det.append(alldet_acc[i])
det_mean=np.mean(np.array(det))
print("mean",det_mean)

#迭代200次, lambd=1e-5
print(test_Acc[-1])
plt.plot(train_Loss[1:])
plt.plot(test_Loss)
plt.plot(train_Acc)
plt.plot(test_Acc)
plt.show()

Sparse_AE.save_model("SparseAE_nodropout_epoch200.pth")
Sparse_AE.load_model("SparseAE_epoch200.pth")


test_seq_x,test_seq_y=test_dataset[100:120]
print(test_seq_y)
Embedding = Sparse_AE.encoder(test_seq_x.cuda())
print(Embedding.shape,Embedding[0,:,:])

print((Embedding==0).sum().float()/(20*6*200))