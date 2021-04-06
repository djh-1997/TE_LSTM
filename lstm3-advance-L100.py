import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

data=np.load("data.npz")  # 读取TE数据
traindata_x,traindata_y,testdata_x,testdata_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

#求正常样本的均值和方差，并以此为基准对所有数据做标准化
mean=np.mean(traindata_x[:500,:],axis=0)
std=np.std(traindata_x[:500,:],axis=0)
traindata_x=(traindata_x-mean)/std  # 处理为正态分布的训练数据
testdata_x=(testdata_x-mean)/std    # 处理为正态分布的测试数据

traindata_x=traindata_x[20:,:].reshape(22,-1,52) #(22, 480, 52) 
traindata_y=traindata_y[20:].reshape(22,-1,1) #(22, 480, 1)
dataset1=np.concatenate((traindata_x,traindata_y),axis=2)  #训练数据集 (22, 500, 53)
# print(dataset1.shape)

#reshape测试集，使得训练集为(22, 960, 53)
testdata_x_pad=testdata_x.reshape(22,-1,testdata_x.shape[1])
testdata_y_pad=testdata_y.reshape(22,-1,1)
# print(testdata_x_pad.shape,testdata_y_pad.shape) # (22, 960, 52) (22, 960, 1)

dataset2=np.concatenate((testdata_x_pad,testdata_y_pad),axis=2) #(22, 960, 53)

def create_dataset(dataset, look_back=100):
    dataX, dataY = [],[]
    for j in range(dataset.shape[0]):
        for i in range(dataset.shape[1]-look_back):
            a = dataset[j,i:(i + look_back),0:-1]  #0：-1，第53个没有取
            dataX.append(a)
            dataY.append(dataset[j,i:(i + look_back),-1])
    
    return np.array(dataX), np.array(dataY,dtype=np.int32)

# 创建好输入输出
train_X, train_Y = create_dataset(dataset1) #（22，500，53）到（22*400，100，52）（22*400，100，1）
test_X, test_Y = create_dataset(dataset2)   #（22，960，53）到（22*860，100，52）（22*860，100，1）
# print(train_X.shape,train_Y.shape,train_X.dtype,train_Y.dtype)
# print(test_X.shape,test_Y.shape,test_X.dtype,test_Y.dtype)

def onehot_encoder(y,cls=22):#y.shape=(n,l)  
    y_hat=np.zeros((y.shape[0],y.shape[1],cls),dtype=np.int)
    print(y_hat.shape)
    for i in range(y.shape[0]):
        for j in range (y.shape[1]):
            index=int(y[i,j])
            y_hat[i,j,index]=1
    return y_hat
data_all = np.concatenate((train_X,test_X),axis=0)
label_all = np.concatenate((train_Y,test_Y),axis=0)
# print(data_all.shape,label_all.shape)

#重新划分训练集和测试集
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(data_all, label_all, test_size=0.2, random_state=42) 
# print(train_X.shape,test_X.shape,train_Y.shape,test_Y.shape)

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
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

#定义模型
class lstm(nn.Module):
    def __init__(self, input_size=52, hidden_size=100, output_size=22, num_layers=3, batch_first=True, device=device):            
          super(lstm, self).__init__()
          self.input_size=input_size
          self.hidden_size=hidden_size
          self.output_size=output_size
          self.num_layers=num_layers
          self.batch_first=batch_first
          self.device=device
        
          self.dropout = nn.Dropout(p=0.2)  #1/(1-p)
          self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
          self.reg = nn.Linear(self.hidden_size, self.output_size)

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.reg.weight.data.uniform_(-initrange, initrange)
        self.reg.bias.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, input):
        minibatch_size = input.size(0) \
                if self.batch_first else input.size(1)
        h0 = torch.zeros(self.num_layers, minibatch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, minibatch_size, self.hidden_size, device=self.device)
        return (h0, c0) 
    
    def forward(self, x):
         h0_c0 = self.init_hidden(x)
         out,ht_ct = self.rnn(self.dropout(x),None) 
         b, l, h = out.shape  #(batch, seq, hidden)
         #print(out.shape)
         out = out.contiguous().view(l*b, h) #转化为线性层的输入方式
         y = self.reg(self.dropout(out))
         #y = y.view(b, l, -1)        
         return y  
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

    
net = lstm(device=device)
net.to(device)
'''
weight=torch.tensor([0.040153845166554944, 0.036382060580354744, 0.036219653713873694, 0.07578880042180282, 0.03661752326822607, 0.03787888800374739, 0.03614607125330093, 0.03612018654701804, 0.036784091527261545, 0.059546093073929995, 0.04380238498517394,
                     0.04286410139544747, 0.03880078879783693, 0.039248820284410964, 0.03688048020961498, 0.1082217246438273, 0.052223568477666946, 0.03733155618292655, 0.04264773962980153, 0.04291206161444662, 0.04511382570380278, 0.03831573451897396]
                    ,device=device)
print(weight.dtype,weight.device)
criterion = nn.CrossEntropyLoss(weight=weight)
'''
criterion = nn.CrossEntropyLoss()

#稀疏编码器加L1约束
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

def val_acc(net,test_loader):
    Losses=[]
    Acces=[] 
    Dets=[]
    #net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
          #if step<3:
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            out = net(batch_x)
            batch_y=batch_y.view(-1)
            loss = criterion(out,batch_y)
            total = batch_y.size(0)
            Losses.append(loss.item()/total)
            
            out=F.softmax(out,1)
            _,index=out.max(dim=1)
            #print(index.size(),index)
            #print('test_result:',np.unique(index.cpu().numpy()))
            
            acc=(index==batch_y).sum().cpu().numpy()       
            Acces.append(acc/total)
            det=((index>0)==(batch_y>0)).sum().cpu().numpy()       
            Dets.append(det/total)
        #print(Losses,Acces)
    return np.mean(Losses),np.mean(Acces),np.mean(Dets)

#Loss,Acc=val_acc(net,test_loader)   
#print(Loss,Acc)

#开始训练
net.init_weights()
net.to(device)
for e in range(20):
    Losses=[]
    Acces=[] 
    Dets=[]
    for step, (batch_x, batch_y) in enumerate(train_loader):
      #if step<300:  
        optimizer.zero_grad()
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        #print('input:',batch_x.size(),batch_y.size())
        out = net(batch_x)
        #print('forward:',out.size(),batch_y.size())
        #out=out.view(-1,22)
        batch_y=batch_y.view(-1)
        #print('reshape:',out.size(),batch_y.size())
        loss = criterion(out,batch_y)
        
        #反向传播      
        loss.backward()
        #grandent clip
        #torch.nn.utils.clip_grad_value_(net.parameters(), 5)
        optimizer.step()
        
        total = batch_y.size(0)
        Losses.append(loss.item()/total)
        
        out=F.softmax(out,1)
        _,index=out.max(dim=1)
        #print(index.size(),index)
        #print('train_result:',np.unique(index.cpu().numpy()))
        
        acc=(index==batch_y).sum().cpu().numpy()       
        Acces.append(acc/total)
        det=((index>0)==(batch_y>0)).sum().cpu().numpy()       
        Dets.append(det/total)
    
    #print(Losses,Acces)        
    if e==0: print('Epoch:0, TrainLoss:{:.5f}, TrainAcc:{:.5f}, TrainDet:{:.5f},'.format(Losses[0],Acces[0],Dets[0]))
    print('Epoch:{}, TrainLoss:{:.5f}, TrainAcc:{:.5f}, TrainDet:{:.5f}'.format(e+1, np.mean(Losses),np.mean(Acces),np.mean(Dets)))
    if (e+1)%10==0:
          Loss,Acc,Det=val_acc(net,test_loader)
          print('  Val Epoch:{}, TestLoss:{:.5f}, TestAcc:{:.5f}, TestDet:{:.5f},'.format(e+1, np.mean(Loss),np.mean(Acc),np.mean(Det)))
          
net.save_model("LSTM3_L100_epoch20.pth")
net.load_model("LSTM3_L100_epoch20.pth")

#batch_size=64
look_back=100
#训练完成之后，我们可以用训练好的模型去预测后面的结果
def jude(x):
    if x in (3,9,15):return True
    else: return False

Acc=[0]*22 
Det=[0]*22
Total=[0]*22
net = net.eval()
with torch.no_grad():
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        out = net(batch_x)
        batch_y=batch_y.view(-1)
        out=F.softmax(out,1)
        _,index=out.max(dim=1)
        
       # print('test_result:',np.unique(index.cpu().numpy()))
        index=index.cpu().numpy()
        #输出3，9，15的预测值和真实值
        ind=index.reshape(-1,look_back)
        #print(ind.shape)
        row=np.where(ind==15)[0] 
        print("********************************************************************")
        print(row.shape)
        #print("row=",row)
        #for i in row:
           # if (ind[i,:]!=batch_y.reshape(-1,look_back)[i,:]).any():
               # print(ind[i,:],batch_y.reshape(-1,look_back)[i,:])
        #print(ind.shape,ind[row,:])
        batch_y=batch_y.cpu().numpy()
        print("********************************************************************")
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

#print(sum(Total)/6,Total)
print(Acc)
print(Det)

fig = plt.figure(figsize=(12,8))
#fig.subsubplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8，hspace=0.2, wspace=0.3)
sub1=plt.subplot(2,1,1)
sub2=plt.subplot(2,1,2)

sub1.plot(Acc)
sub1.set_title(u'classfication accuracy')
sub1.set_xticks(range(22))
sub1.grid(True)
sub2.plot(Det)
sub2.set_title(u'detection accuracy')
sub2.set_xticks(range(22))
sub2.grid(True)
plt.show()

import numpy as np
Acc1=np.array([0.9948960589298277, 0.9997762062483215, 0.9998750988800533, 0.9787325088792845, 0.9990940387751405,
               0.996909355821449, 1.0, 0.9989421882904843, 0.9900534901198002, 0.9820989143546441, 0.987306773909765,
               0.990609657276324, 0.9932675747612337, 0.9938020110957004, 0.998448543354594, 0.9714311668256347,
               0.9860397361712459, 0.9979317476732161, 0.990303648890023, 0.9907189754397341, 0.983900198086568,
               0.9908818920074085])
Det1=np.array([0.9948960589298277, 1.0, 0.9998750988800533, 0.9846805597158629, 0.999909403877514, 0.9983222217316438, 
               1.0, 0.9998620245596284, 0.9991158657884267, 0.9863932448733413, 0.9945540614134305, 0.9968063301396635,
               0.9995694379207766, 0.9965325936199723, 1.0, 0.9789253758459372, 0.9928377776878566, 0.9985719210124587,
               0.9965552436846135, 0.9924211720839751, 0.9886627049353057, 0.9923065963812508])
np.savez('Zclassfiction_acc_100.npz',Acc1)
np.savez('Zdetection_acc_100.npz',Det1)

#给不同类别加权
weights=[1/x for x in Acc]
SUM=sum(weights)
weights=[x/SUM for x in weights]
print(weights)