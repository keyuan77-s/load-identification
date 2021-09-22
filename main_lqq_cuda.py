'''
TDNN for Mackey-Glass Time Series Prediction
Author: Brevin Tilmon
'''

from nn_lqq import TDNN
import torch as t
import scipy.io as io
import numpy as np
import argparse

import pandas as pd
import datetime


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

device = t.device("cuda" if t.cuda.is_available() else "cpu")
#print(t.cuda.is_available())
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())
    
def main(args):
    
    #data = pd.read_excel('data/two20210427/response.xlsx')#excel即使修改了后缀名为csv也不是csv文件，按照pd.read_csv来读取会报错,转一下数据格式
    #data1=data.iloc[:22938,1:]
    #data1.to_csv('data/two20210427/response.csv', index=False, encoding='utf-8')
    #data2 = data.iloc[22939:,1:]
    #data2.to_csv('data/two20210427/response-test.csv', index=False, encoding='utf-8')

    #label= pd.read_excel('data/two20210427/stimulate.xlsx')
    #label1= label.iloc[:22938,1:]
    #label1.to_csv('data/two20210427/stimulate.csv', index=False, encoding='utf-8')
    #label2 = label.iloc[22939:,1:]
    #label2.to_csv('data/two20210427/stimulate-test.csv', index=False, encoding='utf-8')
    
    data = pd.read_csv('data/two20210427/response.csv',header=None)
    datat = pd.read_csv('data/two20210427/response-test.csv',header=None)
    label = pd.read_csv('data/two20210427/stimulate.csv',header=None)
    labelt = pd.read_csv('data/two20210427/stimulate-test.csv',header=None)
    # v为需要查看数据类型的变量
    print(type(data))
    # init data
    #train = io.loadmat('data/train_tau_30.mat')
    #train = t.from_numpy(np.array(list(train.items()))[3,1]).float()
    #test = io.loadmat('data/test_tau_30.mat')
    #test = t.from_numpy(np.array(list(test.items()))[3,1]).float()

    data = t.from_numpy(np.array(data)).float().to(device)#转化成张量
    datat = t.from_numpy(np.array(datat)).float().to(device)#转化成张量
    label = t.from_numpy(np.array(label)).float().to(device)#转化成张量
    labelt = t.from_numpy(np.array(labelt)).float().to(device)#转化成张量

    # normalize data
    #train = normalize(train)
    #test = normalize(test)
    size = data.size() 
    sizet = datat.size()
   # size1 = label.size()   
    
    # init model
    epochs = 100
    #taps = 8#这是什么参数？
    net = TDNN(9).to(device)

    # optimization
    mse_loss = t.nn.MSELoss().to(device)
    optim = t.optim.Adam(net.parameters(), lr=1e-4)
    
    for e in range(epochs):
        net.train()
        for i in range(size[0]-51):#999-9-2=988
            #print(i)

            inp = data[i:i+51].unsqueeze(0).to(device)# 10个数据作为一个样本 train 1000*1 988:997 实际上是 989-998unsqueeze是维度扩充 在第1维扩充维度
            #print(inp)
           # label = label[:i].unsqueeze(0).view(1,1,-1)#这应该是对应的标签 label是999 实际上是1000
            lab = label[i].unsqueeze(0).to(device)
            #print(label)
            optim.zero_grad()

            y = net(inp)
            #print(y)
            error = mse_loss(y,lab).to(device)#y是预测值，label是真值
            error.backward()
            optim.step()
            
    net.eval()
    for i in range(sizet[0]-51):
        inp = datat[i:i+51].unsqueeze(0).to(device)#view 相当于reshape -1代表动态
        labt = labelt[i].unsqueeze(0).to(device)
        with t.no_grad():
            y = net(inp)
            
            #print(y)#只需要把y保存起来，拟合一个线和原本的label对比一下
            df = pd.DataFrame(y.cpu().numpy().tolist())
            
            df.to_csv('result/two_relu_'+str(nowTime)+'.csv', mode='a', header=False)
        error = mse_loss(y, labt).to(device)
    print(error)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int,default=18)
    args = parser.parse_args()
    
    main(args)
