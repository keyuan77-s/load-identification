'''
TDNN for Mackey-Glass Time Series Prediction
Author: Brevin Tilmon
'''
import sys
import os

from nn_lqq import TDNN
from torchsummary import summary
import torch as t
import scipy.io as io
import numpy as np
import argparse

import pandas as pd
import datetime
import time

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

#print(t.cuda.is_available())
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())
    
def main(): 
    #数据读取
    File_Name = 'data/20210915/新做的/dila_conv/2048hz/100%'
    data = pd.read_csv(File_Name + '/response.csv',header=None)
    datat = pd.read_csv(File_Name + '/response-test.csv',header=None)
    label = pd.read_csv(File_Name + '/stimulate.csv',header=None)
    labelt = pd.read_csv(File_Name + '/stimulate-test.csv',header=None)
    print(type(data))
    #转化成张量
    data = t.from_numpy(np.array(data)).float()
    datat = t.from_numpy(np.array(datat)).float()
    label = t.from_numpy(np.array(label)).float()
    labelt = t.from_numpy(np.array(labelt)).float()
    #尺寸
    size = data.size() 
    sizet = datat.size()
   # size1 = label.size()   

#****************************************************************************************************#
# 训练代码    
    # init model+训练参数
    epochs = 100
    in_dim = 32
    out_dim = 2*in_dim
    kernel = 3
    net = TDNN(kernel,in_dim,out_dim)
    # optimization
    mse_loss = t.nn.MSELoss()
    optim = t.optim.Adam(net.parameters(), lr=1e-4)
    trainbegin = time.perf_counter()#代码效率计算开始时间
    print('Start_Time:',nowTime)#开始时间
    for e in range(epochs):
        net.train()
        sum_loss = 0
        for i in range(size[0]-in_dim):#999-9-2=988
            #print(i)
            inp = data[i:i+in_dim].unsqueeze(0)# 10个数据作为一个样本 train 1000*1 988:997 实际上是 989-998unsqueeze是维度扩充 在第1维扩充维度
           # inp = inp.permute(0,2,1)
           # label = label[:i].unsqueeze(0).view(1,1,-1)#这应该是对应的标签 label是999 实际上是1000
            lab = label[i].unsqueeze(0)
            #print(label)
            optim.zero_grad()
            #train
            y = net(inp)
            error = mse_loss(y,lab)#y是预测值，label是真值
            error.backward()
            optim.step()
            sum_loss  += error
            
        print('epoch: {}, loss: {:0.6f}'.format(e,sum_loss/(size[0]-in_dim)))
    trainend = time.perf_counter()#代码效率计算结束时间
    print("traintim:",trainend-trainbegin)        
    
#**********************************************************************************************************#
  # 推理代码
    net.eval()
    begin = time.perf_counter()
    for i in range(sizet[0]-in_dim):
        inp = datat[i:i+in_dim].unsqueeze(0)#view 相当于reshape -1代表动态
        labt = labelt[i].unsqueeze(0)
        with t.no_grad():
            y = net(inp)
            
            #print(y)#只需要把y保存起来，拟合一个线和原本的label对比一下
            df = pd.DataFrame(y.numpy().tolist())
            
            df.to_csv('data/20210915/新做的/dila_conv/2048hz/100%/run_2048Hz_100%_100_'+str(nowTime)+'.csv', mode='a', header=None,index = None)
        error = mse_loss(y, labt)
    end = time.perf_counter()
    print("inferencetim:",end-begin)
    print('End_Time:',nowTime)

#**********************************************************************************************************#

if __name__ == "__main__":  
    main()
