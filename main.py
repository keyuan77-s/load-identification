'''
TDNN for Mackey-Glass Time Series Prediction
Author: Brevin Tilmon
'''

from nn import TDNN
import torch as t
import scipy.io as io
import numpy as np
import argparse


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())
    
def main(args):

    # init data
    train = io.loadmat('data/train_tau_30.mat')#torch.size=1000*1 dtypr torch.float 是一个字典
    #train = t.from_numpy(np.array(list(train.items()))[3,1]).float()#tensor [[]]2维
    test = io.loadmat('data/test_tau_30.mat')
    test = t.from_numpy(np.array(list(test.items()))[3,1]).float()

    # normalize data
    #train = normalize(train)
    #test = normalize(test)
    size = train.size()    
    
    # init model
    epochs = 100
    taps = 9
    net = TDNN(3, args.hidden_size, taps)

    # optimization
    mse_loss = t.nn.MSELoss()
    optim = t.optim.Adam(net.parameters(), lr=1e-4)
    
    for e in range(epochs):
        net.train()
        for i in range(size[0]-taps-2):
            inp = train[i:i+taps+1].unsqueeze(0).view(1,1,-1)
            label = train[i+taps+2].unsqueeze(1).view(1,1,-1)
            optim.zero_grad()

            y = net(inp)
            error = mse_loss(y,label)
            error.backward()
            optim.step()
            
        net.eval()
        for i in range(size[0]-taps-2):
            inp = train[i:i+taps+1].unsqueeze(0).view(1,1,-1)
            label = train[i+taps+2].unsqueeze(1).view(1,1,-1)
            with t.no_grad():
                y = net(inp)
            error = mse_loss(y, label)
        print(error)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int)
    args = parser.parse_args()
    
    main(args)
