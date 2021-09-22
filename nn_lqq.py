from io import BufferedIOBase
import torch as t

class TDNN(t.nn.Module):
    def __init__(self, kernel,in_dim,out_dim):
        super(TDNN, self).__init__()
        self.kernel = kernel
        self.in_dim = in_dim
        self.out_dim = out_dim

        dil = 2
        # conv_size = 9 - self.kernel + 1
        self.conv1 = t.nn.Conv1d(self.in_dim ,self.out_dim ,self.kernel ,dilation = dil ,bias=False)
        self.conv2 = t.nn.Conv1d(self.out_dim ,2*self.out_dim ,2 ,dilation = 1 ,bias=False)
        self.conv3 = t.nn.Conv1d(2*self.out_dim ,1 ,1 ,dilation = 1 ,bias=False)
        self.GELU = t.nn.GELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = self.GELU(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        return x
