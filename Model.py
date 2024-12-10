import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):


    def __init__(self,in_size,out_size):
        super().__init__()

        self.ffn1=nn.Linear(in_features=in_size,out_features=256)
        self.ffn2=nn.Linear(in_features=256,out_features=128)
        self.ffn3=nn.Linear(in_features=128,out_features=32)
        self.ffn4=nn.Linear(in_features=32,out_features=out_size)


    def forward(self,x):

        x=self.ffn1(x)
        x=F.relu(x)

        x = self.ffn2(x)
        x = F.relu(x)

        x = self.ffn3(x)
        x = F.relu(x)

        x = self.ffn4(x)


        return x

