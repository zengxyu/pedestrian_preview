import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __int__(self,statedim,actiondim,policymodel="DP"):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(statedim,32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4800,512)
        self.fc2 = nn.Linear(512, 256)
        self.action_mean = nn.Linear(256, actiondim)
        self.action_std = nn.Linear(256,actiondim)
        self.flatten = nn.Flatten()

    def forward(self,x,dis):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        flatten = F.relu(self.flatten(c3))
        f1 = F.relu(self.fc1(flatten))
        f2 = F.relu(self.fc2(f1))
        action_mean = F.relu(self.action_mean(f2))
        action_std = F.relu(self.action_std(f2))

        return action_mean,action_std


