import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class HedgerNNet(nn.Module):
    def __init__(self, game, args):
        self.action_size = game.getActionSize()
        self.args = args

        super(HedgerNNet, self).__init__()
        self.lin1 = nn.Linear(25, self.args['num_channels'])
        self.lin2 = nn.Linear(self.args['num_channels'], self.args['num_channels'])
        self.lin3 = nn.Linear(self.args['num_channels'], self.args['num_channels'])
        self.lin4 = nn.Linear(self.args['num_channels'], self.args['num_channels'])

        self.bn1 = nn.BatchNorm1d(self.args['num_channels'])
        self.bn2 = nn.BatchNorm1d(self.args['num_channels'])
        self.bn3 = nn.BatchNorm1d(self.args['num_channels'])
        self.bn4 = nn.BatchNorm1d(self.args['num_channels'])

        self.fc1 = nn.Linear(self.args['num_channels'], self.args['num_channels'])
        self.fc_bn1 = nn.BatchNorm1d(self.args['num_channels'])

        self.fc2 = nn.Linear(self.args['num_channels'], self.args['num_channels'])
        self.fc_bn2 = nn.BatchNorm1d(self.args['num_channels'])

        self.fc3 = nn.Linear(self.args['num_channels'], self.action_size)

        self.fc4 = nn.Linear(self.args['num_channels'], 1)

    def forward(self, s):
        s = F.relu(self.bn1(self.lin1(s)))                         
        s = F.relu(self.bn2(self.lin2(s)))                          
        s = F.relu(self.bn3(self.lin3(s)))                         
        s = F.relu(self.bn4(self.lin4(s)))                          

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args['dropout'], training=self.training)  
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args['dropout'], training=self.training)  

        pi = self.fc3(s)                                                                      
        v = self.fc4(s)                                                                        

        return F.log_softmax(pi, dim=1), torch.tanh(v)
