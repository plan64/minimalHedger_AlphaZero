import os
import sys
import time
import logging

import numpy as np
from tqdm import tqdm

from hedger_TV.neuralNet.hedgerNeuralNet_simpleFF import HedgerNNet

sys.path.append('../../')

import torch
import torch.optim as optim

log = logging.getLogger(__name__)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class MinimalDataset(torch.utils.data.Dataset):
    def __init__(self, examples, normalizations):
        self.trainDataNormalized = torch.FloatTensor(np.array([(example[0]/normalizations).flatten() for example in examples]).astype(np.float64))
        self.pis = torch.FloatTensor(np.array([example[1] for example in examples]).astype(np.float64))
        self.vs = torch.FloatTensor(np.array([example[2] for example in examples]).astype(np.float64))
        
    def __len__(self):
        return self.trainDataNormalized.shape[0]

    def __getitem__(self, idx):
        states = self.trainDataNormalized[idx, :]
        pis = self.pis[idx, :]
        vs = self.vs[idx]
        sample = {'states' : states, 'target_pis' : pis, 'target_vs' : vs}
        return sample

class NNetWrapper():
    def __init__(self, game, args):
        self.nnet = HedgerNNet(game, args)
        self.action_size = game.getActionSize()
        self.normalizations = game.getNormalizations()
        self.args = args
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.nnet.cuda()

    def train(self, examples):
        log.info('Training NN')
        
        minimalDataset =  MinimalDataset(examples, self.normalizations)
        
        dataLoader = torch.utils.data.DataLoader(minimalDataset, batch_size = self.args['batch_size'], shuffle=True, drop_last=True)
        optimizer = optim.Adam(self.nnet.parameters())
        for epoch in range(self.args['epochs']):
            log.info('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            
            t = tqdm(dataLoader, desc='Training Net')
            for i, batch in enumerate(t):
                out_pi, out_v = self.nnet(batch['states'])
                l_pi = self.loss_pi(batch['target_pis'], out_pi)
                l_v = self.loss_v(batch['target_vs'], out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), batch['states'].size(0))
                v_losses.update(l_v.item(), batch['states'].size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                #log.info('Loss_pi ' + str(pi_losses) + '/' + 'Loss_v ' + str(v_losses))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, state):
        start = time.time()
        state = torch.FloatTensor((state/self.normalizations).flatten().astype(np.float64)).unsqueeze(0)
        if self.cuda: state = state.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(state)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='save_folder', filename=None):
        if filename:
            filepath = os.path.join(folder, filename)
        else:
            filepath = folder
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='load_folder', filename=None):
        if filename:
            filepath = os.path.join(folder, filename)
        else:
            filepath = folder
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
