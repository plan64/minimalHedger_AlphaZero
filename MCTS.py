import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    #Monte Carlo Tree Search
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  
        self.Nsa = {} 
        self.Ns = {}  
        self.Ps = {}  

        self.Es = {} 

    def getActionProb(self, state, temp=1):
        for i in range(self.args['numMCTSSims']):
            self.search(state)
        s = self.game.stringRepresentation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            print('state collision')
            print('s: ')
            print(s)
            print(state)
            print(self.Nsa)
            print(self.Ps)
            print(self.Qsa)
        probs = [x / counts_sum for x in counts]
        return probs


    def search(self, state):
        s = self.game.stringRepresentation(state)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(state, 1)
        if self.Es[s] != 0:
            return self.Es[s]
            
        
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(state)
            self.Ps[s] = self.Ps[s]
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else: 
                log.error('probs sum to zero')
            self.Ns[s] = 0
            return v
        
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.game.getActionSize()):
            if True:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['nnWeight'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args['nnWeight'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        next_s, next_player = self.game.getNextState(state, 1, a)
        
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
