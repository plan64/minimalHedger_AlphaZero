from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from random import randrange
from scipy.stats import norm

class HedgerPlan_TV_heston():
    def __init__(self, n_episodes):
        self.episode_step = 0
        #GBM Parameters
        self.n_actions = 21
        self.n_samples=60 #number of individual actions per episode
        self.n_episodes = n_episodes #this is used a repository of possible paths
        
        #some params of the heston model
        self.T = self.n_samples/365
        self.dt = self.T*1/self.n_samples#the convention used throughout is that we must devide through 365 to get annualized results
        self.s0 = 1.00
        self.v0  = 0.04
        self.sigma = 2
        self.mu = 0.0
        self.strike = 1.00
        self.rho = -0.7 #correl prices and vars
        self.kappa = 1 # rate of mean reversion
        self.theta = 0.04        # long-term mean of variance 
        
        self.transactionCosts = 0.00
        self.lambd = 1.0
        self.ttmData = self._ttM()[0,:]
        self.ttPaths, self.volaPaths = self._hestonPaths()
        self.gbmPaths = self._gbm()
        self.ttPath = self.ttPaths[0,:]
        self.volaPath = self.volaPaths[0,:]
        self.pricesBS = self._createBSPrices()
        self.priceBS = self.pricesBS[0,:]
        print('Terminal Variance Hedging with Heston model and transaction cost')
    
    def _hestonPaths(self):   
        S = np.full(shape=(self.n_samples+1,self.n_episodes), fill_value=self.s0)
        v = np.full(shape=(self.n_samples+1,self.n_episodes), fill_value=self.v0)
        Z = np.random.multivariate_normal(np.array([0,0]), np.array([[1,self.rho],[self.rho,1]]), (self.n_samples,self.n_episodes))
        for i in range(1,self.n_samples+1):
            S[i] = S[i-1] * np.exp( (self.mu - 0.5*v[i-1])*self.dt + np.sqrt(v[i-1] * self.dt) * Z[i-1,:,0] )
            v[i] = np.maximum(v[i-1] + self.kappa*(self.theta-v[i-1])*self.dt + self.sigma*np.sqrt(v[i-1]*self.dt)*Z[i-1,:,1],0)
        
        return np.asarray(S).T, np.asarray(v).T
        
    def _gbm(self):
        mean = (self.mu-0.5*self.sigma**2)*self.dt
        s = self.s0*np.ones((self.n_episodes, self.n_samples+1))
        bm = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1, (self.n_episodes, self.n_samples))
        s[:, 1:s.shape[1]] = self.s0*np.exp(np.cumsum(mean+bm, 1))
        return s
    
    def _createBSPrices(self):
        bsPricesAllEpisodes = np.zeros((self.ttPaths.shape[0],self.ttPaths.shape[1]))
        
        for it_c in range(self.ttPaths.shape[0]):
            for it_r in list(range(self.ttPaths.shape[1])):
                bsPricesAllEpisodes[it_c,it_r] = self._euro_vanilla_call(self.ttPaths[it_c,it_r], self.strike, self.ttmData[it_r], 0.0, self.sigma)
        return bsPricesAllEpisodes
    
    def _ttM(self):
        tmp=np.linspace(0,(self.n_samples)*self.dt,self.n_samples+1)
        return self.n_samples*self.dt*np.ones((self.n_episodes,self.n_samples+1)) - np.tile(tmp,(self.n_episodes,1))
    
    def _euro_vanilla_call(self, S, K, T, r, sigma):
        if(abs(T-0.0)<0.0000001):
            call = max(0.0, S-K)
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            call = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
        return call
    
    def _euro_vanilla_delta(self, S, K, T, r, sigma):
        if(T==0.0):
            delta = 0.0
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1, 0.0, 1.0)
        return delta
    
    def getEuroVanillaDelta(self, state):
        return self._euro_vanilla_delta(state[0,3],self.strike,state[0,4]/365,0,self.sigma)
    
    def getInitState(self):
        q = randrange(self.n_episodes)
        self.ttPath = self.ttPaths[q,:]
        self.volaPath = self.volaPaths[q,:]
        self.priceBS = self.pricesBS[q,:]
        
        #state 0: vola level
        #state 1: number of shares held
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        state = (self.v0,
                 int(0),
                 0.021,
                 self.s0,
                 self.T*365
                 )
        cost = np.zeros(5) + 0.0001
        return np.array([state,state,cost,state,state])

    def getSize(self):
        return (5,5)

    def getActionSize(self):
        return self.n_actions

    def getNextState(self, state, dummy, action):
        newHoldings = (action - (self.n_actions - 1 ) / 2 ) * 2 / (self.n_actions - 1 ) #BSM actions: self.getEuroVanillaDelta(state)#
        tmp = self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings))
        #new state. State corresponding to t is updated just before t+1.
        #state 0: vola level
        #state 1: number of shares held
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        counter = int(self.T*365-state[0,4]) + 1
        stateNew = (self.volaPath[counter],
                 newHoldings,
                 state[0,2] - state[0,3] * (- state[0,1] + newHoldings) - tmp,
                 self.ttPath[counter],
                 state[0,4]-self.dt*365
                 )

        cost = (state[2,2]+tmp,
                 state[2,2]+tmp,
                 state[2,2]+tmp,
                 state[2,2]+tmp,
                 state[2,2]+tmp
                 )
        return (np.array([stateNew,stateNew,cost,stateNew,stateNew]), dummy)
  
    def stateValue(self, state):
        return state[0,2] + state[0,3] * state[0,1]
        # P_{t+1} = cashNew + S_{t+1}*newholding = cashOld - S_t*actOld + S_{t+1}*(actOld + holdingOld)
        # = cashOld + S_t*holdingOld + Delta S_{t+1} (actOld + holdingsOld)
        # = P_t + Delta S_{t+1} holdingsNew
    def getGameEnded(self, state, dummy):
        if(state[0,4]>0):
            return 0
        else:
            tmp = 100/self.s0**2*((self.stateValue(state)-max(0.0,state[0,:][3]-self.strike))**2+2/self.lambd*state[2,2])
            if tmp > 10:
                tmp = 10
            return -tmp/5+1
    
    def getSymmetries(self, state, pi):
        return [(state,pi)]
    
    def getNormalizations(self):
        #state normalizations chosen as ad-hoc hyperparams for better NN training
        stateNormalizations = (self.v0, self.n_samples, self.priceBS[0], self.s0, self.T*365)
        if abs(self.transactionCosts) < 0.00001:#dummy
            costNormalizations = (1,1,1,1,1)
        else:
            costNormalizations = (self.transactionCosts * self.s0 * self.n_samples,
                                  self.transactionCosts * self.s0 * self.n_samples,
                                  self.transactionCosts * self.s0 * self.n_samples,
                                  self.transactionCosts * self.s0 * self.n_samples,
                                  self.transactionCosts * self.s0 * self.n_samples)
        return np.array([stateNormalizations,stateNormalizations,costNormalizations,stateNormalizations,stateNormalizations])
    
    def stringRepresentation(self, state):
        return str(state[0,4]) + '_' + str(round(state[0,0],3)) + '_' + str(round(state[0,1],2)) + '_' + str(round(state[0,2],2)) + '_' + str(round(state[0,3],2)) + '_' + str(round(state[2,2],2))
