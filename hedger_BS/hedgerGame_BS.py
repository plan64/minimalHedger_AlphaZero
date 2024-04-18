from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from random import randrange
from scipy.stats import norm

class HedgerPlan_BS():
    def __init__(self, config):
        self.episode_step = 0
        self.config = config
        self.n_actions = 21
        self.n_samples=60 #number of individual actions per episode
        self.n_episodes = config['reservoir'] #this is used a repository of possible paths
        self.measureSampleEfficiency = config['measureSampleEfficiency']

        self.T = self.n_samples/365
        self.dt = self.T*1/self.n_samples
        self.s0 = 1
        self.sigma = 0.3
        self.strike = 1
        self.transactionCosts = 0.00
        self.lambd = 1.0
        
        #elements of the trinomial model
        self.upFactor = np.exp(self.sigma*np.sqrt(2*self.dt))
        self.downFactor = 1/self.upFactor
        
        tmpUp = np.exp(self.sigma*np.sqrt(0.5*self.dt))
        self.probUp = ((1-1/tmpUp)/(tmpUp-1/tmpUp))**2
        self.probDown = ((tmpUp-1)/(tmpUp-1/tmpUp))**2
        self.probMid = 1-self.probUp-self.probDown
        self.ttTransitions = self.createData()
        self.cumSumTransitions = np.cumsum(self.ttTransitions,axis=1).astype(int)
        self.ttPaths = self.s0*pow(self.upFactor,self.cumSumTransitions)
        self.ttPath = self.ttPaths[0,:]
        self.pricesBS = self.createBSPrices()
        self.conditionalPricesAlongPaths = self.conditionPrices()
        self.conditionalPricePath = self.conditionalPricesAlongPaths[0,:]
        print('Black Scholes-type Hedging with transaction cost')

    def _euro_vanilla_delta(self, S, K, T, r, sigma):
        if(T==0.0):
            delta = 0.0
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1, 0.0, 1.0)
        return delta
    
    def conditionPrices(self):
        condPrices = np.zeros((self.n_episodes,self.n_samples+1))
        for i in range(self.n_episodes):           
            for j in range(self.n_samples+1):
                if(j==0):
                    condPrices[i,0] = self.pricesBS[0,0]
                else:
                    condPrices[i,j] = self.pricesBS[(-1)*self.cumSumTransitions[i,j]+j,j]
        return condPrices
    
    def createBSPrices(self):
        prices = np.zeros((2*self.n_samples+2,self.n_samples+1))
        for i in range(2*self.n_samples-1):
            prices[i,self.n_samples] = max(0.0,self.s0*pow(self.upFactor,self.n_samples)*pow(self.downFactor,i)-self.strike)
          
        for j in range(self.n_samples):
            for i in range(2*self.n_samples - 2*(j)-1):
                prices[i,self.n_samples-(j+1)] = self.probDown*prices[i+2,self.n_samples-j] + self.probMid* prices[i+1,self.n_samples-j]+ self.probUp*prices[i,self.n_samples-j]
        return prices

    def createData(self):
        ttTransitions = np.zeros((self.n_episodes,self.n_samples+1)) 
        for it in range(self.n_episodes):
            tmp = np.random.choice([1,0,-1], self.n_samples, [self.probUp,self.probMid,self.probDown])
            ttTransitions[it,1:self.n_samples+1] = tmp#a zero is chosen by default in the first entry of the path.
        return ttTransitions
    
    def getEuroVanillaDelta(self, state):
        return self._euro_vanilla_delta(state[0,3],self.strike,state[0,4]/365,0,self.sigma)
    
    def getInitState(self):
        q = randrange(self.n_episodes)
        self.ttPath = self.ttPaths[q,:]
        self.conditionalPricePath = self.conditionalPricesAlongPaths[q,:]
        
        #state 0: counter
        #state 1: number of shares held
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        state = (0,
                 int(0),
                 self.conditionalPricePath[0],
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
        newHoldings = (action - (self.n_actions - 1 ) / 2 ) * 2 / (self.n_actions - 1 )
  
        #new state. State corresponding to t is updated just before t+1.
        #state 0: counter
        #state 1: number of shares held, it is approximately between 0 and 1 due to the similarity with the bsm model.
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        if self.measureSampleEfficiency:
            marketTransition = self.ttPath[int(state[0,0])+1]
        else:
            marketTransition = state[0,3]*self.upFactor**np.random.choice([1,0,-1], 1, p=[self.probUp,self.probMid,self.probDown])[0]
       
        stateNew = (state[0,0]+1,
                 newHoldings,
                 state[0,2] - state[0,3] * (- state[0,1] + newHoldings),
                 marketTransition,
                 state[0,4]-self.dt*365
                 )
        tmp = abs(self.stateValue(state)-self.conditionalPricePath[int(state[0,0])])
        tmp = tmp if tmp < 1 else 1
        cost = (state[2,2] + self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings)) + tmp,
                 state[2,2] + self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings)) + tmp,
                 state[2,2] + self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings)) + tmp,
                 state[2,2] + self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings)) + tmp,
                 state[2,2] + self.transactionCosts * abs(state[0,3] * (- state[0,1] + newHoldings)) + tmp
                 )
        return (np.array([stateNew,stateNew,cost,stateNew,stateNew]), dummy)

    def stateValue(self, state):
        return state[0,2] + state[0,3] * state[0,1]
        # P_{t+1} = cashNew + S_{t+1}*newholding = cashOld - S_t*actOld + S_{t+1}*(actOld + holdingOld)
        # = cashOld + S_t*holdingOld + Delta S_{t+1} (actOld + holdingsOld)
        # = P_t + Delta S_{t+1} holdingsNew
    
    def getGameEnded(self, state, dummy):
        ts = state[0,:]
        if(ts[0]<self.n_samples):
            return 0
        else:
            tmp = 1/self.s0**2*(self.stateValue(state)-max(0.0,state[0,3]-self.strike))**2
            tmp = 20*((tmp if tmp < 1 else 1) + state[2,2])
            tmp = tmp if tmp<10 else 10
            return -tmp/5+1

    def getSymmetries(self, state, pi):
        return [(state,pi)]
    
    def getNormalizations(self):
        #state normalizations chosen as ad-hoc hyperparams for better NN training
        stateNormalizations = (self.n_samples, self.n_samples, self.conditionalPricePath[0], self.s0, self.T*365)
        if abs(self.transactionCosts) < 0.00001:#dummy
            costNormalizations = (self.n_samples,self.n_samples,self.n_samples,self.n_samples,self.n_samples)
        else:
            costNormalizations = (self.n_samples + self.transactionCosts * self.s0 * self.n_samples,
                                  self.n_samples + self.transactionCosts * self.s0 * self.n_samples,
                                  self.n_samples + self.transactionCosts * self.s0 * self.n_samples,
                                  self.n_samples + self.transactionCosts * self.s0 * self.n_samples,
                                  self.n_samples + self.transactionCosts * self.s0 * self.n_samples)
        return np.array([stateNormalizations,stateNormalizations,costNormalizations,stateNormalizations,stateNormalizations])
    
    def stringRepresentation(self, state):
        return str(state[0,0]) + '_' + str(round(state[0,1],2)) + '_' + str(round(state[0,2],2)) + '_' + str(round(state[0,3],2))