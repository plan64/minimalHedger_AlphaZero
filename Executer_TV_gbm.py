import logging
import os
import sys
from pathlib import Path
import coloredlogs
import yaml

from scipy.stats import norm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Be careful to import the right model
from Trainer import Trainer
from hedger_TV.neuralNet.trainerNeuralNet_simpleFF import NNetWrapper as trainerNeuralNet
from hedger_TV.hedgerGame_TV_pureGBMPaths import HedgerPlan_TV_gbm
from MCTS import MCTS


n_testPaths = 20
#Set below the path of the config file to be used for testing
config_file_path = Path(os.getcwd()) / r'ZZZ_Config/config_TV_gbm.yaml'


if len(sys.argv) > 1:
    config_file_path = sys.argv[1]
print(f"Reading config from: '{config_file_path}'")
with open(config_file_path, "r") as f:
    CONFIG = yaml.safe_load(f)

scenario = HedgerPlan_TV_gbm(CONFIG)
nnet = trainerNeuralNet(scenario, CONFIG['nnArgs'])
nnet.load_checkpoint(CONFIG['saveCheckpointsFolder'], filename='best.pth.tar')

mcts_actions_h = np.zeros((n_testPaths, scenario.n_samples))
bs_deltas_h = np.zeros((n_testPaths, scenario.n_samples))
stockPrices_h = np.zeros((n_testPaths, scenario.n_samples+1))
optionPrices_h = np.zeros((n_testPaths, scenario.n_samples+1))
hedgePortfolioValues_h = np.zeros((n_testPaths, scenario.n_samples+1))
bsmHedgePortfolioValues_h = np.zeros((n_testPaths, scenario.n_samples+1))
acc_transactionCosts = np.zeros(n_testPaths)
bsm_transactionCosts = np.zeros(n_testPaths)
#Collect data for tests
for i in range(n_testPaths):
    print('I am at ' +str(i))
    state = scenario.getInitState()
    player = lambda x: np.argmax(nnet.predict(x)[0])
    counter = int(0)
    acc_bsm_cost = 0.0
    while scenario.getGameEnded(state, 1) == 0:
        currentAction = player(state)
        hedgePortfolioValues_h[i, counter] = scenario.stateValue(state)
        bs_deltas_h[i, counter] = scenario.getEuroVanillaDelta(state)
        stockPrices_h[i, counter] = state[0,3]
        state, _ = scenario.getNextState(state, 1, currentAction)
        mcts_actions_h[i, counter] =  (currentAction - (scenario.n_actions - 1 ) / 2 ) * 2 / (scenario.n_actions - 1 )
        
        #optionPrices_h[i, counter] = scenario.priceBS[counter]
        if counter == 0:
            bsmHedgePortfolioValues_h[i,counter] = hedgePortfolioValues_h[i, counter]
            acc_bsm_cost += scenario.transactionCosts*abs((bs_deltas_h[i,counter]-bs_deltas_h[i,counter-1]) * stockPrices_h[i,counter])
        else :
            acc_bsm_cost += scenario.transactionCosts*abs((bs_deltas_h[i,counter]-bs_deltas_h[i,counter-1]) * stockPrices_h[i,counter])
            bsmHedgePortfolioValues_h[i,counter] = bsmHedgePortfolioValues_h[i,counter-1] + (stockPrices_h[i,counter]-stockPrices_h[i,counter-1]) * bs_deltas_h[i,counter-1]
        
        counter += 1
    #one final time because of while loop
    hedgePortfolioValues_h[i, counter] = scenario.stateValue(state)
    stockPrices_h[i, counter] = state[0,3]
    optionPrices_h[i, counter] = max(0,state[0,3]-scenario.strike)
    bsmHedgePortfolioValues_h[i,counter] = bsmHedgePortfolioValues_h[i,counter-1] + (stockPrices_h[i,counter]-stockPrices_h[i,counter-1]) * bs_deltas_h[i,counter-1]
    #acc_bsm_cost += abs((bs_deltas_h[i,counter]-bs_deltas_h[i,counter-1]) * stockPrices_h[i,counter-1])
    acc_transactionCosts[i] = state[2,2]
    bsm_transactionCosts[i] = acc_bsm_cost
    i += 1
    print(scenario.getGameEnded(state, 1))


#Create a scatter plot of terminal variance histograms
if(True):
    endStockPrices = 100*stockPrices_h[:,-1]
    endPortfolioValues = 100*(hedgePortfolioValues_h[:,-1])
    endOptionPrices = 100*optionPrices_h[:,-1] 
    bsmEndPortfolioValues = 100*(bsmHedgePortfolioValues_h[:,-1] - bsm_transactionCosts)

    plt.scatter(endStockPrices, endPortfolioValues, color= 'b', s = 3, label = 'MCTS')
    plt.scatter(endStockPrices, endOptionPrices, color= 'r', s = 3, label = 'Optimal')
    plt.scatter(endStockPrices, bsmEndPortfolioValues, color= 'g', s = 3, label = 'BSM $\delta$-Hedger')
    plt.xlim(60,150)
    plt.ylim(-20,50)
    plt.legend(loc='upper left', prop ={'size':10})
    figure = plt.gcf()
    figure.set_size_inches(8, 8)
    plt.show()

    
    sum_mcts = 0.0
    sum_bsm = 0.0
    for i in range(endStockPrices.shape[0]):
        sum_mcts += 1/10000*(endPortfolioValues[i]-endOptionPrices[i])**2
        sum_bsm += 1/10000*(bsmEndPortfolioValues[i]-endOptionPrices[i])**2
    
    print('mcts average sum of squares ' + str(sum_mcts/endStockPrices.shape[0]))
    print('bsm average sum of squares ' + str(sum_bsm/endStockPrices.shape[0]))

    
if(False): #plot transaction cost hists

    bins = np.linspace(0, 0.05, 100)

    plt.hist(bsm_transactionCosts, label = 'BSM Costs',bins = bins, alpha=0.5, density=True)
    plt.hist(acc_transactionCosts, label = 'MCTS Costs', bins=bins,alpha=0.5, density=True)
    plt.legend()
    
    
    plt.legend(loc='upper right', prop ={'size':10})
    figure = plt.gcf()
    figure.set_size_inches(8, 8)
    plt.show()
    
    print('averge mcts costs ' + str(acc_transactionCosts.sum()/endStockPrices.shape[0]))
    print('averge bsm costs ' + str(bsm_transactionCosts.sum()/endStockPrices.shape[0]))