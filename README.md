# Hedging Financial Derivative Contracts using neural MCTS

Main components of research code from the article [Hedging of Financial Derivative Contracts via Monte Carlo Tree Search](https://arxiv.org/abs/2102.06274)

* This code is a derivative of the repo: https://github.com/suragnair/alpha-zero-general
* This code provides a neural Monte Carlo Tree Search based agent for hedging of Financial Derivative Contracts.
* Various hedging targets are implemented. In all cases rewards are granted only at maturity. Total terminal quadratic deviation target is assumed in hedger_TV. Minimization of immediate quadratic deviation at each individual time step is assumed in hedger_BS.
* Three market models are implemented: Trinomial, GBM, Heston

## Usage
### Training
 * congig.yaml contains training parameters. Specify folder to store outputs of learning process.
 * Execute main.py to start the training process, either from IDE or from Terminal specifying path to config.yaml.
### Execution
 * In Executer.py specify the location of trained neural networks.
 * Execute Executer.py and consider hedging histograms.

## Warning

This is research code that comes with no liability. 
No liability is accepted for any loss or damage arising out of the use of all or any part of this material or reliance upon any information contained herein.

## Bugs

Known bugs are listed below. Please report in case you observe new bugs.

* This is research, not production code. There are many hacks, bugs and experimental components.
* The code contains a bug that affects the construction of the planning tree under the scaling of underlying models by a constant. For experiments it is currently recommended to start underlying prices at 1.0 (e.g. by scaling down the value of the stock and strike).

Author: Oleg Szehr, IDSIA, Switzerland, oleg.szehr@idsia.ch