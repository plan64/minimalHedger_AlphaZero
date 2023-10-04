import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle, randrange

import numpy as np
from tqdm import tqdm

from MCTS import MCTS

log = logging.getLogger(__name__)


class Trainer():
    def __init__(self, game, nnet, config):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, config['nnArgs'])
        self.config = config
        self.mcts = MCTS(self.game, self.nnet, self.config)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.nnFailedLastTime = False
        print('Constructed Trainer for Hedger')

    def executeEpisode(self):
        trainExamples = []
        state = self.game.getInitState()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.config['temp'])

            pi = self.mcts.getActionProb(state, temp=temp)
            action = np.random.choice(len(pi), p=pi)
            trainExamples.append((state, self.curPlayer, pi, action))
            state, self.curPlayer = self.game.getNextState(state, self.curPlayer, action)
            r = self.game.getGameEnded(state, self.curPlayer)

            if r != 0:
                return {self.game.stringRepresentation(x[0]): (x[0], x[2], r, x[3]) for x in trainExamples}

    def learn(self):
        start_time = time.time()
        for i in range(1, self.config['learningCycles'] + 1):
            log.info(f'Starting Iter #{i} at time ' + str(time.time() - start_time))
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = {}
                if not self.nnFailedLastTime:
                    self.mcts = MCTS(self.game, self.nnet, self.config)
                
                before = time.time()
                for _ in tqdm(range(self.config['episodes']), desc="Self Play"):
                    iterationTrainExamples = {**iterationTrainExamples, **self.executeEpisode()}
                after = time.time()
                log.info('samplesEpisodes_' + str(i) + ' ran for ' + str(after-before))
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.config['shortenHistoryAfterIter']:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i - 1)

            trainingData = []
            for e in self.trainExamplesHistory:
                trainingData.extend(list(e.values()))

            shuffle(trainingData)

            self.nnet.save_checkpoint(folder=self.config['saveCheckpointsFolder'], filename='temp.pth.tar')  # new
            self.pnet.load_checkpoint(folder=self.config['saveCheckpointsFolder'], filename='temp.pth.tar')  # past
            self.nnet.train(trainingData)

            log.info('COMPARE AGAINST PAST MODEL')
            
            
            h_l2_distance_oldModel = 0.0
            h_l2_distance_newModel = 0.0
            h_rewards = {'past': [], 'new': [], 'whoWon': []}
            for _ in tqdm(range(self.config['validationCycles']), desc="Validation...."):
                curPlayer = 1
                state = self.game.getInitState()
                nmcts = MCTS(self.game, self.nnet, self.config)
                pmcts = MCTS(self.game, self.pnet, self.config)

                while self.game.getGameEnded(state, curPlayer) == 0:
                    state, curPlayer = self.game.getNextState(state, curPlayer, np.argmax(
                        pmcts.getActionProb(state, temp=0)))
                #h_l2_distance_oldModel+= (self.game.stateValue(state)-self.game.)**2

                result1 = self.game.getGameEnded(state, curPlayer)
                # print('\n Past result ' +str(result1))
                h_rewards['past'].append(result1)

                state = self.game.getInitState()

                while self.game.getGameEnded(state, curPlayer) == 0:
                    state, curPlayer = self.game.getNextState(state, curPlayer, np.argmax(
                        nmcts.getActionProb(state, temp=0)))

                result2 = self.game.getGameEnded(state, curPlayer)
                # print('\n New result ' +str(result2))
                h_rewards['new'].append(result2)

            with open(self.config['saveCheckpointsFolder'] + 'h_rewards_' + str(i), "wb+") as fff:
                Pickler(fff).dump(h_rewards)
            fff.closed

            log.info('NEW/PREV Average ' + str(sum(h_rewards['new']) / self.config['validationCycles']) + '/' + str(
                sum(h_rewards['past']) / self.config['validationCycles']))
            if sum(h_rewards['new']) < self.config['nnUpdateThreshold'] * sum(h_rewards['past']):
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.config['saveCheckpointsFolder'], filename='temp.pth.tar')
                self.nnFailedLastTime = True
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.config['saveCheckpointsFolder'],
                                          filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.config['saveCheckpointsFolder'], filename='best.pth.tar')
                self.nnFailedLastTime = False

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.config['saveCheckpointsFolder']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        examplesFile = self.config['loadCheckpointFile']
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            self.skipFirstSelfPlay = True
