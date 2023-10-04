import logging
import os
import sys
from pathlib import Path
import coloredlogs
import yaml

from Trainer import Trainer
from hedger_TV.neuralNet.trainerNeuralNet_simpleFF import NNetWrapper as trainerNeuralNet
from hedger_TV.hedgerGame_TV_heston import HedgerPlan_TV_heston


def main():
    # Write actual config on file
    with open(os.path.join(CONFIG['saveCheckpointsFolder'], '_config_TV_heston.txt'), "w") as con:
        con.write(str(CONFIG))

    scenario = HedgerPlan_TV_heston(CONFIG['reservoir'])

    # Setup neural network
    nnet = trainerNeuralNet(scenario, CONFIG['nnArgs'])
    print(CONFIG)
    if CONFIG['loadModelFile']:
        log.info('Loading model "%s/%s" ', CONFIG['loadModelFile'])
        nnet.load_checkpoint(CONFIG['loadModelFile'], )

    trainer = Trainer(scenario, nnet, CONFIG)

    if CONFIG['loadCheckpointFile']:
        log.info('Loading data "%s/%s" ', CONFIG['loadModelFile'])
        trainer.loadTrainExamples()

    log.info('Learning.......')
    trainer.learn()


if __name__ == "__main__":
    # Load configuration
    config_file_path = Path(os.getcwd()) / r'ZZZ_Config/config_TV_heston.yaml'
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    print(f"Reading config from: '{config_file_path}'")
    with open(config_file_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Setup checkpoint folder
    checkpoints_folder = Path(CONFIG['saveCheckpointsFolder'])
    if not checkpoints_folder.exists():
        print(f"Checkpoint Directory '{checkpoints_folder}' does not exist. Creating it!")
        checkpoints_folder.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file_path = checkpoints_folder / '_log.txt'
    print(f"Logging to: {log_file_path}")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    log = logging.getLogger(__name__)
    coloredlogs.install(level='INFO')

    # Run main
    main()
