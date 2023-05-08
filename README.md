# INFO 550 Final Project

Deep Q-learning Network is programmed and tested using mediumClassic layout.

Approximate Q-learning Agent can be run by the following command:

  $ python.exe pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 200 -n 250 -l mediumClassic --frameTime 0.0

DQN Agent can be run by the following command:

  $ python.exe pacman.py -p PacmanDQN -x 8000 -n 8200 -l mediumClassic --frameTime 0.0

Pretrained network data is saved as "DQN_policy.pt" and "DQN_target.pt"
Navigate to pacmanAgent.py line 83
To use pre-trained DQN, change "model_trained" to True.
To train DQN, change "model_trained" to False.

Results are displaied in FinalProjectDocument.pdf
