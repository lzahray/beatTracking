import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing
from dataFunctions import *
import argparse
import time


folderPath = "../Features/mel128ChromaCQTWithTargets"
featureFolder = "../Features/mel128ChromaCQT"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]
featuresAndGT = getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = True)
print("finished gettting stuff")
for i in range(len(featuresAndGT)):
    np.save(folderPath+"/"+str(listOfSongsInOrder[i])+"features", np.array(featuresAndGT[i][0]))
    np.save(folderPath+"/"+str(listOfSongsInOrder[i])+"beatTargets", np.array(featuresAndGT[i][1]))
    np.save(folderPath+"/"+str(listOfSongsInOrder[i])+"chordTargets", np.array(featuresAndGT[i][2]))

