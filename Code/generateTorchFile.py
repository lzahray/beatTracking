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


featuresAndGT = []
featureFolder = "../Features/moreFeatures"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
toSaveFolder = "../Features/moreFeaturesTorch"

listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]

for songNumber in listOfSongsInOrder:
    features, targetsBeats, targetsChords = getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = True, songNumber=songNumber)[0]
    torch.save(features, toSaveFolder+"/features"+str(songNumber))
    torch.save(targetsBeats, toSaveFolder+"/beats"+str(songNumber))
    torch.save(targetsChords, toSaveFolder+"/chords"+str(songNumber))

