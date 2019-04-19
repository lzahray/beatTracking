import numpy as np
from dataFunctions import *
import argparse
from functionsForHMM import *

weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]
parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("hmm_guess_folder")
args = parser.parse_args()
mode = args.mode
hmm_guess_folder = args.hmm_guess_folder
args = parser.parse_args()
featureFolder = "../Features/mel128ChromaCQTWithTargets/"
targetFolder = featureFolder
listOfSongsInOrder = np.arange(1,101)
listOfSongsInOrder = [i for i in listOfSongsInOrder if i not in weirdTimes]
for song in listOfSongsInOrder:
    features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolder, song, 0, 0, targetFolder, mode=mode)
    targetsChord = targetsChord.cpu().numpy()
    targetsChord = getPermutedUsToHMM(targetsChord)
    #np.save("../ChordSoftmaxTextFiles/ChordGroundTruth/"+str(song)+".npy", targetsChord)
    
