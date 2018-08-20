import numpy as np 
from dataFunctions import getGroundTruthChords, getMoreFeaturesAndGroundTruthDownbeats
import os
#this is a lie what it's really doing for now is making chord target files yo 

# answerFolder = "../../CHORD/RWC_Pop_Chords"
# beatAnswerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
# audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
# featureRefFolder = "../Features/CQT1at10FPS"

# weirdTimes = ['006', '022', '028', '030', '034', '037', '038', '041',  '043', '050', '057', '071', '076', '077', '095']
# listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]

# rangeNum = [1,101] 
# for i in range(rangeNum[0],rangeNum[1]):
#     if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes:
#         numFrames = np.load(featureRefFolder+"/"+str(i)+"features.npy").shape[0]
#         gt = getGroundTruthChords(numFrames, i, answerFolder, hopSize = 4410)
#         np.save("../Features/ChordTargets10FPSBeatles/"+str(i)+"chordTargets.npy", gt)
#     else:
#         print(i, " is weird")

# for song in listOfSongsInOrder:
#     #beats = getMoreFeaturesAndGroundTruthDownbeats(featureRefFolder, beatAnswerFolder, getChords = False)
#     np.save("../Features/ChordTargets10FPSBeatles/"+str(song)+"beatTargets.npy", np.zeros(5)) #

#answerFolder = "/n/sd1/music/Beatles/annotation/chordlab"#
answerFolder = "../../chordlab/TheBeatles"
subAnswerFolders = [answerFolder+"/"+f for f in sorted(os.listdir(answerFolder))]
featureRefFolder = "../Features/BeatlesCQT3at10FPS"
j = 1
for subFolder in subAnswerFolders:
    validFiles = [subFolder+"/"+file for file in sorted(os.listdir(subFolder)) if file[-3:]=="lab"]
    for i in range(1, len(validFiles)+1):
        numFrames = np.load(featureRefFolder+"/"+str(j)+"features.npy").shape[0]
        gt = getGroundTruthChords(numFrames, i, subFolder, hopSize = 4410)
        np.save("../Features/ChordTargets10FPSBeatles/"+str(j)+"chordTargets.npy", gt)
        np.save("../Features/ChordTargets10FPSBeatles/"+str(j)+"beatTargets.npy", np.zeros(5))
        j+=1