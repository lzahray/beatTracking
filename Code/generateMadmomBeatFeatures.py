import numpy as np 
from dataFunctions import getGroundTruthChords, getMoreFeaturesAndGroundTruthDownbeats, getGroundTruthDownbeats
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

#TODO: Get ground truth beats and fix file names 
#Then we need to actually run stuff on the beatles songs and pray that it does well
#If it does we're happy people! If it doesn't we're sad people
#TIME BUDGET: 2 hours. 1 hour for getting ground truth, 1 hour for running it. 
#While running it we're gonna edit the text of the paper 
#MAKE YOSHII PROUD LETS GO
answerFolder = "../../Downloads/The Beatles Annotations/chordlab/The Beatles"
answerFolderBeats = "../../Downloads/The Beatles Annotations/beat/The Beatles"
subAnswerFolders = [answerFolder+"/"+f for f in sorted(os.listdir(answerFolder))][1:]
subAnswerFolderBeats = [answerFolderBeats+"/"+f for f in sorted(os.listdir(answerFolderBeats))][1:]
print("sub ",subAnswerFolderBeats)
albumNumbers = ["01","02","03","04","05","06","07","08","09","10-1","10-2","11","12"]
featureRefFolder = "../Features/Beatles"
for s in range(len(subAnswerFolderBeats)):
	subFolder = subAnswerFolderBeats[s]
	print("\nSUBFOLDER ", subFolder)
	validFiles = [subFolder+"/"+file for file in sorted(os.listdir(subFolder)) if file[-3:]=="txt"]
	print("\nvalid Files ", validFiles)
	songNumbers = [file[0:2] for file in sorted(os.listdir(subFolder)) if file[-3:]=="txt"]
	for i in range(len(validFiles)):
		fileID = albumNumbers[s]+"-"+songNumbers[i]
		print("FileID:", fileID)
		numFrames = np.load(featureRefFolder+"/"+fileID+".npy").shape[0]
		#gt = getGroundTruthChords(numFrames, i, subFolder)
		gtBeats = getGroundTruthDownbeats(numFrames, i, subFolder)
		print("saving to file ", "../Features/BeatlesTargets/"+fileID+"beatTargets.npy")
		#np.save("../Features/BeatlesTargets/"+fileID+"chordTargets.npy", gt)
		np.save("../Features/BeatlesTargets/"+fileID+"beatTargets.npy", gtBeats)
