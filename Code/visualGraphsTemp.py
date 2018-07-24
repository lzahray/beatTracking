import numpy as np 
import matplotlib.pyplot as plt 
import dataFunctions


featureFolder = "../moreFeatures"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
featuresAndGT = dataFunctions.getMoreFeaturesAndGroundTruthDownbeats(featureFolder,answerFolder)



beatLocations = []
myFile = open("../moreFeaturesBeatFiles/74.txt")
myLines = myFile.readlines()
for line in myLines:
    time = float(line.split("\t")[0])
    #print(time)
    beatLocations.append(int(time*100))
#print("beat locations ", beatLocations)
target = featuresAndGT[0][1]
correct = 0
falsePositives = 0
correctPoints = []
falseNegatives = 0
for frame in range(target.shape[0]):
    #print("looking at frame ", frame)
    # if 6660 < frame < 6680:
    #     print("at the point we were wondering about")
    if target[frame] != 0:
        foundBeat = 0
        # if 6660 < frame < 6680:
        #     print("will look through ", max(0,frame-7), "and ", min(frame+8,target.shape[0]))
        for possibility in range(max(0,frame-7), min(frame+8,target.shape[0])):
            if possibility in beatLocations:
                foundBeat = 1
                correctPoints.append(possibility)
        if foundBeat:
            correct += 1
        else:
            falseNegatives += 1 #false negative - it said there wasn't a beat but there was
            #print("did not find a beat, we looked at ", np.arange(max(0,frame-7), min(frame+8,target.shape[0])))
falsePositives = len(beatLocations)-correct
fmeasure = 2*correct / (2*correct + falsePositives + falseNegatives)
#print("correct: ", correct)
#print("falsePositives: ", falsePositives)
#print("falseNegatives: ", falseNegatives)
print("f measure:  ", fmeasure)
falsePosPoints = [p for p in beatLocations if p not in correctPoints]
#print("length of list is ", len(falsePosPoints), " and should be ", falsePositives)
#print("false pos points ", falsePosPoints)
plt.figure()
#plt.plot(beatLocations, np.ones(len(beatLocations)), 'ro')
plt.plot(correctPoints, np.ones(len(correctPoints)), 'ro')
plt.plot(falsePosPoints, np.ones(len(falsePosPoints)), 'bo')
plt.plot(np.arange(featuresAndGT[0][1].numpy().shape[0]),featuresAndGT[0][1].numpy())
plt.show()
