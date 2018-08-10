import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing
import dataFunctions
import madmom

beatGT = True
numLayers = 3
numHidden = 50
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]

listOfSongsInOrder = np.arange(1,101)
listOfSongsInOrder = [i for i in listOfSongsInOrder if i not in weirdTimes]
print("names are ", listOfSongsInOrder)

featureFolder = "../Features/mel128ChromaCQTWithTargets/"
answerFolder = "../../CHORD"
modelFolder = "../justChordsBeatGTLayers"+str(numLayers)+"Hidden"+str(numHidden)
featureTemp = torch.from_numpy(np.load(featureFolder+"1features.npy"))
numFeatures = featureTemp.shape[1] if not beatGT else featureTemp.shape[1]+1
folderToWriteGuesses = "../ChordGuesses/JustChordBeatGT"+str(numLayers)+"Layers"+str(numHidden)+"Hidden/"


allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1
boundaryPoint = int(0.2*85)+1
testIndicesAll = [allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))] for k in range(5)]


percentages = []

for i in range(5):
    modelFile = modelFolder + "/k"+str(i)+"model.pth"
    with torch.no_grad():
        model = dataFunctions.LSTMAny(numFeatures, numHidden, 25, numLayers).to(DEVICE).eval()
        model.load_state_dict(torch.load(modelFile, map_location="cpu"))
        print("model is ready")
    testIndices = testIndicesAll[i]
    print("test indices: ", testIndices)
    for j in testIndices:
        song = listOfSongsInOrder[j]
        print("song ", song)
        model.hidden = model.init_hidden()
        if beatGT:
            feat = np.load(featureFolder+str(song)+"features.npy")
            #print("feat shape ", feat.shape)
            
            bgt = np.load(featureFolder+str(song)+"beatTargets.npy")
            bgt = np.reshape(bgt, (bgt.shape[0],1))
            #print("beat shape ", bgt.shape)
            features = torch.from_numpy(    np.concatenate((feat, bgt), axis=1 )   ).float().to(DEVICE)
        else:
            features = torch.from_numpy(np.load(featureFolder+str(song)+"features.npy")).to(DEVICE)
        print("time of features ", features.shape[0])
        #targetsBeat = torch.from_numpy(np.load(featureFolder+str(song)+"beatTargets.npy")).to(DEVICE)
        targetsChord = torch.from_numpy(np.load(featureFolder+str(song)+"chordTargets.npy")).to(DEVICE)
        print("time of targets ", targetsChord.shape[0])
        with torch.no_grad():
            output = model(features).detach()
        print("ran the model")
        m = nn.Softmax(dim=1)
        print("created softmax")
        sm = m(output)
        print("softmaxed")
        chordGuess = np.argmax(sm.detach().numpy(), axis=1)
        print("argmaxed")
        diff = chordGuess - targetsChord
        print("subtracted")
        numCorrect = list(diff).count(0)
        print("counted")
        percentCorrect = numCorrect / float(targetsChord.shape[0])
        print("num correct = ", numCorrect)
        print("percentage correct = ",percentCorrect)
        percentages.append(percentCorrect)
        np.save(folderToWriteGuesses+str(song), chordGuess)
        np.save(folderToWriteGuesses+"Percentages", percentages)
# plt.plot(np.arange(chordGuess.shape[0]), chordGuess)
# plt.plot(np.arange(chordGuess.shape[0]), chordTarget)
#plt.plot(np.arange(chordGuess.shape[0]), chordGuess-chordTarget)
#plt.show()

