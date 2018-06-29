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

np.set_printoptions(threshold=np.nan)
print("imported")
mfccFolder = "newmfccs"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
modelFile = "modelDictBCE.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#first off, need to get beat activation function and answers 
trainingMFCCs = dataFunctions.getDataAndGroundTruth(mfccFolder,answerFolder)
print("got the data")
#Now, let's generate the model
model = dataFunctions.LSTMBeat(trainingMFCCs[0][0].shape[1], 25, 2).to(DEVICE)
model.load_state_dict(torch.load(modelFile, map_location='cpu'))
print("model is ready")
#Get the beat activation functions
for i in range(len(trainingMFCCs)):
    answerFound = model(trainingMFCCs[i][0]).detach()
    baf = F.sigmoid(answerFound).numpy()
    print("baf shape is ", baf.shape)
    #Step 1: Autocorrelation
    frameShifts = np.arange(27,151) #max tempo to min tempo 
    autoM = np.zeros((len(frameShifts),len(baf)))
    for j in range(len(frameShifts)):
        autoM[j,:] = np.concatenate((np.zeros(frameShifts[j]), baf[:len(baf)-frameShifts[j]])) 
    #print("autoM is ", autoM)
    print("shape of autoM is ", autoM.shape)
    autoCorrelation = np.dot(autoM, baf)
    #print("autoCorrelation is ", autoCorrelation)
    hamm = np.hamming(15) / np.sum(np.hamming(15))
    print("shape of autoCorrelation ", autoCorrelation.shape)
    autoCorrelation = np.convolve(autoCorrelation, hamm, mode="same")
    print("autoCorrelation after hamm is ", autoCorrelation)
    #Find best tempo, for now just do one tempo for the whole song, will change later
    Tstar = frameShifts[np.argmax(autoCorrelation)]
    print("Tstar is ", Tstar)
    #Find best phase
    maxSum = 0
    pStar = 0
    for p in range(Tstar):
        newSum = np.sum(baf[p::Tstar])
        if newSum > maxSum:
            maxSum = newSum
            pStar = p
    print("pStar is ", pStar)
    d = int(0.1*Tstar)
    #centerPoints = bafs[i][pStar::Tstar]
    centerIndices = np.arange(len(baf))[pStar::Tstar]
    beatLocations = []
    #so we want to redo this whole process at every new beat I think? Well I mean we can do whatever we want...
    #you know really I think we should be dynammic programming or something let's take a looksie at the other papers
    #it's annoying cuz it's like this is what we did in class... but worse because my nn sucks

    #Ok 2016 uses a Dynammic Bayesian Network that sounds fun
    for j in centerIndices:
        if baf[max(j-d,0)] <= baf[j] and baf[j] >= baf[min(j+d,len(baf))]:
            beatLocations.append(j)
    plt.figure()
    plt.plot(beatLocations, np.ones(len(beatLocations)), 'ro')
    #plt.plot(np.arange(trainingMFCCs[i][1].numpy().shape[0]),trainingMFCCs[i][1].numpy())
    
    plt.show()
    
