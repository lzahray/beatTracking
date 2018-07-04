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
featureFolder = "melspectrograms"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
modelFile = "modelDictMel/modelDictMel49.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#first off, need to get beat activation function and answers 
featuresAndGT = dataFunctions.getDataAndGroundTruth(featureFolder,answerFolder)
print("got the data")
testIndices =  [2,  7, 12, 21, 33, 35, 40, 49, 54, 58, 59, 60, 64, 67, 69, 70, 77, 85]
#Now, let's generate the model
# model = dataFunctions.LSTMBeat(featuresAndGT[0][0].shape[1], 25, 2).to(DEVICE)
# model.load_state_dict(torch.load(modelFile, map_location='cpu'))



model = dataFunctions.LSTMBeatMel(featuresAndGT[0][0].shape[1], 96, 48, 24, 2).to(DEVICE).eval()
#Now we load the model file which is the last stage we did that should get low loss
model.load_state_dict(torch.load(modelFile, map_location="cpu"))
print("model is ready")
loss_function = nn.BCEWithLogitsLoss()


for i in range(len(featuresAndGT)):
    #We're just seeing how it does on the TRAINING data
    if i in testIndices: 
        print("next index")  
        targets = featuresAndGT[i][1]
        print("has targets")
        # model.init_hidden()
        # print("inited hidden")
        answerFound = model(featuresAndGT[i][0]).detach()
        print("found the answer")
        loss = loss_function(answerFound, targets).item()
        print("the error cross entropy is ", loss)
        baf = F.sigmoid(answerFound).cpu().numpy()
        #baf = F.sigmoid(answerFound)
        plt.figure()
        plt.plot(np.arange(len(baf)),baf)
        plt.plot(np.arange(featuresAndGT[i][1].numpy().shape[0]),featuresAndGT[i][1].numpy())
        plt.show()
        # plt.show()
        print("baf shape is ", baf.shape)
        #Step 1: Autocorrelation
        frameShifts = np.arange(27,151) #max tempo to min tempo 
        autoM = np.zeros((len(frameShifts),len(baf)))
        for j in range(len(frameShifts)):
            autoM[j,:] = np.concatenate((np.zeros(frameShifts[j]), baf[:len(baf)-frameShifts[j]])) 
        #print("autoM is ", autoM)
        #print("shape of autoM is ", autoM.shape)
        autoCorrelation = np.dot(autoM, baf)
        #print("autoCorrelation is ", autoCorrelation)
        hamm = np.hamming(15) / np.sum(np.hamming(15))
        #print("shape of autoCorrelation ", autoCorrelation.shape)
        autoCorrelation = np.convolve(autoCorrelation, hamm, mode="same")
        #print("autoCorrelation after hamm is ", autoCorrelation)
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
            #for k in range(max(j-d,0), min(j+d,len(baf)-1)):
            if baf[max(j-d,0)] <= baf[j] and baf[j] >= baf[min(j+d,len(baf)-1)]:
                beatLocations.append(j)
            #beatLocations.append(np.argmax(baf[max(j-d,0):min(j+d,len(baf)-1)]) + max(j-d,0))


        #DETECTING F-MEASURE
        #target = featuresAndGT[i][1].numpy() #this should just have 0s and 1s I think
        target = featuresAndGT[i][1]
        correct = 0
        falsePositives = 0
        correctPoints = []
        falseNegatives = 0
        for frame in range(target.shape[0]):
            if target[frame] == 1:
                foundBeat = 0
                for possibility in range(max(0,frame-7), min(frame+8,target.shape[0])):
                    if possibility in beatLocations:
                        foundBeat = 1
                        correctPoints.append(possibility)
                if foundBeat:
                    correct += 1
                else:
                    falseNegatives += 1 #false negative - it said there wasn't a beat but there was
                    #print("false negative because ")
        falsePositives = len(beatLocations)-correct
        fmeasure = 2*correct / (2*correct + falsePositives + falseNegatives)
        print("correct: ", correct)
        print("falsePositives: ", falsePositives)
        print("falseNegatives: ", falseNegatives)
        print("f measure:  ", fmeasure)
        falseNegPoints = [p for p in beatLocations if p not in correctPoints]
        plt.figure()
        #plt.plot(beatLocations, np.ones(len(beatLocations)), 'ro')
        plt.plot(correctPoints, np.ones(len(correctPoints)), 'ro')
        plt.plot(falseNegPoints, np.ones(len(falseNegPoints)), 'bo')
        plt.plot(np.arange(featuresAndGT[i][1].numpy().shape[0]),featuresAndGT[i][1].numpy())
        
        plt.show()