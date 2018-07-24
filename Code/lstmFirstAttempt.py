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

fs = 44100
hopSize = int(44100/100)

#CUDA!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#PREP TRAINING DATA WITH GROUND TRUTH
featuresAndGT = []
featureFolder = "melspectrograms"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"

featuresAndGT, listOfSongsInOrder = getDataAndGroundTruth(featureFolder, answerFolder, True)  
#featuresAndGT is now list of (feature, beatvector)


#INSTANTIATE THE MODEL   
#model = LSTMBeat(featuresAndGT[0][0].shape[1], 25, 2).to(DEVICE)

#Using cross entropy because we're softmaxing 
loss_function = nn.BCEWithLogitsLoss()
#Stochastic Gradient Descent, lr and momentum from paper
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




#Select indices for training data vs. test data 
#let's do 80% train, 20% test
allIndices = np.arange(0, len(featuresAndGT))
# print("length of all indices ", len(allIndices))
# indicesTraining = np.random.choice(allIndices,size=int(0.8*len(featuresAndGT)), replace=False)
# indicesTest = np.array([i for i in allIndices if i not in indicesTraining])


#TRAINING SECTION


boundaryPoint = int(0.2*len(featuresAndGT))+1
for k in range(5):
    if k == 4:
        model = LSTMBeatMel(featuresAndGT[0][0].shape[1], 96, 48, 24, 2).to(DEVICE)
        model.apply(init_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("starting at k= ", k)
        bestLoss = np.inf
        stopCount = 0
        lossesTraining = []
        lossesTest = []
        beatsTraining = []
        beatsTest = []
        #we will train 5 total networks with different sets of test 
        indicesTest = allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))]
        indicesTraining = np.array([thing for thing in allIndices if thing not in indicesTest])
        fileForDict = "modelDictMelCross/k"+str(k)+"epoch"
        print("Training indices are ", indicesTraining)
        print("Training songs are ", listOfSongsInOrder[indicesTraining])
        print("Test indices are ", indicesTest)
        print("Test songs are ", listOfSongsInOrder[indicesTest])
        print("num training: ", len(indicesTraining))
        print("num testing: ", len(indicesTest))
        for epoch in range(200):
            #EVALUATE how our loss is doing so far 
            with torch.no_grad():
                #LOSS FOR TRAINING
                print("on TRAINING songs: ")
                maxBeatProbs = np.zeros(len(indicesTraining))
                losses = np.zeros(len(indicesTraining))
                for j in range(len(indicesTraining)):
                    model.init_hidden()
                    features, targets = featuresAndGT[indicesTraining[j]]
                    tag_scores = model(features)
                    #tag_scores = F.softmax(tag_scores, 1)
                    losses[j] = loss_function(tag_scores, targets).item()
                    #print("final loss ", loss_function(tag_scores, targets).item())
                    maxBeatProbs[j] = tag_scores.max().item()
                    #tag_scores = F.softmax(tag_scores, 1)
                    #print("max prob of beat ", tag_scores.max().item())
                lossesTraining.append(losses.mean())
                beatsTraining.append(maxBeatProbs.mean())
                print("lossesTraining: ", lossesTraining[-1])
                print("beatsTraining: ", beatsTraining[-1])
                
                #LOSS FOR TEST
                print("On TEST songs: ")
                maxBeatProbs = np.zeros(len(indicesTest))
                losses = np.zeros(len(indicesTest))
                for j in range(len(indicesTest)):
                    model.init_hidden()
                    features, targets = featuresAndGT[indicesTest[j]]
                    # print("number of 1s ", targets.sum())
                    # print("percent ones is ", targets.sum().item()/float(len(targets)))
                    tag_scores = model(features)
                    theloss = loss_function(tag_scores, targets).item()
                    losses[j] = theloss
                    maxBeatProbs[j] = tag_scores.max().item()
                    #tag_scores = F.softmax(tag_scores, 1)
                newAvgLoss = losses.mean()
                lossesTest.append(newAvgLoss)
                beatsTest.append(maxBeatProbs.mean())
                print("lossesTest: ", lossesTest[-1])
                print("beatsTest: ", beatsTest[-1])
                if newAvgLoss < bestLoss:
                    stopCount = 0
                    torch.save(model.state_dict(),fileForDict+str(epoch)+'.pth')
                    bestLoss = newAvgLoss
                else:
                    print("didn't make progress this time")
                    stopCount += 1
                averageLoss = newAvgLoss

            #Now actually TRAIN the model
            i = 0
            for j in indicesTraining:

                features, targets = featuresAndGT[j]
                if i%10 == 0:
                    print(i)
                model.zero_grad()
                model.init_hidden()
                scores = model(features)
                loss = loss_function(scores, targets)
                loss.backward()
                optimizer.step()
                i+=1
            
            print("done with epoch ", epoch)
            #SAVE MODEL PARAMETERS after every epoch
            #torch.save(model.state_dict(),'blarghhhh'+str(epoch)+'.pth')
            #SAVE numpy arrays about our average loss each epoch
            np.save("lossesTrainingK"+str(k),lossesTraining)
            np.save("lossesTest"+str(k),lossesTest)
            np.save("beatsTraining"+str(k),beatsTraining)
            np.save("beatsTest"+str(k),beatsTest)

            if stopCount >= 20:
                print("QUIT, did not make progress")
                break

