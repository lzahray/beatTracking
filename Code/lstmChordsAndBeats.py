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

# parser = argparse.ArgumentParser()
# parser.add_argument("cross_validation_set")
# args = parser.parse_args()
# k = args.cross_validation_set
versionNumber = "beatsChords1"
fs = 44100
hopSize = int(44100/100)

#CUDA!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", DEVICE)

#PREP TRAINING DATA WITH GROUND TRUTH
featureFolder = "../Features/beatsChordsFeaturesTargets/"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
    
listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]
listOfSongsInOrder = listOfSongsInOrder[:10]
print("length of songsInOrder ", len(listOfSongsInOrder))

#LOSS FUNCTIONS
loss_function_beat = nn.CrossEntropyLoss()
loss_function_chord = nn.CrossEntropyLoss()
if torch.cuda.is_available:
    loss_function_beat.cuda()
    loss_function_chord.cuda()

numFeatures = np.load("../Features/moreFeatures/RM-P001.npy").shape[1]


#GET FEATURES
#featuresAndGT = getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = True)
featuresAndGT = []
for song in listOfSongsInOrder:
    print(song)
    featuresAndGT.append((torch.from_numpy(np.load(featureFolder+str(song)+"features.npy")).float(), torch.from_numpy(np.load(featureFolder+str(song)+"beatTargets.npy")).long(), torch.from_numpy(np.load(featureFolder+str(song)+"chordTargets.npy")).long()))
print("gots the features yo")

#FOR CROSS-FOLD VALIDATION ORGANIZATION
allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1

#TRAINING SECTION (for each cross-validation partition k)
for k in range(5):
    print("starting at k= ", k)

    #INSTANTIATE MODELS
    model1 = LSTMAny(numFeatures, 80, 50, 2).to(DEVICE)
    model1.apply(init_weight)
    modelBeat = LSTMAny(50, 25, 3, 2).to(DEVICE)
    modelBeat.apply(init_weight)
    modelChord = LSTMAny(50, 25, 25, 2).to(DEVICE) #for now let's just include no chord? ehhhhh
    modelChord.apply(init_weight)

    #OPTIMIZER
    optimizer = optim.Adam(list(model1.parameters())+list(modelBeat.parameters())+list(modelChord.parameters()), lr=0.001)

    #ORGANIZATION THINGS 
    bestLoss = np.inf
    stopCount = 0
    lossesTraining = []
    lossesTest = []
    lossesBeatTraining = []
    lossesBeatTest = []
    lossesChordTraining = []
    lossesChordTest = []
    beatsTraining = []
    beatsTest = []
    #we will train 5 total networks with different sets of test 
    indicesTest = allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))]
    indicesTraining = np.array([thing for thing in allIndices if thing not in indicesTest])
    fileForDict = "../"+versionNumber+"/k"+str(k)
    print("Training indices are ", indicesTraining)
    #print("Training songs are ", listOfSongsInOrder[indicesTraining])
    print("Test indices are ", indicesTest)
    #print("Test songs are ", listOfSongsInOrder[indicesTest])
    print("num training: ", len(indicesTraining))
    print("num testing: ", len(indicesTest))

    #START AN EPOCH
    for epoch in range(200):
        start = time.time()

        #EVALUATE how our loss is doing so far 
        with torch.no_grad():
            model1.eval()
            modelBeat.eval()
            modelChord.eval()


            #LOSS FOR TRAINING
            print("on TRAINING songs: ")
            choice = np.random.choice(indicesTraining, 2, replace=False)
            losses = np.zeros(len(choice))
            chordLosses = np.zeros(len(choice))
            beatLosses = np.zeros(len(choice))
            for j in range(len(choice)):
                #for now save time with break
                song = listOfSongsInOrder[choice[j]]
                model1.hidden = model1.init_hidden()
                modelBeat.hidden = modelBeat.init_hidden()
                modelChord.hidden = modelChord.init_hidden()
                timeBeforeLoad = time.time()
                features = featuresAndGT[choice[j]][0].to(DEVICE)
                targetsBeat = featuresAndGT[choice[j]][1].to(DEVICE)
                targetsChord = featuresAndGT[choice[j]][2].to(DEVICE)
                timeAfterLoad = time.time()

                intermediate = model1(features)
                print("intermediate shape is ", intermediate.shape)
                beatTag = modelBeat(intermediate)
                chordTag = modelChord(intermediate)
                #tag_scores = F.softmax(tag_scores, 1)
                beatLoss = loss_function_beat(beatTag, targetsBeat).item()
                chordLoss = loss_function_chord(chordTag, targetsChord).item()
                losses[j] = ((beatLoss + chordLoss)/2.0)
                chordLosses[j] = chordLoss
                beatLosses[j] = beatLoss
                #print("final loss ", loss_function(tag_scores, targets).item())
                #tag_scores = F.softmax(tag_scores, 1)
                #print("max prob of beat ", tag_scores.max().item())
                if j%5==0:
                    print("finished ", j)
            lossesTraining.append(losses.mean())
            lossesChordTraining.append(chordLosses.mean())
            lossesBeatTraining.append(beatLosses.mean())
            print("lossesTraining: ", lossesTraining[-1])
            print("lossesChordTraining: ", lossesChordTraining[-1])
            print("lossesBeatTraining: ", lossesBeatTraining[-1])


            timeTraining = time.time()
            print("TRAINING EVAL TIME ", timeTraining-start)


            #LOSS FOR TEST
            print("On TEST songs: ")
            losses = np.zeros(len(indicesTest))
            chordLosses = np.zeros(len(indicesTest))
            beatLosses = np.zeros(len(indicesTest))
            for j in range(len(indicesTest)):
                #for now save time for break
                song = listOfSongsInOrder[indicesTest[j]]
                model1.hidden = model1.init_hidden()
                modelBeat.hidden = modelBeat.init_hidden()
                modelChord.hidden = modelChord.init_hidden()
                features = featuresAndGT[indicesTest[j]][0].to(DEVICE)
                targetsBeat = featuresAndGT[indicesTest[j]][1].to(DEVICE)
                targetsChord = featuresAndGT[indicesTest[j]][2].to(DEVICE)

                intermediate = model1(features)
                
                beatTag = modelBeat(intermediate)
                chordTag = modelChord(intermediate)
                beatLoss = loss_function_beat(beatTag, targetsBeat).item()
                chordLoss = loss_function_chord(chordTag, targetsChord).item()
                losses[j] = ((beatLoss + chordLoss)/2.0)
                chordLosses[j] = chordLoss
                beatLosses[j] = beatLoss
                if j%5==0:
                    print("finished ", j)
            newAvgLoss = losses.mean()
            lossesTest.append(newAvgLoss)
            lossesChordTest.append(chordLosses.mean())
            lossesBeatTest.append(beatLosses.mean())
            print("lossesTest: ", lossesTest[-1])
            print("lossesChordTest: ", lossesChordTest[-1])
            print("lossesBeatTest: ", lossesBeatTest[-1])

            #SEE IF WE IMPROVED OUR LOSS
            if newAvgLoss < bestLoss:
                stopCount = 0
                # torch.save(model1.state_dict(),fileForDict+'model1.pth')
                # torch.save(modelBeat.state_dict(),fileForDict+'modelBeat.pth')
                # torch.save(modelChord.state_dict(),fileForDict+'modelChord.pth')
                bestLoss = newAvgLoss
            else:
                print("didn't make progress this time")
                stopCount += 1
            averageLoss = newAvgLoss
        
        #SET INTO TRAIN MODE
        model1.train()
        modelBeat.train()
        modelChord.train()
        timeTesting = time.time()
        print("TESTING EVAL TIME ", timeTesting-timeTraining)


        #TRAIN THE MODEL
        i = 0
        for j in range(len(indicesTraining)):
            song = listOfSongsInOrder[indicesTraining[j]]
            features = featuresAndGT[indicesTraining[j]][0].to(DEVICE)
            targetsBeat = featuresAndGT[indicesTraining[j]][1].to(DEVICE)
            targetsChord = featuresAndGT[indicesTraining[j]][2].to(DEVICE)

            if i%5 == 0:
                print(i)
            model1.zero_grad()
            model1.hidden = model1.init_hidden()
            modelBeat.zero_grad()
            modelBeat.hidden = modelBeat.init_hidden()
            modelChord.zero_grad()
            modelChord.hidden = modelChord.init_hidden()
            
            intermediate = model1(features)
            beatTag = modelBeat(intermediate)
            chordTag = modelChord(intermediate)
            loss_beat = loss_function_beat(beatTag, targetsBeat)
            loss_chord = loss_function_chord(chordTag, targetsChord)
            loss = 0.5 * loss_beat + 0.5 * loss_chord
            # loss = (loss_function_beat(beatTag, targetsBeat) + loss_function_chord(chordTag, targetsChord))/2.0
            loss.backward()
            optimizer.step()
            i+=1
        
        print("done with epoch ", epoch)
        #SAVE MODEL PARAMETERS after every epoch
        # np.save("../"+versionNumber+"/lossesTrainingK"+str(k),lossesTraining)
        # np.save("../"+versionNumber+"/lossesTestK"+str(k),lossesTest)
        # np.save("../"+versionNumber+"/lossesBeatTrainingK"+str(k),lossesBeatTraining)
        # np.save("../"+versionNumber+"/lossesBeatTestK"+str(k),lossesBeatTest)
        # np.save("../"+versionNumber+"/lossesChordTrainingK"+str(k),lossesChordTraining)
        # np.save("../"+versionNumber+"/lossesChordTestK"+str(k),lossesChordTest)
        end = time.time()
        print("ACTUAL TRAINING TIME ", end-timeTesting)
        print("TIME THIS EPOCH TOOK: ", end-start)
        
        
        #DECIDE IF WE SHOULD STOP
        if stopCount >= 20:
            print("QUIT, did not make progress")
            break

