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


#THE MEMORY SITUATION IS DIRE
#really don't think we need the tempogram

parser = argparse.ArgumentParser()
parser.add_argument("num_layers")
parser.add_argument("hidden_dim")
parser.add_argument("includeBeatGT")
args = parser.parse_args()
num_layers = int(args.num_layers)
hidden_dim = int(args.hidden_dim)
includeBeatGT = bool(int(args.includeBeatGT))
print("num layers ", num_layers)
print("hidden_dim ", hidden_dim)


#goal for next 20 minutes: split the songs into different 

versionNumber = "justChordsBeatGTLayers"+str(num_layers)+"Hidden"+str(hidden_dim)
if not os.path.exists("../"+versionNumber):
    os.makedirs("../"+versionNumber)
fs = 44100
hopSize = int(44100/100)

#CUDA!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", DEVICE)

#PREP TRAINING DATA WITH GROUND TRUTH
featureFolder = "../Features/mel128ChromaCQTWithTargets/"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
    
listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]
#listOfSongsInOrder = listOfSongsInOrder[:10]
print("length of songsInOrder ", len(listOfSongsInOrder))

#LOSS FUNCTIONS
loss_function = nn.CrossEntropyLoss()
if torch.cuda.is_available:
    loss_function.cuda()



#GET FEATURES
#NO LONGER DOING THIS HERE BECAUSE (cry) NOT ENOUGH MEMORY
#featuresAndGT = getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = True)
# featuresAndGT = []
# for song in listOfSongsInOrder:
#     print(song)
#     feat = np.load(featureFolder+str(song)+"features.npy")
#     beatTargets = np.load(featureFolder+str(song)+"beatTargets.npy")
#     chordTargets = np.load(featureFolder+str(song)+"chordTargets.npy")
#     if includeBeatGT:
#         featuresAndGT.append((torch.from_numpy(np.concatenate((feat, beatTargets),axis=1)).float(), torch.from_numpy(beatTargets).long(), torch.from_numpy(chordTargets).long()))
#     else:
#         featuresAndGT.append((torch.from_numpy(feat).float(), torch.from_numpy(beatTargets).long(), torch.from_numpy(chordTargets).long()))
# numFeatures = featuresAndGT[0][0].shape[1]
# print("number of features is ", numFeatures)
numFeatures = np.load(featureFolder+"1features.npy").shape[1]

if includeBeatGT:
    numFeatures += 1
print("numFeatures")
#FOR CROSS-FOLD VALIDATION ORGANIZATION
allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1

#TRAINING SECTION (for each cross-validation partition k)
for k in range(5):
    print("starting at k= ", k)

    #INSTANTIATE MODELS
    model = LSTMAny(numFeatures, hidden_dim, 25, num_layers).to(DEVICE)
    model.apply(init_weight)

    #OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #ORGANIZATION THINGS 
    bestLoss = np.inf
    stopCount = 0
    lossesTraining = []
    lossesTest = [] 
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
            model.eval()


            #LOSS FOR TRAINING
            print("on TRAINING songs: ")
            choice = np.random.choice(indicesTraining, 20, replace=False)
            losses = np.zeros(len(choice))
            for j in range(len(choice)):
                #for now save time with break
                song = listOfSongsInOrder[choice[j]]
                model.hidden = model.init_hidden()
                timeBeforeLoad = time.time()

                #     feat = np.load(featureFolder+str(song)+"features.npy")
                #     beatTargets = np.load(featureFolder+str(song)+"beatTargets.npy")
                #     chordTargets = np.load(featureFolder+str(song)+"chordTargets.npy")
                #     if includeBeatGT:
                #         featuresAndGT.append((torch.from_numpy(np.concatenate((feat, beatTargets),axis=1)).float(), torch.from_numpy(beatTargets).long(), torch.from_numpy(chordTargets).long()))
                #     else:
                #         featuresAndGT.append((torch.from_numpy(feat).float(), torch.from_numpy(beatTargets).long(), torch.from_numpy(chordTargets).long()))
                targetsChord = torch.from_numpy(np.load(featureFolder+str(song)+"chordTargets.npy")).to(DEVICE)
                if includeBeatGT:
                        feat = np.load(featureFolder+str(song)+"features.npy")
                        #print("feat shape ", feat.shape)
                        
                        bgt = np.load(featureFolder+str(song)+"beatTargets.npy")
                        bgt = np.reshape(bgt, (bgt.shape[0],1))
                        #print("beat shape ", bgt.shape)
                        features = torch.from_numpy(    np.concatenate((feat, bgt), axis=1 )   ).float().to(DEVICE)
                else:
                    features = torch.from_numpy(np.load(featureFolder+str(song)+"features.npy")).float().to(DEVICE)
                timeAfterLoad = time.time()

                chordTag = model(features)
                #tag_scores = F.softmax(tag_scores, 1)
                chordLoss = loss_function(chordTag, targetsChord).item()
                losses[j] = chordLoss
                #print("final loss ", loss_function(tag_scores, targets).item())
                #tag_scores = F.softmax(tag_scores, 1)
                #print("max prob of beat ", tag_scores.max().item())
                if j%5==0:
                    print("finished ", j)
            lossesTraining.append(losses.mean())
            print("lossesTraining: ", lossesTraining[-1])


            timeTraining = time.time()
            print("TRAINING EVAL TIME ", timeTraining-start)


            #LOSS FOR TEST
            print("On TEST songs: ")
            losses = np.zeros(len(indicesTest))
            for j in range(len(indicesTest)):
                #for now save time for break
                song = listOfSongsInOrder[indicesTest[j]]
                model.hidden = model.init_hidden()
                targetsChord = torch.from_numpy(np.load(featureFolder+str(song)+"chordTargets.npy")).to(DEVICE)
                if includeBeatGT:
                        feat = np.load(featureFolder+str(song)+"features.npy")
                        #print("feat shape ", feat.shape)
                        
                        bgt = np.load(featureFolder+str(song)+"beatTargets.npy")
                        bgt = np.reshape(bgt, (bgt.shape[0],1))
                        #print("beat shape ", bgt.shape)
                        features = torch.from_numpy(    np.concatenate((feat, bgt), axis=1 )   ).float().to(DEVICE)
                else:
                    features = torch.from_numpy(np.load(featureFolder+str(song)+"features.npy")).float().to(DEVICE)

                chordTag = model(features)
                chordLoss = loss_function(chordTag, targetsChord).item()
                losses[j] = chordLoss
                if j%5==0:
                    print("finished ", j)
            newAvgLoss = losses.mean()
            lossesTest.append(newAvgLoss)

            print("lossesTest: ", lossesTest[-1])


            #SEE IF WE IMPROVED OUR LOSS
            if newAvgLoss < bestLoss:
                stopCount = 0
                torch.save(model.state_dict(),fileForDict+'model.pth')
                # torch.save(modelBeat.state_dict(),fileForDict+'modelBeat.pth')
                # torch.save(modelChord.state_dict(),fileForDict+'modelChord.pth')
                bestLoss = newAvgLoss
            else:
                print("didn't make progress this time")
                stopCount += 1
            averageLoss = newAvgLoss
        
        #SET INTO TRAIN MODE
        model.train()
        timeTesting = time.time()
        print("TESTING EVAL TIME ", timeTesting-timeTraining)


        #TRAIN THE MODEL
        i = 0
        for j in range(len(indicesTraining)):
            song = listOfSongsInOrder[indicesTraining[j]]
            targetsChord = torch.from_numpy(np.load(featureFolder+str(song)+"chordTargets.npy")).to(DEVICE)
            if includeBeatGT:
                    feat = np.load(featureFolder+str(song)+"features.npy")
                    #print("feat shape ", feat.shape)
                    
                    bgt = np.load(featureFolder+str(song)+"beatTargets.npy")
                    bgt = np.reshape(bgt, (bgt.shape[0],1))
                    #print("beat shape ", bgt.shape)
                    features = torch.from_numpy(    np.concatenate((feat, bgt), axis=1 )   ).float().to(DEVICE)
            else:
                features = torch.from_numpy(np.load(featureFolder+str(song)+"features.npy")).float().to(DEVICE)

            if i%5 == 0:
                print(i)
            model.zero_grad()
            model.hidden = model.init_hidden()

            
            chordTag = model(features)
            loss = loss_function(chordTag, targetsChord)
            # loss = (loss_function_beat(beatTag, targetsBeat) + loss_function_chord(chordTag, targetsChord))/2.0
            loss.backward()
            optimizer.step()
            i+=1
        
        print("done with epoch ", epoch)
        #SAVE MODEL PARAMETERS after every epoch
        np.save("../"+versionNumber+"/lossesTrainingK"+str(k),lossesTraining)
        np.save("../"+versionNumber+"/lossesTestK"+str(k),lossesTest)
        end = time.time()
        print("ACTUAL TRAINING TIME ", end-timeTesting)
        print("TIME THIS EPOCH TOOK: ", end-start)
        
        
        #DECIDE IF WE SHOULD STOP
        if stopCount >= 20:
            print("QUIT, did not make progress")
            break

