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
import ast



#PARSER 
possibleModes = ["justBeat", "justChord", "simpleJoint", "complexJoint", "conv", "crf"]
parser = argparse.ArgumentParser()
#mode tells us which model type we should be using and which features/targets
parser.add_argument("mode")
#hyper parameters will get passed into the model class so it knows how to instantiate itself
parser.add_argument("hyper_parameters")
#for features to be using, will load from numpy 
parser.add_argument("feature_folder")
parser.add_argument("to_save_folder")
parser.add_argument("chord_ground_truth")
parser.add_argument("beat_ground_truth")

args = parser.parse_args()
print("args are ", args)
mode = args.mode
hyper_parameters = ast.literal_eval(args.hyper_parameters)
print("hyper parameters: ", hyper_parameters)
featureFolder = args.feature_folder
toSaveFolder = args.to_save_folder
print("toSaveFolder ", toSaveFolder)
chord_ground_truth = bool(int(args.chord_ground_truth))
beat_ground_truth = bool(int(args.beat_ground_truth))
print("chord ground truth ", chord_ground_truth)
print("beat ground truth ", beat_ground_truth)

assert(not os.path.exists(toSaveFolder)) #we're gonna try to ensure no overwriting files
os.makedirs(toSaveFolder)
#make a readme
text_file = open(toSaveFolder+"/README.txt", "w")
text_file.write("mode "+mode+"\nhyper parameters " +str(hyper_parameters) + "\nfeature_folder "+featureFolder+"\nchord_ground_truth "+str(chord_ground_truth)+"\nbeat_ground_truth "+str(beat_ground_truth))
text_file.close()


#CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", DEVICE)

listOfSongsInOrder = [i for i in range(1,101) if ('0'*(3-len(str(i))) + str(i)) not in weirdTimes]
beatlesSongsInOrder = [i for i in range(1,181)]
#listOfSongsInOrder = listOfSongsInOrder[:10]
print("length of songsInOrder ", len(listOfSongsInOrder))

#targetFolder = "../Features/mel128ChromaCQTWithTargets"
if mode == "justBeat":
    targetFolder = featureFolder
else:
    targetFolder = "../Features/ChordTargets10FPS"
    targetFolderBeatles = "../Features/BeatlesCQT3at10FPS"

#LOSS FUNCTIONS
loss_functions = []
if mode == "justBeat" or mode=="justChord" or mode == "conv":
    loss_functions = [nn.CrossEntropyLoss()]
elif mode == "simpleJoint":
    loss_functions = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
elif mode == "complexJoint":
    loss_functions = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]  
if torch.cuda.is_available:
    for func in loss_functions:
        func.cuda()

#GET NUMFEATURES
if mode == "conv" or mode == "crf": #i think we just don't care for this but whatever
    featureTemp, dontNeed1, dontNeed2 = createFeaturesAndTargets(featureFolder, 1, chord_ground_truth, beat_ground_truth, targetFolder, mode="conv")
else:
    featureTemp, dontNeed1, dontNeed2 = createFeaturesAndTargets(featureFolder, 1, chord_ground_truth, beat_ground_truth, targetFolder)
    featureTemp = featureTemp.view(featureTemp.shape[0],-1)
numFeatures = featureTemp.shape[1]


#FOR CROSS-FOLD VALIDATION ORGANIZATION
allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1
allBeatlesIndices = np.arange(0,len(beatlesSongsInOrder))
beatlesBoundaryPoint = int(0.2*len(beatlesSongsInOrder))+1



def saveLossesToFile(k, losses, oldLosses, train):
    #it also returns the average main loss which is always the first one in the list
    extra = "train" if train else "test"
    
    lossAvgs = losses.mean(axis=0)
    oldLosses.append(lossAvgs)
    fileName = toSaveFolder + "/loss" + str(k) + extra

    np.save(fileName, oldLosses)
    print("loss avgs are ", lossAvgs)
    return lossAvgs[0]

def saveTorchModel(k, model):
    #we are hoping this works. Oh rip, well we didn't save our data function but.... i don't actually know that we can 
    #rerun the stuff we had......
    fileForDict = toSaveFolder + "/k" + str(k) + "model.pth"
    torch.save(model.state_dict(),fileForDict)



def runModel(evaluate, song, model, optimizer, beatles=False):
    #this if statement only needed because we don't have targets saved in the conv feature folder
    if beatles:
        ff = "../Features/BeatlesCQT3at10FPS"
        tf = "../Features/ChordTargets10FPSBeatles"
    else:
        ff = featureFolder
        tf = targetFolder
    if mode == "conv" or mode == "crf":
        features, targetsBeat, targetsChord = createFeaturesAndTargets(ff, song, chord_ground_truth, beat_ground_truth, tf,mode="conv")
    else:
        features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolder, song, chord_ground_truth, beat_ground_truth, targetFolder)
        features = features.view(features.shape[0],-1)

    if not evaluate:
        optimizer.zero_grad()
    
    #RUN THE MODEL
    if mode == "crf":
        loss = model.neg_log_likelihood(features, targetsChord)
        losses = [loss]
    else:
        tag = model(features)
        
        if mode == "justBeat":
            targets = targetsBeat
        elif mode == "justChord" or mode == "conv":
            targets = targetsChord
        elif mode == "simpleJoint":
            targets = [targetsBeat, targetsChord]
        elif mode == "complexJoint":
            targets = [(targetsBeat==1).long(), targetsChord, (targetsBeat==2).long()]

        losses = model.calculate_loss(loss_functions, tag, targets)
        loss = losses[0]
    #print("ran the model and calculated loss")
    # loss = (loss_function_beat(beatTag, targetsBeat) + loss_function_chord(chordTag, targetsChord))/2.0
    if not evaluate:
        loss.backward()
        optimizer.step()
    return losses 




#TRAINING SECTION (for each cross-validation partition k)
for k in range(5):
    print("starting at k= ", k)

    #INSTANTIATE MODEL
    model = createModel(mode, numFeatures, hyper_parameters)
    model.apply(init_weight)

    #OPTIMIZER
    #print("model parameters are ", list(model.parameters()))
    # if mode == "crf":
    #     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #ORGANIZATION THINGS 
    bestLoss = np.inf
    stopCount = 0
    #we will train 5 total networks with different sets of test 
    indicesTest = allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))]
    indicesTraining = np.array([thing for thing in allIndices if thing not in indicesTest])
    indicesBeatlesTest = allBeatlesIndices[k*beatlesBoundaryPoint:min((k+1)*beatlesBoundaryPoint, len(allBeatlesIndices))]
    indicesBeatlesTraining = np.array([thing for thing in allBeatlesIndices if thing not in indicesBeatlesTest])
    print("Training indices are ", indicesTraining)
    print("Training Beatles indices are ", indicesBeatlesTraining)
    #print("Training songs are ", listOfSongsInOrder[indicesTraining])
    print("Test indices are ", indicesTest)
    print("Test Beatles indices are ", indicesBeatlesTest)
    #print("Test songs are ", listOfSongsInOrder[indicesTest])
    print("num training: ", len(indicesTraining))
    print("num testing: ", len(indicesTest))

    #START AN EPOCH
    trainingLosses = []
    testingLosses = []
    for epoch in range(100):
        start = time.time()

        #EVALUATE how our loss is doing so far 
        with torch.no_grad():
            model.eval()

            #LOSS FOR TRAINING
            print("on TRAINING songs: ")
            choice = np.random.choice(indicesTraining, 15, replace=False)
            #choiceBeatles = np.random.choice(indicesBeatlesTraining, 5, replace=False)
            #losses = np.zeros((len(choice) + len(choiceBeatles), model.num_losses))
            losses = np.zeros((len(choice), model.num_losses))
            for j in range(len(choice)):
                #for now save time with break
                song = listOfSongsInOrder[choice[j]]
                losses[j,:] = runModel(True, song, model, optimizer)
                if j%5==0:
                    print("finished ", j)
            # for j in range(len(choice),len(choice)+len(choiceBeatles)):
            #     #for now save time with break
            #     song = beatlesSongsInOrder[choiceBeatles[j-len(choice)]]
            #     losses[j,:] = runModel(True, song, model, optimizer, beatles=True)
            #     if j%5==0:
            #         print("finished ", j)
            avgLoss = saveLossesToFile(k, losses,trainingLosses, True)
            print("avg training loss: ", avgLoss)
            timeTraining = time.time()
            print("TRAINING EVAL TIME ", timeTraining-start)


            #LOSS FOR TEST 
            #
            print("On TEST songs: ")
            choice = np.random.choice(indicesTest, min(15,len(indicesTest)), replace=False)
            #losses = np.zeros((len(indicesTest)+len(indicesBeatlesTest), model.num_losses))
            losses = np.zeros((len(choice), model.num_losses))
            #for j in range(len(indicesTest)):
            for j in range(len(choice)):
                #for now save time for break
                #song = listOfSongsInOrder[indicesTest[j]]
                song = listOfSongsInOrder[choice[j]]
                losses[j,:] = runModel(True, song, model, optimizer)
                if j%5==0:
                    print("finished ", j)
            # for j in range(len(indicesTest), len(indicesTest)+len(indicesBeatlesTest)):
            #     #for now save time for break
            #     song = beatlesSongsInOrder[indicesBeatlesTest[j-len(indicesTest)]]
            #     losses[j,:] = runModel(True, song, model, optimizer, beatles=True)
            #     if j%5==0:
            #         print("finished ", j)
            newAvgLoss = saveLossesToFile(k, losses, testingLosses, False)
            print("avg test loss: ", newAvgLoss)
            #SEE IF WE IMPROVED OUR LOSS
            if newAvgLoss < bestLoss:
                stopCount = 0
                saveTorchModel(k, model)
                bestLoss = newAvgLoss
            else:
                print("didn't make progress this time, counter at ", stopCount)
                print("best loss is ", bestLoss)
                stopCount += 1
            averageLoss = newAvgLoss
        timeTesting = time.time()
        print("TESTING EVAL TIME ", timeTesting-timeTraining)
        print("Time for the training phase")
        #TRAIN THE MODEL
        model.train()
        i = 0
        for j in range(len(indicesTraining)):
        #for j in range(10):
            song = listOfSongsInOrder[indicesTraining[j]]
            dontNeed = runModel(False, song, model, optimizer) #just don't actually need to save it anywhere
            if i%5 == 0:
                print(i)
            i+=1
        # for j in range(len(indicesTraining),len(indicesTraining)+len(indicesBeatlesTraining)):
        #     song = beatlesSongsInOrder[indicesBeatlesTraining[j-len(indicesTraining)]]
        #     dontNeed = runModel(False, song, model, optimizer,beatles=True) #just don't actually need to save it anywhere
        #     if i%5 == 0:
        #         print(i)
        #     i+=1
        
        print("done with epoch ", epoch)
        end = time.time()
        print("ACTUAL TRAINING TIME ", end-timeTesting)
        print("TIME THIS EPOCH TOOK: ", end-start)
        
        
        #DECIDE IF WE SHOULD STOP
        if stopCount >= 10:
            print("QUIT, did not make progress")
            print("Best loss was ", bestLoss)
            break

