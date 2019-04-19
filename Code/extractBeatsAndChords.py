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
import madmom
import argparse
import ast
from functionsForHMM import *
includeBeatles = False
#This will be universal as well. Refactoring was maybe not the best use of your time girl.... BUT IT WILL BE BEAUTIFUL #
#So maybe you're only doing chords, or only beats, or both. It doesn't matter its allll good
possibleModes = ["justBeat", "justChord", "simpleJoint", "complexJoint"]
parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("hyper_parameters")
parser.add_argument("feature_folder")
parser.add_argument("to_save_folder")
parser.add_argument("model_folder")
parser.add_argument("chord_ground_truth")
parser.add_argument("beat_ground_truth")


args = parser.parse_args()
mode = args.mode
hyper_parameters = ast.literal_eval(args.hyper_parameters)
if type(hyper_parameters) == tuple: 
    hyper_parameters = hyper_parameters[0]
featureFolder = args.feature_folder
toSaveFolder = args.to_save_folder
modelFolder = args.model_folder
chord_ground_truth = bool(int(args.chord_ground_truth))
beat_ground_truth = bool(int(args.beat_ground_truth))
print("chord ground truth ", chord_ground_truth)
print("beat ground truth ", beat_ground_truth)

assert(not os.path.exists(toSaveFolder)) #we're gonna try to ensure no overwriting files
os.makedirs(toSaveFolder)
text_file = open(toSaveFolder+"/README.txt", "w")
text_file.write("mode "+mode+"\nhyper parameters " +str(hyper_parameters) + "\nfeature_folder "+featureFolder+"\nchord_ground_truth "+str(chord_ground_truth)+"\nbeat_ground_truth "+str(beat_ground_truth))
text_file.close()

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]

listOfSongsInOrder = np.arange(1,101)
listOfSongsInOrder = [i for i in listOfSongsInOrder if i not in weirdTimes]
beatlesSongsInOrder = [i for i in range(1,181)]
print("names are ", listOfSongsInOrder)

featureTemp, dontNeed1, dontNeed2 = createFeaturesAndTargets(featureFolder, 1, chord_ground_truth, beat_ground_truth,"../Features/mel128ChromaCQTWithTargets")
numFeatures = featureTemp.shape[1]
print("num features ", numFeatures)
allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1
allBeatlesIndices = np.arange(0,len(beatlesSongsInOrder))
testIndicesAll = [allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))] for k in range(5)]
beatlesBoundaryPoint = int(0.2*len(beatlesSongsInOrder))+1
testIndicesAllBeatles = [allBeatlesIndices[k*beatlesBoundaryPoint:min((k+1)*beatlesBoundaryPoint, len(allBeatlesIndices))] for k in range(5)]
#targetFolder = "../Features/mel128ChromaCQTWithTargets" #
targetFolder = "../Features/ChordTargets10FPS"
targetFolder = featureFolder
targetFolderBeatles = "../Features/ChordTargets10FPSBeatles"
featureFolderBeatles = "../Features/BeatlesCQT3at10FPS"
percentages = []
fmeasures = []
songsForFigure = [2, 5, 8, 10, 11, 13, 21, 27, 32, 48, 52, 55, 56, 58, 59, 60]
for i in range(3):
    print("mode is ", mode)
    modelFile = modelFolder + "/k"+str(i)+"model.pth"
    #model1File = modelFolder + "/k"+str(i)+"model1.pth"
    #modelBeatFile = modelFolder + "/k"+str(i)+"modelBeat.pth"
    #modelChordFile = modelFolder + "/k"+str(i)+"modelChord.pth"
    with torch.no_grad():
        
        model = createModel(mode, numFeatures, hyper_parameters)
        #model.modelLeft.load_state_dict(torch.load(modelBeatFile))
        #model.model1.load_state_dict(torch.load(model1File))
        #model.modelRight.load_state_dict(torch.load(modelChordFile))
        if DEVICE == 'cpu':
            model.load_state_dict(torch.load(modelFile, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(modelFile))
        print("model is ready")
    #lisa is adding this for now to test on all 100 songs
    testIndices = testIndicesAll[i]
    #if i == 0:
    #    testIndices += weirdTimes
    testIndicesBeatles = testIndicesAllBeatles[i]
    if includeBeatles:
        union = list(testIndices) + list(testIndicesBeatles)
    else:
        if False:
            union = list(testIndices) + list(weirdTimes)
        else:
            union = testIndices
    print("test indices: ", testIndices)
    #for j in range(len(testIndices),len(union)):
    for j in range(len(testIndices)):
        if j >= len(testIndices):
            song = beatlesSongsInOrder[testIndicesBeatles[j-len(testIndices)]]
            #song = union[j]
            print("we shouldnt be here song ", song)
            features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolderBeatles, song, chord_ground_truth, beat_ground_truth, targetFolderBeatles, mode=mode)
        else:
            song = listOfSongsInOrder[testIndices[j]]
            #if song not in songsForFigure:
             #   continue
            print("song ", song)
        features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolder, song, chord_ground_truth, beat_ground_truth, targetFolder, mode=mode)
        print("time of features ", features.shape[0])
        print("time of targets ", targetsChord.shape[0])
        
        #FOR NOW 
        #chordGuess = np.load("../Results/JointSimpleB/Guesses/chordGuess"+str(song)+".npy")
        #diff = chordGuess - targetsChord
        #numCorrect = list(diff).count(0)
        #percentCorrect = numCorrect / float(targetsChord.shape[0])
        #print("num correct = ", numCorrect)
        #print("percentage correct = ",percentCorrect)
        #percentages.append(percentCorrect)
            #np.save(toSaveFolder+"/chordGuess"+str(song), chordGuess)
        #np.save(toSaveFolder+"/Percentages", percentages)
        #continue

        with torch.no_grad():
            tag = model(features)
            if type(tag) == list:
                for indexT in range(len(tag)):
                    tag[indexT] = tag[indexT].detach()
            elif type(tag) == tuple: #for us this is only for crf
                tag = tag[1]
            else:
                tag = tag.detach()
        
        ####
        #Ok now we have to get the baf if it's needed and the chord func if it's needed #
        softmax = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()
        if mode == "justBeat":
            smBeat = softmax(tag).detach().cpu().numpy()
            smBeat = smBeat[:,1:]
            smChord = None
        elif mode == "justChord" or mode == "conv":
            print("lord knows why we're in here, mode is ", mode)
            smBeat = None
            smChord = softmax(tag).detach().cpu().numpy()  #for now trying ###
        if mode == "crf":
            smBeat = None
            smChord = np.array(tag)
        elif mode == "simpleJoint":
            smBeat = softmax(tag[0]).detach().cpu().numpy()
            smBeat = smBeat[:,1:] #I just added this in Lisa 10/21/2018
            smChord = softmax(tag[1]).detach().cpu().numpy()
        elif mode == "complexJoint":
            ##for now because we're tired, hardcode to just always do sigmoid for left and right (beat and downbeat), softmax for chord
            smBeat = sigmoid(tag[0][:,1]).detach().cpu().numpy()
            #print("tag[1] shape ", tag[1].shape)
            smChord = softmax(tag[1]).detach().cpu().numpy()
            smDownbeat = sigmoid(tag[2][:,1]).detach().cpu().numpy()
            smBeat = np.column_stack((smBeat,smDownbeat))
        

        #CHORD
        #FOR NOW, for saving text files
        #saveSoftmaxToFile(song, smChord,"../ChordSoftmaxTextFiles/simpleJoint")
        
        #FOR NOW, for saving framewise softmaxes to numpy for newest figure
        #np.save(toSaveFolder+"/smChord"+str(song)+".npy", smChord)
        #np.save(toSaveFolder+"/smBeat"+str(song)+".npy", smBeat)
        print("finished text of ", song)
        #continue

        if smChord is not None:
            if mode == "crf":
                chordGuess = smChord
                print("shape of chordGuess ", chordGuess.shape)
            else:
                chordGuess = np.argmax(smChord, axis=1)
            
            #FORNOW using HMM
            #chordGuess = getGuessFromFile("../ChordSoftmaxTextFiles/simpleJointResults/simpleJoint"+str(song)+".txt")
            diff = chordGuess - targetsChord
            numCorrect = list(diff).count(0)
            percentCorrect = numCorrect / float(targetsChord.shape[0])
            print("num correct = ", numCorrect)
            print("percentage correct = ",percentCorrect)
            percentages.append(percentCorrect)
            np.save(toSaveFolder+"/chordGuess"+str(song), chordGuess)
            #np.save(toSaveFolder+"/Percentages", percentages)

        #BEATS
        #continue
        if smBeat is not None:
            proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor([3,4], fps=100)
            baf = smBeat
            #print("baf is ", baf.shape)
            #print("small section is  ", baf[100:120,:])
            #print("ran madmom")
            #beatInfo = proc(baf)
            beatInfo = proc.process(baf)
            #np.save(toSaveFolder+"/beatGuess"+str(song),beatInfo)
            #EVERYTHING is for now because we're saving particular stuff
            #continue
            #print("beats ", beatInfo[:6,:])
            #print("other beats ", otherBeatInfo[:6,:])
            #prep targets (get times in seconds of first just beats, then downbeats)
            annotationsBeats = np.where(targetsBeat != 0)[0] / 100.0
            #print("type of annotations is ", type(annotationsBeats))
            annotationsDownbeats = np.where(targetsBeat == 2)[0] /100.0


            fmeasureBeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsBeats)
            fmeasureDownbeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsDownbeats, downbeats=True)
            
            #save in the format [[beatF,downbeatF]]
            # oldFile = list(np.load(fileToSaveFMeasures+".npy"))
            # print("length of old file is ", len(oldFile))
            # assert(len(oldFile) >= i)
            # if len(oldFile) > i:
            #     oldFile[i] = [fmeasureBeats,fmeasureDownbeats]
            # elif len(oldFile) == i:
            #     oldFile.append([fmeasureBeats,fmeasureDownbeats])
            fmeasures.append([float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
            print("just appended ",[float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
            np.save(toSaveFolder+"/fmeasures", fmeasures)

            text_file = open(toSaveFolder + "/beatInfo"+str(song) + ".txt", "w")
            for line in range(beatInfo.shape[0]):
                text_file.write(str(beatInfo[line,0]) + "\t" + str(int(beatInfo[line,1])) + "\n")
            text_file.close()
#fme = np.load(toSaveFolder+"/fmeasures.npy")
#for f in fme[:,0]:
#    print(f)
#for f in fme[:,1]:
#    print(f)
#for c in np.load(toSaveFolder+"/Percentages.npy"):
#    print(c)
# plt.plot(np.arange(chordGuess.shape[0]), chordGuess)
# plt.plot(np.arange(chordGuess.shape[0]), chordTarget)
#plt.plot(np.arange(chordGuess.shape[0]), chordGuess-chordTarget)
#plt.show()

