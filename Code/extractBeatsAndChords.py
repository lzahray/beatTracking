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
print("names are ", listOfSongsInOrder)

featureTemp, dontNeed1, dontNeed2 = createFeaturesAndTargets(featureFolder, 1, chord_ground_truth, beat_ground_truth,"../Features/mel128ChromaCQTWithTargets")
numFeatures = featureTemp.shape[1]
print("num features ", numFeatures)
allIndices = np.arange(0, len(listOfSongsInOrder))
boundaryPoint = int(0.2*len(listOfSongsInOrder))+1
testIndicesAll = [allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))] for k in range(5)]


percentages = []
fmeasures = []
for i in range(3):
    print("mode is ", mode)
    modelFile = modelFolder + "/k"+str(i)+"model.pth"
    with torch.no_grad():
        model = createModel(mode, numFeatures, hyper_parameters)
        if DEVICE == 'cpu':
            model.load_state_dict(torch.load(modelFile, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(modelFile))
        print("model is ready")
    testIndices = testIndicesAll[i]
    print("test indices: ", testIndices)
    for j in testIndices:
        song = listOfSongsInOrder[j]
        print("song ", song)
        features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolder, song, chord_ground_truth, beat_ground_truth, "../Features/mel128ChromaCQTWithTargets", mode=mode)
        print("time of features ", features.shape[0])
        print("time of targets ", targetsChord.shape[0])
        with torch.no_grad():
            tag = model(features)
            if type(tag) == list:
                for indexT in range(len(tag)):
                    tag[indexT] = tag[indexT].detach()
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
        elif mode == "simpleJoint":
            smBeat = softmax(tag[0]).detach().cpu().numpy()
            smChord = softmax(tag[1]).detach().cpu().numpy()
        elif mode == "complexJoint":
            ##for now because we're tired, hardcode to just always do sigmoid for left and right (beat and downbeat), softmax for chord
            smBeat = sigmoid(tag[0][:,1]).detach().cpu().numpy()
            #print("tag[1] shape ", tag[1].shape)
            smChord = softmax(tag[1]).detach().cpu().numpy()
            smDownbeat = sigmoid(tag[2][:,1]).detach().cpu().numpy()
            smBeat = np.column_stack((smBeat,smDownbeat))
        

        #CHORD
        if smChord is not None:
            chordGuess = np.argmax(smChord, axis=1)
            diff = chordGuess - targetsChord
            numCorrect = list(diff).count(0)
            percentCorrect = numCorrect / float(targetsChord.shape[0])
            print("num correct = ", numCorrect)
            print("percentage correct = ",percentCorrect)
            percentages.append(percentCorrect)
            np.save(toSaveFolder+"/chordGuess"+str(song), chordGuess)
            np.save(toSaveFolder+"/Percentages", percentages)

        #BEATS
        if smBeat is not None:
            proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(4, fps=100)
            baf = smBeat
            #print("baf is ", baf.shape)
            #print("row 102 is ", baf[102])
            #print("ran madmom")
            #beatInfo = proc(baf)
            beatInfo = proc.process(baf)
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


# plt.plot(np.arange(chordGuess.shape[0]), chordGuess)
# plt.plot(np.arange(chordGuess.shape[0]), chordTarget)
#plt.plot(np.arange(chordGuess.shape[0]), chordGuess-chordTarget)
#plt.show()

