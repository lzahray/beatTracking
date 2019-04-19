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

includeBeatles = False
#This will be universal as well. Refactoring was maybe not the best use of your time girl.... BUT IT WILL BE BEAUTIFUL #
#So maybe you're only doing chords, or only beats, or both. It doesn't matter its allll good
possibleModes = ["justBeat", "justChord", "simpleJoint", "complexJoint"]
# parser = argparse.ArgumentParser()
# parser.add_argument("mode")
# parser.add_argument("hyper_parameters")
# parser.add_argument("feature_folder")
# parser.add_argument("to_save_folder")
# parser.add_argument("model_folder")
# parser.add_argument("chord_ground_truth")
# parser.add_argument("beat_ground_truth")


# args = parser.parse_args()
# mode = args.mode
# hyper_parameters = ast.literal_eval(args.hyper_parameters)
# featureFolder = args.feature_folder
# toSaveFolder = args.to_save_folder
# modelFolder = args.model_folder
# chord_ground_truth = bool(int(args.chord_ground_truth))
# beat_ground_truth = bool(int(args.beat_ground_truth))
# print("chord ground truth ", chord_ground_truth)
# print("beat ground truth ", beat_ground_truth)


mode = "simpleJoint"
hyper_parameters_1 = {"hidden_dim":100,"num_layers":1}
hyper_parameters_beat = {"hidden_dim":40,"num_layers":2}
hyper_parameters_chord = {"hidden_dim":40,"num_layers":2}
featureFolder = "../Features/Beatles"
toSaveFolder = "../ismirTests/bestJointWith2"
targetFolder = "../Features/BeatlesTargets"
modelFolder = "../beatsChordsLayerOne1HiddenOne100OutputOne75LayerOther2HiddenOther40"
chord_ground_truth = False
beat_ground_truth = False
#assert(not os.path.exists(toSaveFolder)) #we're gonna try to ensure no overwriting files
os.makedirs(toSaveFolder)
text_file = open(toSaveFolder+"/README.txt", "w")
text_file.write("mode "+mode+"\nhyper parameters_1 " +str(hyper_parameters_1) + "\nfeature_folder "+featureFolder+"\nchord_ground_truth "+str(chord_ground_truth)+"\nbeat_ground_truth "+str(beat_ground_truth))
text_file.close()

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]

featureTemp, dontNeed1, dontNeed2 = createFeaturesAndTargets(featureFolder, "01-01", chord_ground_truth, beat_ground_truth,"../Features/BeatlesTargets")
numFeatures = featureTemp.shape[1]
print("num features ", numFeatures)
#targetFolder = "../Features/mel128ChromaCQTWithTargets" #
#we just need to get a list of all the possible songIDs now 
albumNumbers = ["01","02","03","04","05","06","07","08","09","10-1","10-2","11","12"]
answerFolder = "../../Downloads/The Beatles Annotations/chordlab/The Beatles"
songIDs = []
subAnswerFolders = [answerFolder+"/"+f for f in sorted(os.listdir(answerFolder))][1:]
for s in range(len(subAnswerFolders)):
    subFolder = subAnswerFolders[s]
    #print("subFolder is ",subFolder)
    validFiles = [subFolder+"/"+file for file in sorted(os.listdir(subFolder)) if file[-3:]=="lab"]
    songNumbers = [file[0:2] for file in sorted(os.listdir(subFolder)) if file[-3:]=="lab"]
    #print("song Numbers: ",songNumbers)
    for n in range(len(songNumbers)):
        songIDs.append(albumNumbers[s]+"-"+songNumbers[n])
print("songIDs: ", songIDs)

percentages = []
fmeasures = []
for i in range(1,2): #it doesn't matter which we use i guess 
    print("mode is ", mode)
    modelFileJoint = modelFolder + "/k"+str(i)+"model1.pth"
    modelFileBeat = modelFolder + "/k"+str(i)+"modelBeat.pth"
    modelFileChord = modelFolder + "/k"+str(i)+"modelChord.pth"
    with torch.no_grad():
        model1 = createModel("weMessedUp", numFeatures, hyper_parameters_1)
        modelBeat = createModel("justBeat", 75, hyper_parameters_beat)
        modelChord = createModel("justChord",75,hyper_parameters_chord)
        #if DEVICE == 'cpu':
        model1.load_state_dict(torch.load(modelFileJoint, map_location="cpu"))
        modelBeat.load_state_dict(torch.load(modelFileBeat, map_location="cpu"))
        modelChord.load_state_dict(torch.load(modelFileChord, map_location="cpu"))
        #else:
            #print("lol how")
            #model1.load_state_dict(torch.load(modelFile))
        print("model is ready")
    for j in range(len(songIDs)):
        song = songIDs[j]
        if song=="10-2-12":
            continue
        print("song ", song)
        features, targetsBeat, targetsChord = createFeaturesAndTargets(featureFolder, song, chord_ground_truth, beat_ground_truth, targetFolder, mode=mode)
        print("time of features ", features.shape[0])
        print("time of targets ", targetsChord.shape[0])
        with torch.no_grad():
            tag1 = model1(features)
            tagBeat = modelBeat(tag1)
            tagChord = modelChord(tag1)
            if type(tag1) == list:
                for indexT in range(len(tag1)):
                    tag1[indexT] = tag1[indexT].detach()
            if type(tagBeat) == list:
                for indexT in range(len(tagBeat)):
                    tagBeat[indexT] = tagBeat[indexT].detach()
            if type(tagChord) == list:
                for indexT in range(len(tagChord)):
                    tagChord[indexT] = tagChord[indexT].detach()
            #elif type(tag) == tuple: #for us this is only for crf
            #    tag = tag[1]
            #else:
            #    tag = tag.detach()
        
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
            print("tagBeat is ", tagBeat.shape)
            smBeat = softmax(tagBeat).detach().cpu().numpy()
            smBeat = smBeat[:,1:]
            print("smBeat is ", smBeat.shape)
            smChord = softmax(tagChord).detach().cpu().numpy()
        elif mode == "complexJoint":
            ##for now because we're tired, hardcode to just always do sigmoid for left and right (beat and downbeat), softmax for chord
            smBeat = sigmoid(tag[0][:,1]).detach().cpu().numpy()
            #print("tag[1] shape ", tag[1].shape)
            smChord = softmax(tag[1]).detach().cpu().numpy()
            smDownbeat = sigmoid(tag[2][:,1]).detach().cpu().numpy()
            smBeat = np.column_stack((smBeat,smDownbeat))
        

        #CHORD
        if smChord is not None:
            if mode == "crf":
                chordGuess = smChord
                print("shape of chordGuess ", chordGuess.shape)
            else:
                chordGuess =np.array(np.argmax(smChord, axis=1))
            diff = chordGuess - np.array(targetsChord)
            numCorrect = list(diff).count(0)
            percentCorrect = numCorrect / float(targetsChord.shape[0])
            print("num correct = ", numCorrect)
            print("percentage correct = ",percentCorrect)
            percentages.append(percentCorrect)
            np.save(toSaveFolder+"/chordGuess"+str(song), chordGuess)
            np.save(toSaveFolder+"/Percentages", percentages)

        #BEATS
        if smBeat is not None:
            proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4], fps=100)
            baf = np.array(smBeat)
            print("baf is ", baf.shape)
            print("row 102 is ", baf[102])
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

