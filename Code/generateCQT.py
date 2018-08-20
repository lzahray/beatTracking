import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.display
import os
import torch
import torch.nn as nn
import torch.nn.functional as feature
import torch.optim as optim
import sklearn.preprocessing

def normalize(originalFeatureVec):
    return (originalFeatureVec-originalFeatureVec.mean())/np.std(originalFeatureVec)


weirdTimes = ['006', '022', '028', '030', '034', '037', '038', '041',  '043', '050', '057', '071', '076', '077', '095']
#toSave = ["../Features/CQT1at10FPSTuned/", "../Features/CQT2at10FPSTuned/", "../Features/CQT3at10FPSTuned/"]
toSave = ["../Features/BeatlesCQTActually3at10FPSTuned"]
for i in range(len(toSave)):
    if not os.path.exists(toSave[i]): #we're gonna try to ensure no overwriting files
        os.makedirs(toSave[i])
#folders where audio and beat annotations are located 
# audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
# answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/Beatles/44100s"
#hopSize = int(44100/100)
hopSize = int(44100/10)
num_octaves = 7
files = [f for f in sorted(os.listdir(audioFolder))]
for number in range(1,len(files)+1):
    #get record of song's number label
    filename = files[number-1]
    #Skip the readme
    if filename == "README":
        continue
    #ignore weird time signatures
    if number in weirdTimes:
        print(number, "is weird")
        continue

    #load the song using librosa
    x, fs = librosa.load(audioFolder + '/' + filename, sr=None)
    #newSr = 384 * fs / hopSize
    newSr = 4352 * fs / hopSize
    xResampled = librosa.core.resample(x, fs, newSr)
    for i in range(len(toSave)):
        num_divisions = 3
        cqt = librosa.core.cqt(xResampled, sr=newSr, hop_length = 4352, n_bins=num_octaves*num_divisions*12, bins_per_octave=num_divisions*12, tuning=None)
        cqt = np.abs(cqt)
        cqt = normalize(cqt)
        cqt = cqt.transpose()
        print("we are expecting time by ", 84*num_divisions)
        print("cqt shape after transpose ", cqt.shape) 
        cqtReshaped = np.reshape(cqt, (cqt.shape[0], num_octaves, num_divisions*12))
        print("\n We are expecting time by octave by pitch so time by ", num_octaves, " by ", num_divisions*12)
        print("cqt reshaped is ", cqtReshaped.shape)
        #cqtFifths will have everything arranged in fifths, and it will wrap things completely (almost twice itself except we don't need to repeat first note at the end)
        cqtFifths = np.empty((cqtReshaped.shape[0], cqtReshaped.shape[1], cqtReshaped.shape[2]+cqtReshaped.shape[2]-num_divisions))
        for j in range(cqtFifths.shape[2]):
            #####LISA ATTENTION NOW IN THE MORNING ha morning is 7pm
            #Think about what you want to do about the number of divisions and how to make that math work. 
            #Idea: ignore it for now, always make it 1, so you can train your features
            #better idea: it's not that hard just do it girl
            #ok i need to sleep night night
            nextThingsIndex = (((7*(j//num_divisions))%12)*num_divisions + (j%num_divisions))
            cqtFifths[:,:,j] = cqtReshaped[:,:,nextThingsIndex]
        print("\n We are expeting the same shape as before except last dimension (pitch) should be double minus num_divions")
        print("cqtFifths.shape ", cqtFifths.shape)
        #that was some extreme wrapping! We probably won't need all that wrapping, but! We are preparing for the worst my friend!!
        np.save(toSave[i]+str(int(number)) + "features",cqtFifths)


#THIS SAVES SOMETHING THAT IS time x octave x pitch. If there are multiple pitch divisions, they are all next to each other so like C0a C0b C0c G0a G0b G0c