
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

#we don't want to consider the songs that have a weird time signature
weirdTimes = ['006', '034', '037', '038', '043', '050', '071', '072', '076', '077']

#folders where audio and beat annotations are located 
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"

#print the sound file names (test)
print("list: ", os.listdir(audioFolder))

#we want to use the same hop size, but 3 different window sizes
sizes = [1024,2048, 4096]
hopSize = int(44100/100)

#now we loop through each song 
for filename in os.listdir(audioFolder):
    #get record of song's number label
    number = filename[4:7]

    #initialize this to None for now
    finalFeatures = None

    #Skip the readme
    if filename == "README":
        continue
    #ignore weird time signatures
    if number in weirdTimes:
        print(number, "is weird")
        continue

    #load the song using librosa
    x, fs = librosa.load(audioFolder + '/' + filename, sr=None)

    #this doesn't happen (test) but we want everything to have same fs
    if fs != 44100:
        print("egad!!!!")
        print("fs is ", fs)
    
    #For each window size, take the mfcc and first order difference
    for windowSize in sizes:

        #get mfcc for specified hop and window size
        mfccs = librosa.feature.mfcc(x,sr=fs, hop_length = hopSize, n_fft = windowSize)

        #delete first row, the internet said this doesn't give much info
        mfccs = mfccs[1:]
        
        #Make the mean for each feature 0 - COMMENTED OUT FOR NOW, unsure it's correct, we can do it later anyway
        #mfccs = sklearn.preprocessing.scale(mfccs,axis=1) #apparently this does it across rows, same as np yay
        # print(mfccs.mean(axis=1))
        # print("mfccs size ", mfccs.shape)
        # print("dims: ", np.shape(mfccs))

        #we're going to want time on the 0 axis for pytorch, so transpose
        mfccs = mfccs.transpose()

        #First order difference:
        #add an extra time line so our first order difference is correct dims
        mfccAugmented = np.insert(mfccs, 0, 0, axis=0)
        #take the difference 
        firstOrderDiff = np.diff(mfccAugmented,axis=0)
        #we only care about positive diffs
        firstOrderDiff[firstOrderDiff<0] = 0

        #Now stick everything together, our features so far with our new mfccs and firstOrderDiff
        if finalFeatures is None:
            finalFeatures = np.concatenate((mfccs,firstOrderDiff),axis=1)
        else:
            nextFeatures = np.concatenate((mfccs,firstOrderDiff),axis=1)
            finalFeatures = np.concatenate((finalFeatures, nextFeatures),axis=1)

    print("dim of finalFeatures: ", finalFeatures.shape)

    #Write features for this song to file
    np.save("newmfccs/RM-P"+str(number),finalFeatures)
        


