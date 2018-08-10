
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


def getFirstOrderDiff(originalFeatureVec):
    originalFeatureVecAug = np.insert(originalFeatureVec, 0, 0, axis=0)
    firstOrderDiff = np.diff(originalFeatureVecAug,axis=0)
    #we only care about positive diffs
    firstOrderDiff[firstOrderDiff<0] = 0
    return firstOrderDiff

def stack3Features(originalFeatureVec):
    leftStack = np.concatenate((np.zeros((1,originalFeatureVec.shape[1])), originalFeatureVec[:-1,:]), axis=0)
    rightStack = np.concatenate((originalFeatureVec[1:,:], np.zeros((1,originalFeatureVec.shape[1]))), axis=0)
    return np.concatenate((np.concatenate((leftStack,originalFeatureVec),axis=1), rightStack),axis=1)

def stackNFeatures(originalFeatureVec, N):
    #N is total and should be odd, we're doing 11
    toReturn = np.zeros((originalFeatureVec.shape[0], originalFeatureVec.shape[1]*N))
    paddedZeros = np.concatenate((np.concatenate((np.zeros((int(N/2), originalFeatureVec.shape[1])), originalFeatureVec), axis=0), np.zeros((int(N/2), originalFeatureVec.shape[1]))), axis=0) 
    for i in range(N):
            placeLeft = originalFeatureVec.shape[1]*i
            placeRight = originalFeatureVec.shape[1]*(1+i)
            toReturn[:,placeLeft:placeRight] = paddedZeros[i:i+originalFeatureVec.shape[0], :] 
    return toReturn



def normalize(originalFeatureVec):
    return (originalFeatureVec-originalFeatureVec.mean())/np.std(originalFeatureVec)

def melFromFreq(freq):
    return 2595.0*np.log10(1.0+freq/700.0)
#we don't want to consider the songs that have a weird time signature
#maybe 72 is weird?
weirdTimes = ['006', '022', '028', '030', '034', '037', '038', '041',  '043', '050', '057', '071', '076', '077', '095']

#folders where audio and beat annotations are located 
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"

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
    finalMels = None
    finalDiffs = None
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
    
    #For each window size, take the mel spectrogram logged and first order difference after the log. Cuz why not. 
    # Also, we're gonna do one tempogram 
    #Also also, we're gonna do some chroma. 
    #And we'll normalize everything separate from each other and before we save it to a file for ease of... stuff 
    for windowSize in sizes:

        #get mfcc for specified hop and window size
        mel = librosa.feature.melspectrogram(x,sr=fs, hop_length = hopSize, n_fft = windowSize, fmax=17000, fmin=30, n_mels = 40)
        #spectro = librosa.core.spectrum._spectrogram(x,sr=fs,n_fft = windowSize, hop_length = hopSize,power=2)
        #print("spectro shape is ", spectro.shape)
        # print("window size ", windowSize)
        # print("mfcc shape is ", mfccs.shape)
        #delete first row, the internet said this doesn't give much info
        #mfccs = mfccs[1:]
        
        #Make the mean for each feature 0 - COMMENTED OUT FOR NOW, unsure it's correct, we can do it later anyway
        #mfccs = sklearn.preprocessing.scale(mfccs,axis=1) #apparently this does it across rows, same as np yay
        # print(mfccs.mean(axis=1))
        # print("mfccs size ", mfccs.shape)
        # print("dims: ", np.shape(mfccs))

        #we're going to want time on the 0 axis for pytorch, so transpose
        mel = mel.transpose()
        melDb = librosa.core.power_to_db(mel) 

        #First order difference:
        #This time we're gonna try using the median difference from the 2011 paper, see if it does better 
        #add an extra time line so our first order difference is correct dims
        melAugmentedDb = np.insert(melDb, 0, 0, axis=0)
        #take the difference 
        firstOrderDiff = np.diff(melAugmentedDb,axis=0)
        #we only care about positive diffs
        firstOrderDiff[firstOrderDiff<0] = 0

        #Now we're gonna normalize each thing individually
        melDb = normalize(melDb)
        firstOrderDiff = normalize(firstOrderDiff)

        #Now stick everything together, our features so far with our new mfccs and firstOrderDiff
        if finalMels is None:
            finalMels = melDb
            finalDiffs = firstOrderDiff
        else:
            finalMels = np.concatenate((finalMels,melDb),axis=1)
            finalDiffs = np.concatenate((finalDiffs, firstOrderDiff),axis=1)

    print("shape of finalDiffs is ", finalDiffs.shape)
    print("shape of final mels before stacking is ", finalMels.shape)
    #We should feature stack
    #let's just do 3 total frames worth of stacking 
    #think this through some more girl
    print("shape of finalMels[:-1,:] is ", finalMels[:-1,:].shape)
    print("zero vector shape is ", np.zeros((1,finalMels.shape[1])).shape)
    melsStacked = stack3Features(finalMels)
    print("final shape of melsStacked is ", melsStacked.shape)
    #We should pick one window for the chroma, honestly shouldn't be too big of a feature... i think... 
    #Oh yeah bud it's only 12 long cool beans
    chroma = librosa.feature.chroma_stft(y=x, sr = fs, n_fft=2048, hop_length=hopSize).transpose()
    chromaDiff = getFirstOrderDiff(chroma)
    chroma = normalize(chroma)
    chromaDiff = normalize(chromaDiff)
    chromaStacked = stack3Features(chroma)
    allChroma = np.concatenate((chromaStacked,chromaDiff),axis=1)
    print("final shape of allChroma is ", allChroma.shape)
    #Now we're gonna do the tempogram, HYPE
    #Tempo probably within [55,215]
    tempo = librosa.feature.tempogram(y=x, sr=fs, hop_length=hopSize, win_length=890).transpose()
    #but now I feel like we have too many features in this here tempogram
    maxSecondsPerBeat = 1/(40/60.0)
    minSecondsPerBeat = 1/(250/60.0) #we'll be generous, why not
    minFrameShift = int(minSecondsPerBeat*100)
    maxFrameShift = int(maxSecondsPerBeat*100)
    tempo = tempo[:,minFrameShift:maxFrameShift]
    tempo = normalize(tempo)
    print("final shape of tempo is ", tempo.shape)
    #We're gonna have a million features.......
    finalFeatures = np.concatenate((np.concatenate((melsStacked,finalDiffs), axis=1), np.concatenate((allChroma,tempo),axis=1)),axis=1)
    print("dim of finalFeatures: ", finalFeatures.shape)

    #Write features for this song to file
    np.save("../moreFeatures/RM-P"+str(number),finalFeatures)
        
