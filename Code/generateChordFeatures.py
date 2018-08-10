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
toSave = "../Features/mel128ChromaCQT/"
#folders where audio and beat annotations are located 
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"

#print the sound file names (test)
print("list: ", os.listdir(audioFolder))

#we want to use the same hop size, but 3 different window sizes
sizes = [2048]
hopSize = int(44100/100)

#now we loop through each song 
for filename in os.listdir(audioFolder):
    #get record of song's number label
    number = filename[4:7]
    print("number ", number)

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

        # #Mel spectrogram first
        # mel = librosa.feature.melspectrogram(x,sr=fs, hop_length = hopSize, n_fft = windowSize, fmax=17000, fmin=30, n_mels = 80)

        # #we're going to want time on the 0 axis for pytorch, so transpose
        # mel = mel.transpose()
        # melDb = librosa.core.power_to_db(mel) 
        # melDb = normalize(melDb)

        #Really I think we might just want a constant Q transform and then chroma from that, although that seems repetitive? 
        #ok actually let's do a normal log scaled spectrogram and the chroma? 
        newSr = 384 * fs / hopSize
        xResampled = librosa.core.resample(x, fs, newSr)
        cqt = librosa.core.cqt(xResampled, sr=newSr, hop_length = 384)
        chroma_cqt = np.abs(librosa.feature.chroma_cqt(C=cqt))
        chroma_cqt = chroma_cqt.transpose()
        chroma_cqt = normalize(chroma_cqt)
        # plt.figure()
        # plt.title("CQT Chroma")
        
        # librosa.display.specshow(chroma_cqt, y_axis="chroma")
        # plt.show()
        # print("cqt shape is ", chroma_cqt.shape)
        # plt.figure()

        #compare with normal chromagram
        # plt.title("STFT Chroma")
        # chroma_stft = librosa.feature.chroma_stft(y=x, sr=fs, hop_length=hopSize)
        # librosa.display.specshow(chroma_stft,y_axis="chroma")
        # print("stft shape is ", chroma_stft.shape)
        
        mel = librosa.feature.melspectrogram(x,sr=fs, hop_length = hopSize, n_fft = windowSize, fmax=12000, fmin=30, n_mels = 128)
        mel = mel.transpose()
        mel = normalize(librosa.core.power_to_db(mel) )

        featureMatrix = np.concatenate((chroma_cqt,mel), axis=1)
        featureMatrix = stackNFeatures(featureMatrix,11)
        #np.save(toSave+"RM-P"+str(number),featureMatrix)






        
