
import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.display
import os
import torch
import torch.nn as nn
import torch.nn.functional as feature
import torch.optim as optim
import sklearn

weirdTimes = ['006', '034', '037', '038', '043', '050', '071', '072', '076', '077']
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
print("list: ", os.listdir(audioFolder))
for filename in os.listdir(audioFolder):
    if filename == "README":
        continue
    number = filename[4:7]
    print("number is ", number)
    if number in weirdTimes:
        print(number, "is weird")
        continue
    x, fs = librosa.load(audioFolder + '/' + filename)
    print("fs is ", fs)
    mfccs = librosa.feature.mfcc(x,sr=fs)
    mfccs = mfccs[1:]
    mfccs = sklearn.preprocessing.scale(mfccs,axis=1) #apparently this does it across rows, same as np yay
    print("dims: ", np.shape(mfccs))
    #np.save("Documents/mfccs/" + filename[:-4], mfccs)
    #plt.figure()
    #librosa.display.specshow(mfccs,sr=fs,x_axis='time')
    #plt.show()

