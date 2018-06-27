import numpy as np
import matplotlib.pyplot as plt
import librosa 
import sklearn
import os


weirdTimes = ['006', '034', '037', '038', '043', '050', '071', '072', '076', '077']
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
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

    answerFile = open(answerFolder + "/RM-P"+number+".BEAT.TXT", "r")
    beatTimes = []
    for line in answerFile.readlines():
        line = line.strip()
        beatTimes.append(line.split("\t")[0])
    beatTimes = np.array(beatTimes).astype(float) * 0.01
    beatFrames = beatTimes * fs / 512
    beatFrames = np.rint(beatFrames).astype(int)
    print("max beat frame is ", beatFrames[-1])
    target = np.zeros(mfccs.shape[1])
    target[beatFrames] = 1
    mfccs = mfccs.transpose()

    mfccAugmented = np.insert(mfccs, 0, 0, axis=0)
    print("shape of mfccAugmented ", mfccAugmented.shape)
    print("first row is ", mfccAugmented[0,:])
    firstOrderDiff = np.diff(mfccAugmented,axis=0)
    print("shape of firstOrderDiff ", firstOrderDiff.shape)
    halfWave = firstOrderDiff.copy()
    sumDiff = halfWave.sum(axis=1)
    #sumDiff[sumDiff < 0 ] = 0
    
    
    print("shape of sumDiff ", sumDiff.shape)
    

    plt.figure()
    plt.plot(np.arange(sumDiff.shape[0]),sumDiff)
    plt.plot(np.arange(len(target)), np.multiply(target, sumDiff), 'ro')
    

    plt.figure()
    plt.plot(np.arange(len(target)), target)
    plt.show()
    #ok now we need to test if the beats make any sense


