import numpy as np
import matplotlib.pyplot as plt
import librosa 
import sklearn
import os

np.set_printoptions(threshold=np.nan)
weirdTimes = ['006', '034', '037', '038', '043', '050', '071', '072', '076', '077']
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
mfccFolder = "newmfccs"
hopSize = int(44100/100)
fs = 44100
print("list: ", os.listdir(audioFolder))
for filename in os.listdir(mfccFolder):
    number = filename[4:7]
    print("number is ", number)
    features = np.load(mfccFolder+"/"+filename)
    features = sklearn.preprocessing.scale(features,axis=0) #let's just try this
    print("dims: ", np.shape(features))

    answerFile = open(answerFolder + "/RM-P"+number+".BEAT.TXT", "r")
    beatTimes = []
    for line in answerFile.readlines():
        line = line.strip()
        beatTimes.append(line.split("\t")[0])
    beatTimes = np.array(beatTimes).astype(float) * 0.01
    beatFrames = beatTimes * fs / hopSize
    beatFrames = np.rint(beatFrames).astype(int)
    print("max beat frame is ", beatFrames[-1])
    target = np.zeros(features.shape[0])
    target[beatFrames] = 1
    print(target.shape)
    print(target.sum())
    indices = np.argwhere(target==1)
    #print(indices)
    points = []
    for thing in indices:
        for i in range(0,features.shape[1]):
            points.append([thing[0],i])
    points = np.array(points)
    #print(points)
    print("min is ", np.amin(features), " max is ", np.amax(features))
    plt.figure()
    plt.imshow(features, interpolation="nearest", origin="upper", aspect="auto", cmap="Greys", vmin=np.amin(features),vmax=np.amax(features))
    plt.plot(points[:,1],points[:,0], 'ro',markersize=0.3)
    plt.show()
    #ok now we need to test if the beats make any sense


