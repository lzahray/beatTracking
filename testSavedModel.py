import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing
#import torch.Tensor
#from lstmFirstAttempt import LSTMBeat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMBeat(nn.Module):
    def __init__(self, feature_dim, hidden_dim, tagset_size):
        super(LSTMBeat, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(6, 1, self.hidden_dim, device=DEVICE), torch.zeros(6, 1, self.hidden_dim, device=DEVICE))

    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space[:,1]

fs = 44100
hopSize = 441

#PREP TRAINING DATA WITH GROUND TRUTH
trainingMFCCs = []
mfccFolder = "newmfccs"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
filename = "RM-P036.npy"
#get song number
number = filename[4:7]

#load features we made in firstAttemptBeatTracking.py
features = np.load(mfccFolder+"/"+filename)
#I have tried with an without the following line (normalizing each feature for all time)
features = sklearn.preprocessing.scale(features,axis=0)
#open beat annotations file for this song
answerFile = open(answerFolder + "/RM-P"+number+".BEAT.TXT", "r")

#Get array of beat times as written in the annotations file (first column)
beatTimes = []
for line in answerFile.readlines():
    line = line.strip()
    beatTimes.append(line.split("\t")[0])

#Get beatTimes in seconds (in file they're in 10ms)
beatTimes = np.array(beatTimes).astype(float) * 0.01
#Turn seconds into feature frames
beatFrames = beatTimes * fs / hopSize
beatFrames = np.rint(beatFrames).astype(int)

#using beatFrames, get target vector of 1s and 0s
target = np.zeros(features.shape[0])
target[beatFrames] = 1

#append the features and target to trainingMFCCs
trainingMFCCs.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).long()))


model = LSTMBeat(trainingMFCCs[0][0].shape[1], 25, 2).to(DEVICE)
model.load_state_dict(torch.load("modelDictBCE.pth", map_location='cpu'))



answerFound = model(trainingMFCCs[0][0]).detach()
answerFound = F.sigmoid(answerFound).numpy()
print("answer found is ", answerFound)
answerReal = target
x = np.arange(target.shape[0])
plt.figure()
#plt.plot(x,answerFound, 'ro', markersize = 3)
plt.plot(x,answerFound)
#plt.plot(x,answerReal)
plt.show()