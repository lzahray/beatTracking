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
class LSTMBeatMel(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
        super(LSTMBeatMel, self).__init__()
        self.hidden_dim1 = hidden_dim1
        print("hidden dim 1 is ", self.hidden_dim1)
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.feature_dim = feature_dim
        #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
        self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
        #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
        #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
        self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.init_hidden()
    
    def init_hidden(self):
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))

    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        lstm_out1, self.hidden1 = self.lstm1(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden1)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #print("lstm_out1 shape is ", lstm_out1.shape)
        #out1 = self.between12(lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        #print("lstm_out2 shape is ", lstm_out2.shape)
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
        #print("lstm_out3 shape is ", lstm_out3.shape)
        tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
        #print("tag space shape is ", tag_space.shape)
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space[:,1]
fs = 44100
hopSize = 441

#PREP TRAINING DATA WITH GROUND TRUTH
trainingMFCCs = []
mfccFolder = "newmfccs"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
filename = "RM-P041.npy"
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


model = LSTMBeatMel(trainingMFCCs[0][0].shape[1], 96, 48, 24, 2).to(DEVICE)
model.load_state_dict(torch.load("modelDictMel/ModelDictMel78.pth", map_location='cpu'))



answerFound = model(trainingMFCCs[0][0]).detach()
answerFound = F.sigmoid(answerFound).numpy()
print("answer found is ", answerFound)
answerReal = target
x = np.arange(target.shape[0])
plt.figure()
#plt.plot(x,answerFound, 'ro', markersize = 3)
plt.plot(x,answerFound)
plt.plot(x,answerReal*answerFound, 'ro',markersize=3)
plt.show()