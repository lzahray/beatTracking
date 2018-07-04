##We keep rewriting the same code, it's ridiculous! 
import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMBeatMFCC(nn.Module):
    def __init__(self, feature_dim, hidden_dim, tagset_size):
        super(LSTMBeatMFCC, self).__init__()
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
        print("we are in hidden init")
        print("and our device is ", DEVICE)
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        print("we inited the first one")
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        print("we inited the second one ")
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))
        print("we inited them all")
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


def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print("found a linear")
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        print("found an lstm")
        for name, param in m.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.1, 0.1)
                # nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    # if classname.find('Conv1d') != -1:
    #     nn.init.kaiming_normal_(m.weight.data)
    #     if isinstance(m.bias, nn.parameter.Parameter):
    #         m.bias.data.fill_(0)



def getDataAndGroundTruth(mfccFolder, answerFolder, getSongsInOrder=False):
    listOfSongsInOrder = []
    trainingMFCCs = []
    fs = 44100
    hopSize = 441
    for filename in os.listdir(mfccFolder):
        #get song number
        number = filename[4:7]
        listOfSongsInOrder.append(number)

        #load features we made in firstAttemptBeatTracking.py
        features = np.load(mfccFolder+"/"+filename)
        #I have tried with an without the following line (normalizing each feature for all time)
        features = (features - features.mean()) / np.std(features)
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
        trainingMFCCs.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).float()))
    listOfSongsInOrder = np.array(listOfSongsInOrder)
    print("songsInOrder are ", listOfSongsInOrder) 
    if getSongsInOrder:
        return trainingMFCCs, listOfSongsInOrder
    else:
        return trainingMFCCs




# class LSTMBeatMel(nn.Module):
#     def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
#         super(LSTMBeatMel, self).__init__()
#         self.hidden_dim1 = hidden_dim1
#         print("hidden dim 1 is ", self.hidden_dim1)
#         self.hidden_dim2 = hidden_dim2
#         self.hidden_dim3 = hidden_dim3
#         self.feature_dim = feature_dim
#         #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
#         self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
#         self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
#         self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
#         #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
#         #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
#         self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
#         self.hidden1 = None
#         self.hidden2 = None
#         self.hidden3 = None
#         self.init_hidden()
    
#     def init_hidden(self):
#         self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
#         self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
#         self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))

#     def forward(self, mfcc):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out1, self.hidden1 = self.lstm1(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden1)
#         #print("shape of hidden is ", self.hidden1[0].shape)
#         #print("lstm_out1 shape is ", lstm_out1.shape)
#         #out1 = self.between12(lstm_out1)
#         lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
#         #print("lstm_out2 shape is ", lstm_out2.shape)
#         lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
#         #print("lstm_out3 shape is ", lstm_out3.shape)
#         tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
#         #print("tag space shape is ", tag_space.shape)
#         # tag_scores = F.softmax(tag_space, dim=1)
#         return tag_space[:,1]

# class LSTMBeatMFCC(nn.Module):
#     def __init__(self, feature_dim, hidden_dim, tagset_size):
#         super(LSTMBeatMFCC, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.feature_dim = feature_dim
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
#         self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
#         self.hidden = None
#         self.init_hidden()
    
#     def init_hidden(self):
#         self.hidden = (torch.zeros(6, 1, self.hidden_dim, device=DEVICE), torch.zeros(6, 1, self.hidden_dim, device=DEVICE))

#     def forward(self, mfcc):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))
#         # tag_scores = F.softmax(tag_space, dim=1)
#         return tag_space[:,1]