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

weirdTimes = ['006', '022', '028', '030', '034', '037', '038', '041',  '043', '050', '057', '071', '076', '077', '095']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device in data functions is ", DEVICE)
class LSTMAny(nn.Module):
    def __init__(self, feature_dim, hidden_dim, tagset_size, num_layers):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMAny, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.tagset_size = tagset_size
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = self.num_layers, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE), torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE))

    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        #print("running foward one time yo")
        #h = torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE)
        #c = torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE)
        #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0],1,mfcc.shape[1]), (h,c))

        tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))
        # tag_scores = F.softmax(tag_space, dim=1)
        if self.tagset_size == 2:
            return tag_space[:,1]
        else:
            return tag_space



class LSTMBeatMel(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
        super(LSTMBeatMel, self).__init__()
        self.hidden_dim1 = hidden_dim1
        #print("hidden dim 1 is ", self.hidden_dim1)
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
        # print("we are in hidden init")
        # print("and our device is ", DEVICE)
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        #print("we inited the first one")
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        #print("we inited the second one ")
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))
        #print("we inited them all")
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


class LSTMMulticlass3(LSTMBeatMel):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
        super(LSTMMulticlass3,self).__init__(feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size)
    def forward(self,features):
        lstm_out1, self.hidden1 = self.lstm1(features.view(features.shape[0], 1, features.shape[1]), self.hidden1)
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
        return tag_space

class LSTMMulticlass4(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, tagset_size):
        super(LSTMMulitclass4,self).__init__()
        self.hidden_dim1 = hidden_dim1
        #print("hidden dim 1 is ", self.hidden_dim1)
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.feature_dim = feature_dim
        #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
        self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
        self.lstm4 = nn.LSTM(hidden_dim3*2, hidden_dim4, num_layers = 1, bidirectional=True)
        #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
        #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
        self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.hidden4 = None
        self.init_hidden()
        
        self.hidden2tag = nn.Linear(hidden_dim4*2, tagset_size)
    def init_hidden(self):
        # print("we are in hidden init")
        # print("and our device is ", DEVICE)
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        #print("we inited the first one")
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        #print("we inited the second one ")
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))
        self.hidden4 = (torch.zeros(2, 1, self.hidden_dim4, device=DEVICE), torch.zeros(2, 1, self.hidden_dim4, device=DEVICE))
        #print("we inited them all")
    def forward(self,features):
        lstm_out1, self.hidden1 = self.lstm1(features.view(features.shape[0], 1, features.shape[1]), self.hidden1)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #print("lstm_out1 shape is ", lstm_out1.shape)
        #out1 = self.between12(lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        #print("lstm_out2 shape is ", lstm_out2.shape)
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
        lstm_out4, self.hidden4 = self.lstm4(lstm_out3, self.hidden4)
        #print("lstm_out3 shape is ", lstm_out3.shape)
        tag_space = self.hidden2tag(lstm_out4.view(lstm_out3.shape[0], -1))
        #print("tag space shape is ", tag_space.shape)
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space


# class LSTMDownbeatBatchSimple(nn.Module):
#     def __init__(self, feature_dim, hidden_dim, tagset_size, batch_size):
#         super(LSTMDownbeatBatchSimple, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_dim = hidden_dim
#         self.feature_dim = feature_dim
#         self.tagset_size = tagset_size
#         #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True)
#         #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
#         #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
#         self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
#         self.init_hidden()
    
#     def init_hidden(self):
#         # print("we are in hidden init")
#         # print("and our device is ", DEVICE)
#         self.hidden = (torch.zeros(2, self.batch_size, self.hidden_dim, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim, device=DEVICE))
#         #print("we inited them all")

#     def forward(self, features):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out1, self.hidden1 = self.lstm1(features, self.hidden1)
#         #print("shape of hidden is ", self.hidden1[0].shape)
#         #print("lstm_out1 shape is ", lstm_out1.shape)
#         #out1 = self.between12(lstm_out1)
#         lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
#         #print("lstm_out2 shape is ", lstm_out2.shape)
#         lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
#         #print("lstm_out3 shape is ", lstm_out3.shape)

#         #right now lstm_out3 has output shape (seq_len, batch_size, hidden3* 2)
#         tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
#         #print("tag space shape is ", tag_space.shape)
#         # tag_scores = F.softmax(tag_space, dim=1)
#         #this should be seq_length, batch, 
#         return tag_space


# class LSTMDownbeatBatch(nn.Module):
#     def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size, batch_size):
#         super(LSTMDownbeatBatch, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_dim1 = hidden_dim1
#         #print("hidden dim 1 is ", self.hidden_dim1)
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
#         # print("we are in hidden init")
#         # print("and our device is ", DEVICE)
#         self.hidden1 = (torch.zeros(2, self.batch_size, self.hidden_dim1, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim1, device=DEVICE))
#         #print("we inited the first one")
#         self.hidden2 = (torch.zeros(2, self.batch_size, self.hidden_dim2, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim2, device=DEVICE))
#         #print("we inited the second one ")
#         self.hidden3 = (torch.zeros(2, self.batch_size, self.hidden_dim3, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim3, device=DEVICE))
#         #print("we inited them all")
#     def forward(self, features):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out1, self.hidden1 = self.lstm1(features, self.hidden1)
#         #print("shape of hidden is ", self.hidden1[0].shape)
#         #print("lstm_out1 shape is ", lstm_out1.shape)
#         #out1 = self.between12(lstm_out1)
#         lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
#         #print("lstm_out2 shape is ", lstm_out2.shape)
#         lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
#         #print("lstm_out3 shape is ", lstm_out3.shape)

#         #right now lstm_out3 has output shape (seq_len, batch_size, hidden3* 2)
#         tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
#         #print("tag space shape is ", tag_space.shape)
#         # tag_scores = F.softmax(tag_space, dim=1)
#         #this should be seq_length, batch, 
#         return tag_space


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
            beatTimes.append(line.split("\t")[0]) #beware, this /t is sketch idk what it was before, check on github

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


def getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = False, songNumber=None):
    #this one will return stuff in the song numerical order 
    #it also assumes stuff has already been normalized 
    fs = 44100
    hopSize = 441
    featuresAndGT = []
    if songNumber:
        rangeNum = [songNumber, songNumber+1]
    else:
        rangeNum = [1,101] 
    for i in range(rangeNum[0],rangeNum[1]):
        numberString = '0'*(3-len(str(i))) + str(i)
        if numberString not in weirdTimes:
            features = np.load(featureFolder + "/RM-P" + numberString + '.npy')
            beatTimes = []
            downbeatTimes = []
            answerFile = open(answerFolder + "/RM-P"+numberString+".BEAT.TXT", "r")
            for line in answerFile.readlines():
                line = line.strip()
                lineElements = line.split("\t")
                beatTimes.append(lineElements[0]) 
                if lineElements[2] == '384':
                    downbeatTimes.append(lineElements[0])
            answerFile.close()
            #Get beatTimes in seconds (in file they're in 10ms)
            beatTimes = np.array(beatTimes).astype(float) * 0.01
            downbeatTimes = np.array(downbeatTimes).astype(float) * 0.01
            #Turn seconds into feature frames
            beatFrames = beatTimes * fs / hopSize
            downbeatFrames = downbeatTimes * fs / hopSize
            beatFrames = np.rint(beatFrames).astype(int)
            downbeatFrames = np.rint(downbeatFrames).astype(int)
            #using beatFrames, get target vector of 1s and 0s
            target = np.zeros(features.shape[0])
            target[beatFrames] = 1
            target[downbeatFrames] = 2
            #print("num downbeats is ", list(target).count(2))
            #print("num beats is ",list(target).count(1) )
            if getChords:
                targetChords = getGroundTruthChords(features.shape[0], i, "../../CHORD/RWC_Pop_Chords")
                #print("targetChords shape ", targetChords.shape)
                #print("target beats shape is ", target.shape)
                #featuresAndGT.append((features,target,targetChords))
                featuresAndGT.append((torch.from_numpy(features).float(), torch.from_numpy(target).long(), torch.from_numpy(targetChords).long()))
            else:
                featuresAndGT.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).long()))          
    return featuresAndGT
    #ok forget about batches for now, really not a priority    

#screw it, this function will get a single chord for the song specified 
def getGroundTruthChords(numFrames, songNumber, answerFolder):
    #features are just being passed in so we know the length of the song which sounds dumb but... well too late
    #print("in the func")
    fs = 44100
    hopSize = 441
    modificationDict = {"dim":"N", "7":"maj","sus":"N", "aug": "N","hdi":"min", "9":"maj", "min":"min","maj":"maj"}
    stringToIndex = {"N":0}
    equivalence = [("A","Bbb","G##"), ("A#","Bb","Cbb"), ("B","Cb","A##"), ("C","B#","Dbb"), ("C#","Db","B##"), ("D","C##","Ebb"),("D#","Eb","Fbb"),("E","D##","Fb"),("F","E#","Gbb"),("F#","Gb","E##"),("G","F##","Abb"),("G#","Ab")]
    number = 1
    for item in equivalence:
        for note in item:
            stringToIndex[note+"min"] = number
            stringToIndex[note+"maj"] = number+1
        number += 2
    files = [answerFolder+"/"+file for file in sorted(os.listdir(answerFolder)) if file[-3:]=="lab"]
    answerFile = open(files[songNumber-1])
    chordGT = np.zeros(numFrames)
    lines = answerFile.readlines()
    answerFile.close()
    #print("num lines is ", len(lines))
    for line in lines:
        info = line.strip().split("\t")
        #print("info is ", info)
        if info[2] == "N":
            category = 0
        else:
            s = info[2].split(":")
            note = s[0]
            if len(s[1]) == 1:
                mod = s[1]
            else:
                if s[1][0] == "7" or s[1][0] == "9":
                    mod = modificationDict[s[1][0]]
                else:
                    mod = s[1][:3]
            if modificationDict[mod] == "N":
                newString = "N"
            else:
                newString = s[0] + modificationDict[mod]
            category = stringToIndex[newString]
            #print("parsed ", newString)
            #print("category ", category)
        startFrame = int(round(float(info[0])* fs / hopSize))
        endFrame = int(round(float(info[1]) * fs / hopSize))
        # print("start frame: ", startFrame)
        # print("end frame ", endFrame)
        chordGT[startFrame:min(endFrame+1, chordGT.shape[0])] = category
    # plt.plot(np.arange(numFrames), chordGT)
    # plt.show()
    return chordGT


        

