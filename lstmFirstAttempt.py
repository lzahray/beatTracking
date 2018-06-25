import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import sklearn

fs = 22050
class LSTMBeat(nn.Module):
    def __init__(self, feature_dim, hidden_dim, tagset_size):
        super(LSTMBeat, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim,num_layers = 1, bidirectional=False)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        #print("dim of lstm_out is ", lstm_out.shape)
        tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))
        #print("dim of tag space ", tag_space.shape)
        tag_scores = F.softmax(tag_space, dim=1)
        #print("dim of tag_scores ", tag_scores.shape)
        return tag_scores




#PREP TRAINING DATA WITH GROUND TRUTH
trainingMFCCs = []
mfccFolder = "mfccs"
answerFolder = "../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
for filename in os.listdir(mfccFolder):
    number = filename[4:7]
    mfcc = np.load(mfccFolder+"/"+filename)
    #we need to convert the times (10ms) to frames to correspond with mfcc
    #first step, actually get the np array of the times in 10ms
    answerFile = open(answerFolder + "/RM-P"+number+".BEAT.TXT", "r")
    beatTimes = []
    for line in answerFile.readlines():
        line = line.strip()
        beatTimes.append(line.split("\t")[0])

    #audio = librosa.load(audioFolder + "/RM-P"+number+".wav")[0]

    beatTimes = np.array(beatTimes).astype(float) * 0.01
    beatFrames = beatTimes * fs / 512
    beatFrames = np.rint(beatFrames).astype(int)
    
    #get target vector of 1s and 0s
    target = np.zeros(mfcc.shape[1])
    target[beatFrames] = 1
    #print("mfcc: ", mfcc)
    mfcc = mfcc.transpose()
    trainingMFCCs.append((torch.from_numpy(mfcc).float(), torch.from_numpy(target).long()))
#trainingMFCCs is now list of (mfcc, beatvector)

#PREP THE MODEL
#19 bins for mfcc, 12 hidden dimens because sure, 2 for 0 or 1 (no beat or beat)    
model = LSTMBeat(19, 12, 2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#need to evaluate before training to see how it does
with torch.no_grad():
    target = trainingMFCCs[0][1]
    tag_scores = model(trainingMFCCs[0][0])
    print("original loss ", loss_function(tag_scores, target))

#TRAINING
for epoch in range(300):
    i = 0
    for mfccs, targets in trainingMFCCs:
        if i%10 == 0:
            print(i)
        #print("NEXT SONG")
        mfccs = mfccs
        #print("mfccs dim ", mfccs)
        model.zero_grad()
        #print("zerod the grad")
        model.hidden = model.init_hidden()
        #print("did init hidden")
        #so apparently we need to switch the dimensions I think
        scores = model(mfccs)
        # if i%30==0:
        #     print("scores: ", scores)
        #print("found scores")
        #print("and target size is ", targets.shape)
        #print("score dim ", scores.shape)
        #print("targets dim ", targets.shape)
        #print("scores are ", scores)
        loss = loss_function(scores, targets)
        #print("found loss")
        loss.backward()
        #print("did backward on loss")
        optimizer.step()
        i += 1
    print("finished one epoch")
    #EVALUATE PERFORMANCE
    with torch.no_grad():
        for i in range(5):
            mfccs, targets = trainingMFCCs[i]
            tag_scores = model(mfccs)
            print("final loss ", loss_function(tag_scores, targets))
        #ok let's just take a looksie at some random graph of one compared to actuality and spectrogram - sanity check folks!
        #random number between 0 and 80: 
        index = np.random.randint(80)
        x1 = x2 = np.arange(trainingMFCCs[index][0].shape[0])
        y1 = trainingMFCCs[index][1].numpy()
        y2 = model(trainingMFCCs[index][0]).numpy()[:,1]
        plt.subplot(2,1,1)
        plt.plot(x1,y1)
        plt.ylabel('actual')
        plt.subplot(2,1,2)
        plt.plot(x2,y2)
        plt.ylabel('lstm result')
        plt.savefig("epoch"+epoch)
