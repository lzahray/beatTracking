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
        self.lstm = nn.LSTM(feature_dim, hidden_dim,num_layers = 3, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(6, 1, self.hidden_dim, device=DEVICE), torch.zeros(6, 1, self.hidden_dim, device=DEVICE))

    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        #print("dim of lstm_out is ", lstm_out.shape)
        tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))
        #print("dim of tag space ", tag_space.shape)
        tag_scores = F.softmax(tag_space, dim=1)
        #print("dim of tag_scores ", tag_scores.shape)
        return tag_scores


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#PREP TRAINING DATA WITH GROUND TRUTH
trainingMFCCs = []
mfccFolder = "mfccs"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
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
    mfcc = mfcc.transpose()
    #ok the time is nigh, the time for the first order difference! 
    mfccAugmented = np.insert(mfcc, 0, 0, axis=0)
    firstOrderDiff = np.diff(mfccAugmented,axis=0)
    allFeatures = np.concatenate((mfcc,firstOrderDiff),axis=1)
    ###verify in matplotlib that we're not insane
    plt.imshow(allFeatures, interpolation="nearest", origin="upper", aspect="auto")


    #TESTING SOMETHING
    newFirst = firstOrderDiff.copy()
    newFirst[newFirst<0]=0
    newFirst = newFirst.sum(axis=1)
    print("newFirst shape ", newFirst.shape)
    plt.figure()
    plt.plot(np.arange(newFirst.shape[0]),newFirst)
    plt.plot(np.arange(len(target)), np.multiply(target, newFirst), 'ro')
    plt.show()




    #DONE TESTING    

    print("all features shape ", allFeatures.shape)
    print("target is ", target.shape)
    plt.figure()
    forPlot = np.tile(target, (allFeatures.shape[1],1)).transpose()
    np.set_printoptions(threshold=np.nan)
    #print(forPlot)
    #plt.figure()
    plt.imshow(forPlot,interpolation="nearest", origin="upper", aspect="auto")
    print("forPlot is ", forPlot.shape)
    #print(np.argwhere(forPlot)[:,0])
    rowsPlot = np.argwhere(forPlot)[:,0]
    #print("num xs ", xsPlot.shape)
    colsPlot = np.argwhere(forPlot)[:,1]
    #print("num ys ", ysPlot.shape)
    #plt.scatter(colsPlot, rowsPlot)
    

    trainingMFCCs.append((torch.from_numpy(allFeatures).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).long()))
#trainingMFCCs is now list of (mfcc, beatvector)


#PREP THE MODEL
#19 bins for mfcc, 25 hidden dimens because maybe that's what the paper means idk, 2 for 0 or 1 (no beat or beat)    
model = LSTMBeat(trainingMFCCs[0][0].shape[1], 25, 2).to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#need to evaluate before training to see how it does
with torch.no_grad():
    target = trainingMFCCs[0][1]
    tag_scores = model(trainingMFCCs[0][0])
    print("original loss ", loss_function(tag_scores, target))

    index = np.random.randint(80)
    x1 = x2 = np.arange(trainingMFCCs[index][0].shape[0])
    y1 = trainingMFCCs[index][1].cpu().numpy()
    y2 = model(trainingMFCCs[index][0]).cpu().numpy()[:,1]
    # plt.subplot(2,1,1)
    # plt.plot(x1,y1)
    # plt.ylabel('actual')
    # plt.subplot(2,1,2)
    # plt.plot(x2,y2)
    # plt.ylabel('lstm result')
    # plt.savefig("start")

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
    print("finished ", epoch, " epochs")
    #EVALUATE PERFORMANCE
    with torch.no_grad():
        for i in range(5):
            mfccs, targets = trainingMFCCs[i]
            tag_scores = model(mfccs)
            print("final loss ", loss_function(tag_scores, targets))
        if epoch %4==0:
            torch.save(model.state_dict(),'modelDict.pth')
        #ok let's just take a looksie at some random graph of one compared to actuality and spectrogram - sanity check folks!
        #random number between 0 and 80:
        # plt.figure() 
        # index = np.random.randint(80)
        # x1 = x2 = np.arange(trainingMFCCs[index][0].shape[0])
        # y1 = trainingMFCCs[index][1].cpu().numpy()
        # y2 = model(trainingMFCCs[index][0]).cpu().numpy()[:,1]
        # plt.subplot(2,1,1)
        # plt.plot(x1,y1)
        # plt.ylabel('actual')
        # plt.subplot(2,1,2)
        # plt.plot(x2,y2)
        # plt.ylabel('lstm result')
        # plt.savefig("epoch"+str(epoch))
