import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing

fs = 44100
hopSize = int(44100/100)
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

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
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



#CUDA!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#PREP TRAINING DATA WITH GROUND TRUTH
trainingMFCCs = []
mfccFolder = "newmfccs"
answerFolder = "../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
for filename in os.listdir(mfccFolder):
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
    trainingMFCCs.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).float()))  

    
#trainingMFCCs is now list of (mfcc, beatvector)


#INSTANTIATE THE MODEL
#numFeatures, 25 hidden dimens (is that what 25 units means?), 2 categories for 0 or 1 (no beat or beat)    
model = LSTMBeat(trainingMFCCs[0][0].shape[1], 25, 2).to(DEVICE)
model.apply(init_weight)
#Using cross entropy because we're softmaxing 
loss_function = nn.BCEWithLogitsLoss()
#Stochastic Gradient Descent, lr and momentum from paper
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)



#Select indices for training data vs. test data 
#let's do 80% train, 20% test
allIndices = np.arange(0, len(trainingMFCCs))
print("length of all indices ", len(allIndices))
indicesTraining = np.random.choice(allIndices,size=int(0.8*len(trainingMFCCs)), replace=False)
indicesTest = np.array([i for i in allIndices if i not in indicesTraining])
print("num training: ", len(indicesTraining))
print("num testing: ", len(indicesTest))


#evaluate before training on first song 
with torch.no_grad():
    target = trainingMFCCs[0][1]
    print("target size is ", target.shape)
    tag_scores = model(trainingMFCCs[0][0])
    print("tag shape is ", tag_scores.shape)
    print("original loss ", loss_function(tag_scores, target).item())

#TRAINING
averageLoss = np.inf
stopCount = 0
for epoch in range(500):
    i = 0


    mfccs, targets = trainingMFCCs[0]


    with torch.no_grad():
        #First 3 from training: 
        print("on first three TRAINING songs: ")
        for j in range(3):
            mfccs, targets = trainingMFCCs[indicesTraining[j]]
            tag_scores = model(mfccs)
            #tag_scores = F.softmax(tag_scores, 1)
            print("final loss ", loss_function(tag_scores, targets).item())
            #tag_scores = F.softmax(tag_scores, 1)
            print("max prob of beat ", tag_scores.max().item())
        #Now 3 test data
        print("On first three TEST songs: ")
        losses = [] 
        for j in range(len(indicesTest)):
            mfccs, targets = trainingMFCCs[indicesTest[j]]
            # print("number of 1s ", targets.sum())
            # print("percent ones is ", targets.sum().item()/float(len(targets)))
            tag_scores = model(mfccs)
            theloss = loss_function(tag_scores, targets).item()
            losses.append(theloss) 
            #tag_scores = F.softmax(tag_scores, 1)
            if j<3:
                print("final loss ", theloss)
                #tag_scores = F.softmax(tag_scores, 1)
                print("max prob of beat ", tag_scores.max().item())
        #if epoch %4==0:
        newAvgLoss = np.average(losses)
        if newAvgLoss < averageLoss:
            stopCount = 0
        else:
            print("didn't make progress this time")
            stopCount += 1



    for mfccs, targets in trainingMFCCs:
        if i%10 == 0:
            print(i)
        model.zero_grad()
        model.hidden = model.init_hidden()
        scores = model(mfccs)
        loss = loss_function(scores, targets)
        loss.backward()
        optimizer.step()
        i+=1
    
    print("done with epoch ", epoch)
    #EVALUATE PERFORMANCE EVERY EPOCH
    
    torch.save(model.state_dict(),'modelDictBCE.pth')
    
    if stopCount >= 20:
        print("We haven't made progress forever")
        break

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
