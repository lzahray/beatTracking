import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing
import dataFunctions
import madmom
import signal



class MyTimeoutError(Exception):
    def __init__(self):
        pass

def handler(signum, frame):
    print("In handler")
    raise MyTimeoutError()

signal.signal(signal.SIGALRM, handler)
weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]
namesForBeatFiles = np.arange(1,101)
namesForBeatFiles = [i for i in namesForBeatFiles if i not in weirdTimes]
print("names are ", namesForBeatFiles)



np.set_printoptions(threshold=np.nan)
fileToSaveFMeasures = "../254C/254CFMeasures"
np.save(fileToSaveFMeasures, np.array([]))
print("imported")
#featureFolder = "melspectrograms"
featureFolder = "../Features/moreFeatures"
answerFolder = "../../../Downloads/AIST.RWC-MDB-P-2001.BEAT"
#modelFiles = ["modelDictMelCross/k0epoch37.pth", "modelDictMelCross/k1epoch25.pth", "modelDictMelCross/k2epoch22.pth", "modelDictMelCross/k3epoch27.pth", "modelDictMelCross/k4epoch31.pth"]
modelFiles = ["../254C/k"+str(i)+".pth" for i in range(5)]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#first off, need to get beat activation function and answers 

#featuresAndGT = dataFunctions.getDataAndGroundTruth(featureFolder,answerFolder)
featuresAndGT = dataFunctions.getMoreFeaturesAndGroundTruthDownbeats(featureFolder,answerFolder)
print("shape of features ", featuresAndGT[0][0].shape)
print("got the data")
#testIndices =  [2,  7, 12, 21, 33, 35, 40, 49, 54, 58, 59, 60, 64, 67, 69, 70, 77, 85]
#testIndices = np.arange(20)
#Now, let's generate the model
# model = dataFunctions.LSTMBeat(featuresAndGT[0][0].shape[1], 25, 2).to(DEVICE)
# model.load_state_dict(torch.load(modelFile, map_location='cpu'))



#model = dataFunctions.LSTMBeatMel(featuresAndGT[0][0].shape[1], 96, 48, 24, 2).to(DEVICE).eval()
with torch.no_grad():
    model = dataFunctions.LSTMDownbeat4(featuresAndGT[0][0].shape[1], 25, 25, 25,25, 3).to(DEVICE).eval()
#Now we load the model file which is the last stage we did that should get low loss

print("model is ready")
loss_function = nn.CrossEntropyLoss()


boundaryPoint = int(0.2*85)+1
allIndices = np.arange(0, 86)
testIndicesAll = [allIndices[k*boundaryPoint:min((k+1)*boundaryPoint, len(allIndices))] for k in range(5)]

fmeasures = []
for k in range(4,5):
    print("k = ", k)
    testIndices = testIndicesAll[k]
    
    
    modelFile = modelFiles[k]
    with torch.no_grad():
        model.load_state_dict(torch.load(modelFile, map_location="cpu"))
    #model.load_state_dict(torch.load(modelFile))
    print("test indices: ", testIndices)
    for i in testIndices:
        #if i == 6 or i==22 or i==24 or i==46 or i==55:
        #if i==54 or i==0:
        if True: #we'll want to do this for certain indices later but this for now
            signal.alarm(30)
            # try:
            print("index ", i)
            print("Song ", str(namesForBeatFiles[i]))
            #We're just seeing how it does on the TRAINING data
            #print("next index")  
            targets = featuresAndGT[i][1]
            #print("has targets")
            # model.init_hidden()
            # print("inited hidden")
            with torch.no_grad():
                answerFound = model(featuresAndGT[i][0]).detach()
            #print("shape of answerFound is ", answerFound.shape)
            # plt.figure()
            # plt.plot(np.arange(answerFound.shape[0]), answerFound)
            # plt.show()
            #print("found the answer")
            #loss = loss_function(answerFound, targets).item()
            #print("the error cross entropy is ", loss)
            #print("first answer found is ", answerFound[0,:])
            m = nn.Softmax(dim=1)
            baf = m(answerFound)
            #print("first baf is ", baf[0,:])
            baf = baf[:,1:].numpy()
            print("max of beats is ", max(baf[:,0]))
            print("max of downbeats is ", max(baf[:,1]))
            # plt.plot(np.arange(baf.shape[0]), baf[:,1])
            # temp = targets.numpy().copy()
            # print("temp not 0", temp[temp!=0])
            # temp[temp<2]=0
            # temp[temp==2]=1
            # plt.plot(np.arange(baf.shape[0]), temp*baf[:,1],'ro')
            # plt.show()
            #print("baf shape is ", baf.shape)
            #baf = F.sigmoid(answerFound).cpu().numpy()
            

            ##baf = F.sigmoid(answerFound)
            # plt.figure()
            # plt.plot(np.arange(len(baf)),baf)
            # plt.plot(np.arange(featuresAndGT[i][1].numpy().shape[0]),featuresAndGT[i][1].numpy())
            # plt.show()
            # plt.show()
            #print("baf shape is ", baf.shape)
            #print("about to run madmom")
        
            #proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100, threshold=0.2, max_bpm = 200)
            proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(4, fps=100)
            #print("ran madmom")
            #beatInfo = proc(baf)
            beatInfo = proc.process(baf)
            #print("beats ", beatInfo[:6,:])
            #print("other beats ", otherBeatInfo[:6,:])
            #prep targets (get times in seconds of first just beats, then downbeats)
            annotationsBeats = np.where(targets != 0)[0] / 100.0
            #print("type of annotations is ", type(annotationsBeats))
            annotationsDownbeats = np.where(targets == 2)[0] /100.0


            fmeasureBeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsBeats)
            fmeasureDownbeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsDownbeats, downbeats=True)
            
            #save in the format [[beatF,downbeatF]]
            # oldFile = list(np.load(fileToSaveFMeasures+".npy"))
            # print("length of old file is ", len(oldFile))
            # assert(len(oldFile) >= i)
            # if len(oldFile) > i:
            #     oldFile[i] = [fmeasureBeats,fmeasureDownbeats]
            # elif len(oldFile) == i:
            #     oldFile.append([fmeasureBeats,fmeasureDownbeats])
            fmeasures.append([float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
            print("just appended ",[float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
            np.save(fileToSaveFMeasures+"k4",np.array(fmeasures))
            
            print("beats:  ", fmeasureBeats)
            print("downbeats: ", fmeasureDownbeats)

            #now save the beat file for this song
            #txt in the format time in seconds, tab, number
            text_file = open("../254C/BeatFiles/" + str(namesForBeatFiles[i]) + ".txt", "w")
            for line in range(beatInfo.shape[0]):
                text_file.write(str(beatInfo[line,0]) + "\t" + str(int(beatInfo[line,1])) + "\n")
            text_file.close()


            # beatGuessInS = [beatInfo[myTime,0] for myTime in range(beatInfo.shape[0]) if beatInfo[myTime,1]!=1]
            # downbeatGuessInS = [beatInfo[myTime,0] for myTime in range(beatInfo.shape[0]) if beatInfo[myTime,1]==1]
            # beatLocations = (beatGuessInS*100).astype(int)
            # downbeatLocations = ((downbeatGuessInS*100).astype(int))
            # except MyTimeoutError:
            #     print("It's taking way too long folks")
            #     continue
            # except MemoryError:
            #     print("memory error sob")
            #     continue
            # finally:
            #     signal.alarm(0)
            



            

            #COMMENT FOR NOW
            #Step 1: Autocorrelation
            # frameShifts = np.arange(27,151) #max tempo to min tempo 
            # autoM = np.zeros((len(frameShifts),len(baf)))
            # for j in range(len(frameShifts)):
            #     autoM[j,:] = np.concatenate((np.zeros(frameShifts[j]), baf[:len(baf)-frameShifts[j]])) 
            # #print("autoM is ", autoM)
            # #print("shape of autoM is ", autoM.shape)
            # autoCorrelation = np.dot(autoM, baf)
            # #print("autoCorrelation is ", autoCorrelation)
            # hamm = np.hamming(15) / np.sum(np.hamming(15))
            # #print("shape of autoCorrelation ", autoCorrelation.shape)
            # autoCorrelation = np.convolve(autoCorrelation, hamm, mode="same")
            # #print("autoCorrelation after hamm is ", autoCorrelation)
            # #Find best tempo, for now just do one tempo for the whole song, will change later
            # Tstar = frameShifts[np.argmax(autoCorrelation)]
            # print("Tstar is ", Tstar)
            # #Find best phase
            # maxSum = 0
            # pStar = 0
            # for p in range(Tstar):
            #     newSum = np.sum(baf[p::Tstar])
            #     if newSum > maxSum:
            #         maxSum = newSum
            #         pStar = p
            # print("pStar is ", pStar)
            # d = int(0.1*Tstar)
            # #centerPoints = bafs[i][pStar::Tstar]
            # centerIndices = np.arange(len(baf))[pStar::Tstar]
            # beatLocations = []
            # #so we want to redo this whole process at every new beat I think? Well I mean we can do whatever we want...
            # #you know really I think we should be dynammic programming or something let's take a looksie at the other papers
            # #it's annoying cuz it's like this is what we did in class... but worse because my nn sucks

            # #Ok 2016 uses a Dynammic Bayesian Network that sounds fun
            # for j in centerIndices:
            #     #for k in range(max(j-d,0), min(j+d,len(baf)-1)):
            #     if baf[max(j-d,0)] <= baf[j] and baf[j] >= baf[min(j+d,len(baf)-1)]:
            #         beatLocations.append(j)
            #     #beatLocations.append(np.argmax(baf[max(j-d,0):min(j+d,len(baf)-1)]) + max(j-d,0))
            #END COMMENT FOR NOW

            #DETECTING F-MEASURE
    #         target = featuresAndGT[i][1]
    #         correct = 0
    #         falsePositives = 0
    #         correctPoints = []
    #         falseNegatives = 0
    #         for frame in range(target.shape[0]):
    #             #print("looking at frame ", frame)
    #             if target[frame] == 1:
    #                 foundBeat = 0
    #                 for possibility in range(max(0,frame-7), min(frame+8,target.shape[0])):
    #                     if possibility in beatLocations:
    #                         foundBeat = 1
    #                         correctPoints.append(possibility)
    #                 if foundBeat:
    #                     correct += 1
    #                 else:
    #                     falseNegatives += 1 #false negative - it said there wasn't a beat but there was
    #                     #print("did not find a beat, we looked at ", np.arange(max(0,frame-7), min(frame+8,target.shape[0])))
    #         falsePositives = len(beatLocations)-correct
    #         fmeasure = 2*correct / (2*correct + falsePositives + falseNegatives)
    #         #print("correct: ", correct)
    #         #print("falsePositives: ", falsePositives)
    #         #print("falseNegatives: ", falseNegatives)
    #         print("f measure:  ", fmeasure)
    #         fmeasures.append(fmeasure)
    #         falsePosPoints = [p for p in beatLocations if p not in correctPoints]
    #         #print("length of list is ", len(falsePosPoints), " and should be ", falsePositives)
    #         #print("false pos points ", falsePosPoints)
    #         plt.figure()
    #         #plt.plot(beatLocations, np.ones(len(beatLocations)), 'ro')
    #         plt.plot(correctPoints, np.ones(len(correctPoints)), 'ro')
    #         plt.plot(falsePosPoints, np.ones(len(falsePosPoints)), 'bo')
    #         plt.plot(np.arange(featuresAndGT[i][1].numpy().shape[0]),featuresAndGT[i][1].numpy())
    #         print("mean so far is ", np.mean(fmeasures))
    #         plt.show()


    # print("mean of these f measures is ", np.mean(fmeasures))