import numpy as np

def saveSoftmaxToFile(song, softmax, toSaveFolder):
    print("shape of softmax is ", softmax.shape)
    #assuming softmax is time by 25 
    permutation = [8,10,12,14,16,18,20,22,24,2,4,6,7,9,11,13,15,17,19,21,23,1,3,5,0]
    rearranged = softmax[:,permutation]
    with open(toSaveFolder+str(song)+".txt", 'a') as the_file:
        for i in range(softmax.shape[0]):
            line = ''
            for j in range(softmax.shape[1]):
                line += str(rearranged[i,j])+'\t'
            the_file.write(line+'\n')
def getGuessFromFile(fileName):
    chords = np.loadtxt(fileName)[:-1]
    return getPermutedHMMToUs(chords)
    
def getPermutedHMMToUs(chords):
    permutation = [8,10,12,14,16,18,20,22,24,2,4,6,7,9,11,13,15,17,19,21,23,1,3,5,0]
    permDict = {}
    for i in range(len(permutation)):
        permDict[i] = permutation[i]
    for i in range(len(chords)):
        chords[i] = permDict[chords[i]]
    return chords

def getPermutedUsToHMM(chords):
    permutation = [8,10,12,14,16,18,20,22,24,2,4,6,7,9,11,13,15,17,19,21,23,1,3,5,0]
    permDict = {}
    for i in range(len(permutation)):
        permDict[permutation[i]] = i
    for i in range(len(chords)):
        chords[i] = permDict[chords[i]]
    return chords

def fmeasure(ground_truth, guess_values):
    gt = np.array(ground_truth)
    print("hi")
    guess = np.array(guess_values)
    guess_locations = np.nonzero(guess)[0]
    correct = 0
    falsePositives = 0
    correctPoints = []
    falseNegatives = 0
    for frame in range(len(gt)):
        if gt[frame] == 1:
            found = 0
            for poss in range(max(0,frame-7),min(frame+8,len(gt))):
                if poss in guess_locations:
                    found = 1
                    correctPoints.append(poss)
            if found:
                correct += 1
            else:
                falseNegatives += 1
    print("correct = ", correct)
    falsePositives = len(guess_locations)-correct
    fmeasure = 2*correct / (2*correct + falsePositives+falseNegatives)
    return fmeasure, correct, falsePositives, falseNegatives
        

