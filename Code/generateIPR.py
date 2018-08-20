import numpy as np 

fs = 44100
hopSize = 4410

#chordSequence = np.load("../JointGuesses/ChordVersion1/1.npy").astype(int)
chordSequence = np.load("../Features/ChordTargets10FPS/13chordTargets.npy").astype(int)
#chordSequence = np.load("../Results/Conv3Div10FPSB/ChordGuesses/chordGuess13.npy").astype(int)

modificationDict = {"dim":"min", "7":"maj","sus":"maj", "aug": "maj","hdi":"min", "9":"maj", "min":"min","maj":"maj"}
stringToIndex = {"N":0}
equivalence = [("A","Bbb","G##"), ("A#","Bb","Cbb"), ("B","Cb","A##"), ("C","B#","Dbb"), ("C#","Db","B##"), ("D","C##","Ebb"),("D#","Eb","Fbb"),("E","D##","Fb"),("F","E#","Gbb"),("F#","Gb","E##"),("G","F##","Abb"),("G#","Ab")]
number = 1
for item in equivalence:
    for note in item:
        stringToIndex[note+"min"] = number
        stringToIndex[note+"maj"] = number+1
    number += 2

aMinor = np.array([69, 72, 76])
aMajor = np.array([69,73,76])
indexToPitch = {0:[]}
number = 1
for i in range(24):
    indexToPitch[number] = aMinor + i
    indexToPitch[number+1] = aMajor + i
    number += 2

onsets = []
chordSequence[0] 
# text_file = open(toSaveFolder + "/beatInfo"+str(song) + ".txt", "w")
# for line in range(beatInfo.shape[0]):
#     text_file.write(str(beatInfo[line,0]) + "\t" + str(int(beatInfo[line,1])) + "\n")
# text_file.close()
indexOnsetOffsetTimes = []
lastChord = chordSequence[0] 
indexOnsetOffsetTimes.append([lastChord, 0])
for i in range(len(chordSequence)):
    if chordSequence[i] != lastChord:
        currentTime = hopSize / fs * i
        indexOnsetOffsetTimes[-1].append(currentTime) #offset 
        indexOnsetOffsetTimes.append([chordSequence[i],currentTime]) #chord and onset 
        lastChord = chordSequence[i]

indexOnsetOffsetTimes[-1].append((len(chordSequence)-1) * hopSize/fs)

#now we need to generate the file, getting the actual pitches
text_file = open("../ChordIPR/song13GT.txt", "w")
for i in range(len(indexOnsetOffsetTimes)):
    chord, onset, offset = indexOnsetOffsetTimes[i]
    pitches = indexToPitch[chord]
    for j in range(len(pitches)):
        text_file.write("0\t"+str(onset)+"\t"+str(offset)+"\t"+str(pitches[j]) + "\t" +"80\t80\t0\n")
text_file.close() 



    


