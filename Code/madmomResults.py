import madmom
import numpy as np

weirdTimes = [6, 22, 28, 30, 34, 37, 38, 41, 43, 50, 57, 71, 76, 77, 95]

listOfSongsInOrder = np.arange(1,101)
listOfSongsInOrder = [i for i in listOfSongsInOrder if i not in weirdTimes]

audioFolder = "/n/sd1/music/RWC-MDB/P/wav"
targetFolder = "../Features/mel128ChromaCQTWithTargets/"
#toSaveFolder = "../JointGuesses/BeatVersion1/"
toSaveFolder = "../Madmom/"
#well that was pretty damn dumb, it's ok we put it in excel
proc = madmom.features.downbeats.RNNDownBeatProcessor()
proc2 = madmom.features.downbeats.DBNDownBeatTrackingProcessor(4, fps=100)
fmeasures = []
for song in listOfSongsInOrder:
    songString = '0'*(3-len(str(song))) + str(song)
    print(songString)
    targets = np.load(targetFolder+str(song)+"beatTargets.npy")
    baf = proc(audioFolder+"/RM-P"+songString+".wav")
    beatInfo = proc2.process(baf)

    annotationsBeats = np.where(targets != 0)[0] / 100.0
    annotationsDownbeats = np.where(targets == 2)[0] /100.0

    fmeasureBeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsBeats)
    fmeasureDownbeats = madmom.evaluation.beats.BeatEvaluation(beatInfo, annotationsDownbeats, downbeats=True)
    print("beats: ", fmeasureBeats)
    print("downbeats: ", fmeasureDownbeats)

    fmeasures.append([float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
    print("just appended ",[float(str(fmeasureBeats)[11:16]), float(str(fmeasureDownbeats)[11:16])])
    np.save(toSaveFolder+"fmeasures",fmeasures)
    
    text_file = open(toSaveFolder + str(song) + ".txt", "w")
    for line in range(beatInfo.shape[0]):
        text_file.write(str(beatInfo[line,0]) + "\t" + str(int(beatInfo[line,1])) + "\n")
    text_file.close()