This was trained on our original MoreFeatures (3 spectrograms with first order time splicing, first order difference, chroma, and tempogram) under the architecture:

featuresAndGT = getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords=True)

model1 = LSTMAny(featuresAndGT[0][0].shape[1], 80, 50, 2).to(DEVICE)
modelBeat = LSTMAny(50, 25, 2, 2).to(DEVICE)
modelChord = LSTMAny(50, 25, 2, 25).to(DEVICE)

intermediate = model1(features)
beatTag = modelBeat(intermediate)
chordTag = modelChord(intermediate)
loss = (loss_function(beatTag, targetsBeat) + loss_function(chordTag, targetsChord))/2.0

Adam optimizer learning rate .001
