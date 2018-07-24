import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing
from dataFunctions import *
import argparse
import time



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
training = torch.rand(100,60).to(DEVICE)
targetsBeat = torch.zeros(100).long().random_(0,3).to(DEVICE)
targetsChord = torch.zeros(100).long().random_(0,25).to(DEVICE)

#INIT MODEL
model1 = LSTMAny(60, 80, 50, 2).to(DEVICE)
model1.apply(init_weight)
modelBeat = LSTMAny(50, 25, 3, 2).to(DEVICE)
modelBeat.apply(init_weight)
modelChord = LSTMAny(50, 25, 25, 2).to(DEVICE) #for now let's just include no chord? ehhhhh
modelChord.apply(init_weight)

#LOSS
loss_function_beat = nn.CrossEntropyLoss().to(DEVICE)
loss_function_chord = nn.CrossEntropyLoss().to(DEVICE)

#OPTIMIZER
optimizer = optim.Adam(list(model1.parameters())+list(modelBeat.parameters())+list(modelChord.parameters()), lr=0.001)

#RUN MODEL
for epoch in range(10):
    #resetting stuff
    model1.zero_grad()
    model1.hidden = model1.init_hidden()
    modelBeat.zero_grad()
    modelBeat.hidden = modelBeat.init_hidden()
    modelChord.zero_grad()
    modelChord.hidden = modelChord.init_hidden()

    #run models
    intermediate = model1(training)
    beatTag = modelBeat(intermediate)
    chordTag = modelChord(intermediate)

    #beatTag, chordTag = model1(training)

    tag = []
    #calculate loss
    loss_beat = loss_function_beat(beatTag, targetsBeat)
    loss_chord = loss_function_chord(chordTag, targetsChord)
    loss = 0.5*(loss_beat+loss_chord)

    #gradient and optimize
    loss.backward()
    optimizer.step()

    print("loss beat", epoch, " was ", loss_beat.item())
    print("loss chord ", epoch, " was ", loss_chord.item())
    print("loss ", epoch, " was ", loss.item(), "\n")
    


