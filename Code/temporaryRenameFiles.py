import numpy as numpy
import os

# files = os.listdir('../Features/CQT2at10FPS')
# for f in files:
#     numberString = f[-7:-4]
#     print("numberString ", numberString)
#     number = str(int(numberString))
    
#     os.rename('../Features/CQT2at10FPS/'+f, '../Features/CQT2at10FPS/'+number+"features.npy")

files = os.listdir("../Features/BeatlesCQT3at10FPS")
for f in files:
    wantedName = f[18:]
    os.rename('../Features/BeatlesCQT3at10FPS/'+f, '../Features/BeatlesCQT3at10FPS/'+wantedName)