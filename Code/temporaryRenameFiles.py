import numpy as numpy
import os

files = os.listdir('../Features/CQT3')
for f in files:
    numberString = f[-7:-4]
    print("numberString ", numberString)
    number = str(int(numberString))
    
    os.rename('../Features/CQT3/'+f, '../Features/CQT3/'+number+"features.npy")