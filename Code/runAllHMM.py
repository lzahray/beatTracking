import subprocess
from os import walk
for root, dirs, files in walk("../ChordSoftmaxTextFiles/simpleJoint"):
   
    for f in files:
        print("f: ",f)
        subprocess.call(["../HMM/HMM_ViterbiEstimation-2", "../HMM/param_HMM.txt", "0.95", "0.01", "../ChordSoftmaxTextFiles/simpleJoint"+f, "../ChordSoftmaxTextFiles/simpleJointResults/"+f[-4]+".txt"])

