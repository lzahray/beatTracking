from collections import OrderedDict
import numpy as np
import dataFunctions

import torch.nn as nn
import torch
softmax = nn.Softmax(dim=1)
featureFolder = "../Features/mel128ChromaCQTWithTargets"
targetFolder = featureFolder
features, targetsBeat, targetsChord = dataFunctions.createFeaturesAndTargets(featureFolder, 1, False, False, targetFolder, mode="simpleJoint")
toSaveFile = "../intermediateSavedForPictures"
modelFile ="../beatsChordsLayerOne1HiddenOne100OutputOne75LayerOther2HiddenOther40/"
model1File = modelFile + "k0model1.pth"
modelBeatFile = modelFile+"k0modelBeat.pth"
modelChordFile = modelFile + "/k0modelChord.pth"
model = dataFunctions.createModel("simpleJointTempForImages",1540,{"output_model1":75, "hidden_dim_model1":100, "num_layers_model1":1, "hidden_dim_modelLeft":40, "num_layers_modelLeft":2, "hidden_dim_modelRight":40, "num_layers_modelRight":2})
#don't forget to edit soit creates correct simpleJoint
modelfordict = dataFunctions.createModel("simpleJoint",1540,{"output_model1":75, "hidden_dim_model1":100, "num_layers_model1":1, "hidden_dim_modelLeft":40, "num_layers_modelLeft":2, "hidden_dim_modelRight":40, "num_layers_modelRight":2})
modelfordict.modelLeft.load_state_dict(torch.load(modelBeatFile))
stateDictLeft = modelfordict.modelLeft.state_dict()
stateDictRight = modelfordict.modelRight.state_dict()
keys0 = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse']
keys1 = ['lstm.weight_ih_l1', 'lstm.weight_hh_l1', 'lstm.bias_ih_l1', 'lstm.bias_hh_l1', 'lstm.weight_ih_l1_reverse', 'lstm.weight_hh_l1_reverse', 'lstm.bias_ih_l1_reverse', 'lstm.bias_hh_l1_reverse']
keys1Mod = {'weight_ih_l1': 'weight_ih_l0', 'weight_hh_l1': 'weight_hh_l0', 'weight_ih_l1_reverse': 'weight_ih_l0_reverse', 'weight_hh_l1_reverse': 'weight_hh_l0_reverse', 'bias_ih_l1': 'bias_ih_l0', 'bias_hh_l1': 'bias_hh_l0', 'bias_ih_l1_reverse': 'bias_ih_l0_reverse', 'bias_hh_l1_reverse': 'bias_hh_l0_reverse'}

#d2 = OrderedDict([('__C__', v) if k == 'c' else (k, v) for k, v in d.items()])
theItems =  stateDictLeft.items()
print("original state dict keys ", stateDictLeft.keys())
for k, v in theItems:
    if k in keys0:
        newKey = k[0:4]+"1"+k[4:]
    elif k in keys1:
        newKey = k[0:4]+"2."+keys1Mod[k[5:]]
    else:
        newKey = k
    stateDictLeft = OrderedDict([(newKey, thev) if thek==k else (thek, thev) for thek,thev in stateDictLeft.items() ])

theItems =  stateDictRight.items()
for k, v in theItems:
    if k in keys0:
        newKey = k[0:4]+"1"+k[4:]
    elif k in keys1:
        newKey = k[0:4]+"2."+keys1Mod[k[5:]]
    else:
        newKey = k
    stateDictRight = OrderedDict([(newKey, thev) if thek==k else (thek, thev) for thek,thev in stateDictRight.items()])
print("stateDictRight keys are ", stateDictRight.keys())

model.model1.load_state_dict(torch.load(model1File))
model.modelLeft.load_state_dict(stateDictLeft)
model.modelRight.load_state_dict(stateDictRight)
with torch.no_grad():
    tag = model(features)
#np.save(toSaveFile+"/tag.npy",tag)
smBeat = softmax(tag[0]).detach().cpu().numpy()
smChord = softmax(tag[1]).detach().cpu().numpy()
np.save(toSaveFile+"/softmaxBeat.npy",smBeat)
np.save(toSaveFile+"/softmaxChord.npy",smChord)

