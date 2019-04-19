##We keep rewriting the same code, it's ridiculous! 
import numpy as np
import matplotlib.pyplot as plt
#import librosa 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing

weirdTimes = ['006', '022', '028', '030', '034', '037', '038', '041',  '043', '050', '057', '071', '076', '077', '095']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device in data functions is ", DEVICE)
toSaveImages = "../intermediateSavedForPictures/"
lstmCount = 0
names = ["left.npy", "right.npy"]
class LSTMAny(nn.Module):
    def __init__(self, feature_dim, tagset_size, hyper_parameters):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMAny, self).__init__()
        self.hidden_dim = hyper_parameters["hidden_dim"]
        self.feature_dim = feature_dim
        self.num_layers = hyper_parameters["num_layers"]
        self.tagset_size = tagset_size
        #sometimes we've had dropout but I think not for most times that worked
        self.lstm = nn.LSTM(feature_dim, self.hidden_dim, num_layers = self.num_layers, bidirectional=True)

        self.hidden2tag = nn.Linear(self.hidden_dim*2, tagset_size)
        self.init_hidden()
        self.num_losses = 1
    
    def init_hidden(self):
        self.hidden = (torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE), torch.zeros(2*self.num_layers, 1, self.hidden_dim, device=DEVICE))

    def forward(self, mfcc):
        self.init_hidden()
        lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))

        # tag_scores = F.softmax(tag_space, dim=1)

        return tag_space
    
    def calculate_loss(self, loss_functions, tag, targets ):
        loss_function = loss_functions[0]
        loss = loss_function(tag, targets)
        return [loss]
class LSTMAnyTempForImage(nn.Module):
    def __init__(self, feature_dim, tagset_size, hyper_parameters, name):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMAnyTempForImage, self).__init__()
        self.hidden_dim = hyper_parameters["hidden_dim"]
        self.feature_dim = feature_dim
        self.num_layers = 2
        self.tagset_size = tagset_size
        self.name = name
        #sometimes we've had dropout but I think not for most times that worked
        self.lstm1 = nn.LSTM(feature_dim, self.hidden_dim, num_layers = 1, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, num_layers = 1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim*2, tagset_size)
        self.init_hidden()
        self.num_losses = 1
    
    def init_hidden(self):
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim, device=DEVICE), torch.zeros(2, 1, self.hidden_dim, device=DEVICE))
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim, device=DEVICE), torch.zeros(2, 1, self.hidden_dim, device=DEVICE))
    def forward(self, mfcc):
        self.init_hidden()
        lstm_out1, self.hidden1 = self.lstm1(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden1)
        np.save(toSaveImages+self.name, lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        tag_space = self.hidden2tag(lstm_out2.view(mfcc.shape[0], -1))

        # tag_scores = F.softmax(tag_space, dim=1)

        return tag_space
    
    def calculate_loss(self, loss_functions, tag, targets ):
        loss_function = loss_functions[0]
        loss = loss_function(tag, targets)
        return [loss]

#yeah we'll have to make sure this works.....
class LSTMSimpleJoint(nn.Module):
    def __init__(self, feature_dim, tagset_sizes, hyper_parameters):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMSimpleJoint, self).__init__()
        print("hyper parameters ", hyper_parameters)
        print("output mod 1 ", hyper_parameters["output_model1"])
        print("hidden dim 1 ", hyper_parameters["hidden_dim_model1"])
        print("num layers 1 ", hyper_parameters["num_layers_model1"])
        self.model1 = LSTMAny(feature_dim, hyper_parameters["output_model1"], {"hidden_dim":hyper_parameters["hidden_dim_model1"], "num_layers": hyper_parameters["num_layers_model1"]}).to(DEVICE)
        self.model1.init_hidden()
        self.modelLeft = LSTMAny(hyper_parameters["output_model1"], tagset_sizes["left"], {"hidden_dim":hyper_parameters["hidden_dim_modelLeft"], "num_layers": hyper_parameters["num_layers_modelLeft"]}).to(DEVICE)
        self.modelLeft.init_hidden()
        self.modelRight = LSTMAny(hyper_parameters["output_model1"], tagset_sizes["right"], {"hidden_dim":hyper_parameters["hidden_dim_modelRight"], "num_layers": hyper_parameters["num_layers_modelRight"]}).to(DEVICE)
        self.modelRight.init_hidden()
        self.num_losses = 3

    def forward(self, features):
        #hidden will already be inited in the forward of each dude
        intermediate = self.model1(features)
        leftTag = self.modelLeft(intermediate)
        rightTag = self.modelRight(intermediate)
        return [leftTag, rightTag]
    
    def calculate_loss(self, loss_functions, tag, targets):
        leftTag, rightTag = tag
        leftTargets, rightTargets = targets
        loss_function_left, loss_function_right = loss_functions
        loss_left = loss_function_left(leftTag, leftTargets)
        loss_right = loss_function_right(rightTag, rightTargets)
        loss = 0.5 * loss_left + 0.5 * loss_right
        return [loss, loss_left, loss_right]
class LSTMSimpleJointTempForImage(nn.Module):
    def __init__(self, feature_dim, tagset_sizes, hyper_parameters):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMSimpleJointTempForImage, self).__init__()
        print("hyper parameters ", hyper_parameters)
        print("output mod 1 ", hyper_parameters["output_model1"])
        print("hidden dim 1 ", hyper_parameters["hidden_dim_model1"])
        print("num layers 1 ", hyper_parameters["num_layers_model1"])
        self.model1 = LSTMAny(feature_dim, hyper_parameters["output_model1"], {"hidden_dim":hyper_parameters["hidden_dim_model1"], "num_layers": hyper_parameters["num_layers_model1"]}).to(DEVICE)
        self.model1.init_hidden()
        self.modelLeft = LSTMAnyTempForImage(hyper_parameters["output_model1"], tagset_sizes["left"], {"hidden_dim":hyper_parameters["hidden_dim_modelLeft"], "num_layers": 2},"left.npy").to(DEVICE)
        self.modelLeft.init_hidden()
        self.modelRight = LSTMAnyTempForImage(hyper_parameters["output_model1"], tagset_sizes["right"], {"hidden_dim":hyper_parameters["hidden_dim_modelRight"], "num_layers": 1},"right.npy").to(DEVICE)
        self.modelRight.init_hidden()
        self.num_losses = 3

    def forward(self, features):
        #hidden will already be inited in the forward of each dude
        intermediate = self.model1(features)
        print("about to save intermediate to ", toSaveImages+"intermediate.npy")
        np.save(toSaveImages+"intermediate.npy",intermediate) 
        leftTag = self.modelLeft(intermediate)
        rightTag = self.modelRight(intermediate)
        return [leftTag, rightTag]
    
    def calculate_loss(self, loss_functions, tag, targets):
        leftTag, rightTag = tag
        leftTargets, rightTargets = targets
        loss_function_left, loss_function_right = loss_functions
        loss_left = loss_function_left(leftTag, leftTargets)
        loss_right = loss_function_right(rightTag, rightTargets)
        loss = 0.5 * loss_left + 0.5 * loss_right
        return [loss, loss_left, loss_right]

class LSTMComplexJoint(nn.Module):
    def __init__(self, feature_dim, tagset_sizes, hyper_parameters):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMComplexJoint, self).__init__()
        self.model1 = LSTMAny(feature_dim, hyper_parameters["output_model1"], {"hidden_dim":hyper_parameters["hidden_dim_model1"], "num_layers": hyper_parameters["num_layers_model1"]}).to(DEVICE)
        self.model1.init_hidden()

        self.modelLeft = LSTMAny(hyper_parameters["output_model1"], tagset_sizes["left"], {"hidden_dim":hyper_parameters["hidden_dim_modelLeft"], "num_layers": hyper_parameters["num_layers_modelLeft"]}).to(DEVICE)
        self.modelLeft.init_hidden()

        self.modelCenter = LSTMAny(hyper_parameters["output_model1"] + tagset_sizes["left"], tagset_sizes["center"], {"hidden_dim":hyper_parameters["hidden_dim_modelCenter"], "num_layers": hyper_parameters["num_layers_modelCenter"]}).to(DEVICE)
        self.modelCenter.init_hidden()

        self.modelRight = LSTMAny(hyper_parameters["output_model1"] + tagset_sizes["left"] + tagset_sizes["center"], tagset_sizes["right"], {"hidden_dim":hyper_parameters["hidden_dim_modelRight"], "num_layers": hyper_parameters["num_layers_modelRight"]}).to(DEVICE)
        self.modelRight.init_hidden()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.num_losses = 4

    def forward(self, features):
        #hidden will already be inited in the forward of each dude
        intermediate = self.model1(features)

        leftTag = self.modelLeft(intermediate)

        if self.modelLeft.tagset_size == 2:
            #print("sigmoiding left")
            forCenter = torch.cat((intermediate, self.sigmoid(leftTag) ), 1).to(DEVICE)
        else:
            #print("softmaxing left")
            forCenter = torch.cat((intermediate, self.softmax(leftTag)), 1).to(DEVICE)
        centerTag = self.modelCenter(forCenter)

        if self.modelCenter.tagset_size == 2: 
            #print("sigmoiding center")
            forRight = torch.cat((forCenter, self.sigmoid(centerTag)), 1).to(DEVICE)
        else:
            #print("softmaxing center")
            forRight = torch.cat((forCenter, self.softmax(centerTag)),1).to(DEVICE)
        rightTag = self.modelRight(forRight)

        return [leftTag, centerTag, rightTag]
    
    def calculate_loss(self, loss_functions, tag, targets):
        leftTag, centerTag, rightTag = tag
        leftTargets, centerTargets, rightTargets = targets
        loss_function_left, loss_function_center, loss_function_right = loss_functions
        #We are assuming that if the tagset of one of the models is 2, the user will 
        #and pass in the binary cross entropy loss function with logits! 
        if self.modelLeft.tagset_size == 2:
            loss_left = loss_function_left(leftTag[:,1], leftTargets.float())
        else:
            loss_left = loss_function_left(leftTag, leftTargets)
        if self.modelCenter.tagset_size == 2:
            loss_center = loss_function_center(centerTag[:,1], centerTargets)
        else:
            loss_center = loss_function_center(centerTag, centerTargets)
        if self.modelRight.tagset_size == 2:
            loss_right = loss_function_right(rightTag[:,1], rightTargets.float())
        else:
            loss_right = loss_function_right(rightTag, rightTargets)
        loss = (loss_left + loss_center + loss_right) / 3.0
        return [loss, loss_left, loss_right, loss_center]

#we can fix this later to take more than just chord, but for now let's just do chord
class ConvolutionAndLSTM(nn.Module):
    def __init__(self, feature_dim, tagset_sizes, hyper_parameters):
        super(ConvolutionAndLSTM, self).__init__()
        #right now feature_dim isn't used but might be in the future 
        self.num_losses = 1
        self.hyper_parameters = hyper_parameters
        self.num_divisions = hyper_parameters.get("num_divisions",1)
        self.num_octaves =  hyper_parameters.get("num_octaves",7)
        self.octave_group = hyper_parameters.get("octave_group",4) #we'll make 4 our kernal size in this direction
        #For each of these shapes, we'll do 2 channels of convolution
        self.num_out_channels = hyper_parameters.get("num_out_channels",3)
        self.conv1 = nn.Conv2d(1,self.num_out_channels,(self.num_octaves, self.num_divisions), stride=(1,self.num_divisions) ).to(DEVICE)
        self.conv3 = nn.Conv2d(1,self.num_out_channels,(self.octave_group, self.num_divisions*3), stride=(1,self.num_divisions) ).to(DEVICE) #4 is just some # of octaves I like for no reason
        self.conv5 = nn.Conv2d(1,self.num_out_channels,(self.octave_group, self.num_divisions*5), stride=(1,self.num_divisions) ).to(DEVICE)
        
        self.feature_dim_conv1 = self.num_out_channels*12 ###
        self.feature_dim_conv3 = (self.num_octaves-self.octave_group+1)*self.num_out_channels*12
        self.feature_dim_conv5 = (self.num_octaves-self.octave_group+1)*self.num_out_channels*12
        self.feature_dim_conv = self.feature_dim_conv1+self.feature_dim_conv3+self.feature_dim_conv5
        print("feature_dim_conv ", self.feature_dim_conv)
        #To test only conv1
        #self.feature_dim_conv = self.feature_dim_conv5

        self.lstm = LSTMAny(self.feature_dim_conv, tagset_sizes, {"hidden_dim": hyper_parameters["hidden_dim"], "num_layers":hyper_parameters["num_layers"]}).to(DEVICE)

        self.lstm.init_hidden()

    def forward(self, features):
        out1 = self.conv1(features[:,:,:,0:12*self.num_divisions])
        out1 = out1.view((out1.shape[0], self.feature_dim_conv1))

        out3 = self.conv3(features[:,:,:,0:(12+3-1)*self.num_divisions])
        #print("out3 natural shape ", out3.shape)
        out3 = out3.view((out3.shape[0], self.feature_dim_conv3))

        out5 = self.conv5(features[:,:,:,0:(12+5-1)*self.num_divisions])
        out5 = out5.view((out5.shape[0], self.feature_dim_conv5))

        #To test one of conv1, conv2, or conv3, comment out the following line
        features_lstm = torch.cat((out1,out3,out5),1)
        #and uncomment the following line (change to out3 or out5 if desired)
        #features_lstm = out5
        #print("lstmfeature shape ", features_lstm.shape)

        #self.lstm does its own init_hidden() within its forward method, see LSTMAny class 
        final_tag = self.lstm(features_lstm)
        #print("final tag shape ", final_tag.shape)
        return final_tag
    
    def calculate_loss(self,loss_functions, tag, targets):
        return self.lstm.calculate_loss(loss_functions, tag, targets )


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, feature_dim, tagset_size, hyper_parameters):
        super(BiLSTM_CRF, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = 25
        self.stop_tag = 26
        self.num_losses = 1
        #make this convolution and lstm
        self.lstm = ConvolutionAndLSTM(feature_dim, tagset_size, hyper_parameters).to(DEVICE)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(DEVICE)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        #START TAG is 26
        #STOP TAG is 27
        self.transitions.data[self.start_tag, :] = -10000
        self.transitions.data[:, self.stop_tag] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(DEVICE)
        # START_TAG has all of the score.
        init_alphas[0][self.start_tag] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.stop_tag]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, features):
        lstm_feats = self.lstm(features)
        return lstm_feats

    def _score_sequence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(DEVICE)
        tags = torch.cat([torch.tensor([self.start_tag], dtype=torch.long).to(DEVICE), tags]).to(DEVICE)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.stop_tag, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        #added a toDevice
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(DEVICE)
        init_vvars[0][self.start_tag] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.stop_tag]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.start_tag  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, origFeatures, tags):
        feats = self._get_lstm_features(origFeatures)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sequence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



# class LSTMAnyFlexible(nn.Module):
#     def __init__(self, feature_dim, tagset_size, hyper_parameters):
#         #can have any number of layers but they need to have the same hidden size which for now is fine 
#         super(LSTMAny, self).__init__()
#         self.hidden_dim = hyper_parameters["hidden_dim"] #this is now a list! Huzzah! 
#         self.feature_dim = feature_dim
#         self.num_layers = hyper_parameters["num_layers"]
#         self.tagset_size = tagset_size
#         self.lstm = [nn.LSTM(feature_dim, self.hidden_dim[0], num_layers = 1, bidirectional=True)]
#         for i in range(len(1,self.hidden_dim)):
#             self.lstm.append(nn.LSTM(self.hidden_dim[i-1]*2, self.hidden_dim[i], num_layers = 1, bidirectional=True))
#         self.hidden = []
#         self.init_hidden()
#         self.num_losses = 1

#     def init_hidden(self):
#         for i in range(len(self.hidden_dim)):
#             self.hidden.append(torch.zeros(2, 1, self.hidden_dim[i], device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))

#     def forward(self, mfcc):
#         self.init_hidden()
#         for 
#         lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         tag_space = self.hidden2tag(lstm_out.view(mfcc.shape[0], -1))

#         # tag_scores = F.softmax(tag_space, dim=1)

#         return tag_space
    
#     def calculate_loss(self, loss_functions, tag, targets ):
#         loss_function = loss_functions[0]
#         loss = loss_function(tag, targets)
#         return [loss]



class LSTMAny3(nn.Module):
    def __init__(self, feature_dim, tagset_size, hyper_parameters):
        #can have any number of layers but they need to have the same hidden size which for now is fine 
        super(LSTMAny3, self).__init__()
        self.hidden_dim = hyper_parameters["hidden_dims"]
        self.output_model1 = hyper_parameters.get("output_model1",self.hidden_dim[0])
        print("output 1 is ", self.output_model1)
        assert(len(self.hidden_dim) == 3)
        self.feature_dim = feature_dim
        self.num_layers = hyper_parameters["num_layers"]
        assert(self.num_layers == 3)
        self.tagset_size = tagset_size
        self.lstm1 = LSTMAny(feature_dim, self.output_model1, {"num_layers":1, "hidden_dim":self.hidden_dim[0]})
        self.lstm2 = LSTMAny(self.output_model1, tagset_size, {"num_layers": 2, "hidden_dim":self.hidden_dim[1]})
        self.num_losses = 1
    
    def forward(self, features):
        lstm_out1 = self.lstm1(features)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #out1 = self.between12(lstm_out1)
        tag_space = self.lstm2(lstm_out1)

        # tag_scores = F.softmax(tag_space, dim=1)

        return tag_space
    
    def calculate_loss(self, loss_functions, tag, targets ):
        loss_function = loss_functions[0]
        loss = loss_function(tag, targets)
        return [loss]









class LSTMBeatMel(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
        super(LSTMBeatMel, self).__init__()
        self.hidden_dim1 = hidden_dim1
        #print("hidden dim 1 is ", self.hidden_dim1)
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.feature_dim = feature_dim
        #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
        self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
        #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
        #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
        self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.init_hidden()
    
    def init_hidden(self):
        # print("we are in hidden init")
        # print("and our device is ", DEVICE)
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        #print("we inited the first one")
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        #print("we inited the second one ")
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))
        #print("we inited them all")
    def forward(self, mfcc):
        #mfcc is axis 0=time, axis 1=features
        #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
        lstm_out1, self.hidden1 = self.lstm1(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden1)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #print("lstm_out1 shape is ", lstm_out1.shape)
        #out1 = self.between12(lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        #print("lstm_out2 shape is ", lstm_out2.shape)
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
        #print("lstm_out3 shape is ", lstm_out3.shape)
        tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
        #print("tag space shape is ", tag_space.shape)
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space[:,1]


class LSTMMulticlass3(LSTMBeatMel):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size):
        super(LSTMMulticlass3,self).__init__(feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size)
    def forward(self,features):
        lstm_out1, self.hidden1 = self.lstm1(features.view(features.shape[0], 1, features.shape[1]), self.hidden1)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #print("lstm_out1 shape is ", lstm_out1.shape)
        #out1 = self.between12(lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        #print("lstm_out2 shape is ", lstm_out2.shape)
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
        #print("lstm_out3 shape is ", lstm_out3.shape)
        tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
        #print("tag space shape is ", tag_space.shape)
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space

class LSTMMulticlass4(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, tagset_size):
        super(LSTMMultitclass4,self).__init__()
        self.hidden_dim1 = hidden_dim1
        #print("hidden dim 1 is ", self.hidden_dim1)
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.feature_dim = feature_dim
        #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
        self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
        self.lstm4 = nn.LSTM(hidden_dim3*2, hidden_dim4, num_layers = 1, bidirectional=True)
        #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
        #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
        self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.hidden4 = None
        self.init_hidden()
        
        self.hidden2tag = nn.Linear(hidden_dim4*2, tagset_size)
    def init_hidden(self):
        # print("we are in hidden init")
        # print("and our device is ", DEVICE)
        self.hidden1 = (torch.zeros(2, 1, self.hidden_dim1, device=DEVICE), torch.zeros(2, 1, self.hidden_dim1, device=DEVICE))
        #print("we inited the first one")
        self.hidden2 = (torch.zeros(2, 1, self.hidden_dim2, device=DEVICE), torch.zeros(2, 1, self.hidden_dim2, device=DEVICE))
        #print("we inited the second one ")
        self.hidden3 = (torch.zeros(2, 1, self.hidden_dim3, device=DEVICE), torch.zeros(2, 1, self.hidden_dim3, device=DEVICE))
        self.hidden4 = (torch.zeros(2, 1, self.hidden_dim4, device=DEVICE), torch.zeros(2, 1, self.hidden_dim4, device=DEVICE))
        #print("we inited them all")
    def forward(self,features):
        lstm_out1, self.hidden1 = self.lstm1(features.view(features.shape[0], 1, features.shape[1]), self.hidden1)
        #print("shape of hidden is ", self.hidden1[0].shape)
        #print("lstm_out1 shape is ", lstm_out1.shape)
        #out1 = self.between12(lstm_out1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        #print("lstm_out2 shape is ", lstm_out2.shape)
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
        lstm_out4, self.hidden4 = self.lstm4(lstm_out3, self.hidden4)
        #print("lstm_out3 shape is ", lstm_out3.shape)
        tag_space = self.hidden2tag(lstm_out4.view(lstm_out3.shape[0], -1))
        #print("tag space shape is ", tag_space.shape)
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space


# class LSTMDownbeatBatchSimple(nn.Module):
#     def __init__(self, feature_dim, hidden_dim, tagset_size, batch_size):
#         super(LSTMDownbeatBatchSimple, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_dim = hidden_dim
#         self.feature_dim = feature_dim
#         self.tagset_size = tagset_size
#         #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True)
#         #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
#         #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
#         self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
#         self.init_hidden()
    
#     def init_hidden(self):
#         # print("we are in hidden init")
#         # print("and our device is ", DEVICE)
#         self.hidden = (torch.zeros(2, self.batch_size, self.hidden_dim, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim, device=DEVICE))
#         #print("we inited them all")

#     def forward(self, features):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out1, self.hidden1 = self.lstm1(features, self.hidden1)
#         #print("shape of hidden is ", self.hidden1[0].shape)
#         #print("lstm_out1 shape is ", lstm_out1.shape)
#         #out1 = self.between12(lstm_out1)
#         lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
#         #print("lstm_out2 shape is ", lstm_out2.shape)
#         lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
#         #print("lstm_out3 shape is ", lstm_out3.shape)

#         #right now lstm_out3 has output shape (seq_len, batch_size, hidden3* 2)
#         tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
#         #print("tag space shape is ", tag_space.shape)
#         # tag_scores = F.softmax(tag_space, dim=1)
#         #this should be seq_length, batch, 
#         return tag_space


# class LSTMDownbeatBatch(nn.Module):
#     def __init__(self, feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, tagset_size, batch_size):
#         super(LSTMDownbeatBatch, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_dim1 = hidden_dim1
#         #print("hidden dim 1 is ", self.hidden_dim1)
#         self.hidden_dim2 = hidden_dim2
#         self.hidden_dim3 = hidden_dim3
#         self.feature_dim = feature_dim
#         #self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = 3, bidirectional=True, dropout = 0.5)
#         self.lstm1 = nn.LSTM(feature_dim, hidden_dim1, num_layers = 1, bidirectional=True)
#         self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers = 1, bidirectional=True)
#         self.lstm3 = nn.LSTM(hidden_dim2*2, hidden_dim3, num_layers = 1, bidirectional=True)
#         #self.between12 = nn.Linear(hidden_dim1*2, hidden_dim2*2)
#         #self.between23 = nn.Linear(hidden_dim2*2, hidden_dim3*2)
#         self.hidden2tag = nn.Linear(hidden_dim3*2, tagset_size)
#         self.hidden1 = None
#         self.hidden2 = None
#         self.hidden3 = None
#         self.init_hidden()
    
#     def init_hidden(self):
#         # print("we are in hidden init")
#         # print("and our device is ", DEVICE)
#         self.hidden1 = (torch.zeros(2, self.batch_size, self.hidden_dim1, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim1, device=DEVICE))
#         #print("we inited the first one")
#         self.hidden2 = (torch.zeros(2, self.batch_size, self.hidden_dim2, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim2, device=DEVICE))
#         #print("we inited the second one ")
#         self.hidden3 = (torch.zeros(2, self.batch_size, self.hidden_dim3, device=DEVICE), torch.zeros(2, self.batch_size, self.hidden_dim3, device=DEVICE))
#         #print("we inited them all")
#     def forward(self, features):
#         #mfcc is axis 0=time, axis 1=features
#         #lstm_out, self.hidden = self.lstm(mfcc.view(mfcc.shape[0], 1, mfcc.shape[1]), self.hidden)
#         lstm_out1, self.hidden1 = self.lstm1(features, self.hidden1)
#         #print("shape of hidden is ", self.hidden1[0].shape)
#         #print("lstm_out1 shape is ", lstm_out1.shape)
#         #out1 = self.between12(lstm_out1)
#         lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
#         #print("lstm_out2 shape is ", lstm_out2.shape)
#         lstm_out3, self.hidden3 = self.lstm3(lstm_out2, self.hidden3)
#         #print("lstm_out3 shape is ", lstm_out3.shape)

#         #right now lstm_out3 has output shape (seq_len, batch_size, hidden3* 2)
#         tag_space = self.hidden2tag(lstm_out3.view(lstm_out3.shape[0], -1))
#         #print("tag space shape is ", tag_space.shape)
#         # tag_scores = F.softmax(tag_space, dim=1)
#         #this should be seq_length, batch, 
#         return tag_space

def createModel(mode, numFeatures, hyper_parameters):
    if mode == "justBeat": 
        #just for now
        model = LSTMAny(numFeatures, 3, hyper_parameters ).to(DEVICE)
        #model = LSTMAny3(numFeatures, 3, hyper_parameters ).to(DEVICE)
    elif mode == "justChord":
        model = LSTMAny(numFeatures, 25, hyper_parameters ).to(DEVICE)
    elif mode == "simpleJoint":
        model = LSTMSimpleJoint(numFeatures, {"left":3,"right":25}, hyper_parameters).to(DEVICE)
    elif mode == "simpleJointTempForImages":
        model = LSTMSimpleJointTempForImage(numFeatures, {"left":3,"right":25}, hyper_parameters).to(DEVICE)
    elif mode == "complexJoint":
        model = LSTMComplexJoint(numFeatures, {"left":2, "center": 25, "right": 2}, hyper_parameters).to(DEVICE)
    elif mode == "conv":
        model = ConvolutionAndLSTM(numFeatures, 25, hyper_parameters).to(DEVICE)
    elif mode == "crf":
        model = BiLSTM_CRF(numFeatures, 27, hyper_parameters)
    return model

def createFeaturesAndTargets(featureFolder, song, chord_ground_truth, beat_ground_truth, targetFolder, mode=None):
    features = torch.from_numpy(np.load(featureFolder+"/"+str(song)+"features.npy")).float().to(DEVICE)
    targetsBeat = torch.from_numpy(np.load(targetFolder+"/"+str(song)+"beatTargets.npy")).long().to(DEVICE)
    targetsChord = torch.from_numpy(np.load(targetFolder+"/"+str(song)+"chordTargets.npy")).long().to(DEVICE)
    
    if chord_ground_truth:
        #here we will concat the chord features
        chordsFeat = np.zeros((targetsChord.shape[0], 25))
        chordsFeat[np.arange(targetsChord.shape[0]), targetsChord] = 1
        features = torch.from_numpy(np.column_stack((features, stackNFeatures(chordsFeat,11)))).float().to(DEVICE)
        #print("are chords nonzero? sum across time is ", np.sum(chordsFeat,axis=0))
    if beat_ground_truth:
        #here we will concat the beat features
        beatsFeat = np.zeros((targetsBeat.shape[0], 3))
        beatsFeat[np.arange(targetsBeat.shape[0]), targetsBeat] = 1
        features = torch.from_numpy(np.column_stack((features,stackNFeatures(beatsFeat,11)))).float().to(DEVICE)
        print("are beats nonzero? sum across time is ", np.sum(beatsFeat,axis=0))
    if mode == "conv" or mode =="crf":
        features = features.view((features.shape[0],1, features.shape[1],features.shape[2]))
        #print("feature shape ", features.shape)
    return [features, targetsBeat, targetsChord]

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print("found a linear")
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        print("found an lstm")
        for name, param in m.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.1, 0.1)
                # nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv2d') != -1:
        print("found a conv")
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)



def getDataAndGroundTruth(mfccFolder, answerFolder, getSongsInOrder=False):
    listOfSongsInOrder = []
    trainingMFCCs = []
    fs = 44100
    hopSize = 441
    for filename in os.listdir(mfccFolder):
        #get song number
        number = filename[4:7]
        listOfSongsInOrder.append(number)

        #load features we made in firstAttemptBeatTracking.py
        features = np.load(mfccFolder+"/"+filename)
        #I have tried with an without the following line (normalizing each feature for all time)
        features = (features - features.mean()) / np.std(features)
        #open beat annotations file for this song
        answerFile = open(answerFolder + "/RM-P"+number+".BEAT.TXT", "r")

        #Get array of beat times as written in the annotations file (first column)
        beatTimes = []
        for line in answerFile.readlines():
            line = line.strip()
            beatTimes.append(line.split("\t")[0]) #beware, this /t is sketch idk what it was before, check on github

        #Get beatTimes in seconds (in file they're in 10ms)
        beatTimes = np.array(beatTimes).astype(float) * 0.01
        #Turn seconds into feature frames
        beatFrames = beatTimes * fs / hopSize
        beatFrames = np.rint(beatFrames).astype(int)

        #using beatFrames, get target vector of 1s and 0s
        target = np.zeros(features.shape[0])
        target[beatFrames] = 1

        #append the features and target to trainingMFCCs
        trainingMFCCs.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).float()))
    listOfSongsInOrder = np.array(listOfSongsInOrder)
    print("songsInOrder are ", listOfSongsInOrder) 
    if getSongsInOrder:
        return trainingMFCCs, listOfSongsInOrder
    else:
        return trainingMFCCs


def getMoreFeaturesAndGroundTruthDownbeats(featureFolder, answerFolder, getChords = False, songNumber=None):
    #this one will return stuff in the song numerical order 
    #it also assumes stuff has already been normalized 
    fs = 44100
    hopSize = 441
    featuresAndGT = []
    if songNumber:
        rangeNum = [songNumber, songNumber+1]
    else:
        rangeNum = [1,101] 
    for i in range(rangeNum[0],rangeNum[1]):
        numberString = '0'*(3-len(str(i))) + str(i)
        if True: #lisa october edit for all features 
        #if numberString not in weirdTimes:
            features = np.load(featureFolder + "/RM-P" + numberString + '.npy')
            beatTimes = []
            downbeatTimes = []
            answerFile = open(answerFolder + "/RM-P"+numberString+".BEAT.TXT", "r")
            for line in answerFile.readlines():
                line = line.strip()
                lineElements = line.split("\t")
                beatTimes.append(lineElements[0]) 
                if lineElements[2] == '384':
                    downbeatTimes.append(lineElements[0])
            answerFile.close()
            #Get beatTimes in seconds (in file they're in 10ms)
            beatTimes = np.array(beatTimes).astype(float) * 0.01
            downbeatTimes = np.array(downbeatTimes).astype(float) * 0.01
            #Turn seconds into feature frames
            beatFrames = beatTimes * fs / hopSize
            downbeatFrames = downbeatTimes * fs / hopSize
            beatFrames = np.rint(beatFrames).astype(int)
            downbeatFrames = np.rint(downbeatFrames).astype(int)
            #using beatFrames, get target vector of 1s and 0s
            target = np.zeros(features.shape[0])
            target[beatFrames] = 1
            target[downbeatFrames] = 2
            #print("num downbeats is ", list(target).count(2))
            #print("num beats is ",list(target).count(1) )
            if getChords:
                targetChords = getGroundTruthChords(features.shape[0], i, "../../CHORD/RWC_Pop_Chords")
                #print("targetChords shape ", targetChords.shape)
                #print("target beats shape is ", target.shape)
                #featuresAndGT.append((features,target,targetChords))
                featuresAndGT.append((torch.from_numpy(features).float(), torch.from_numpy(target).long(), torch.from_numpy(targetChords).long()))
            else:
                featuresAndGT.append((torch.from_numpy(features).to(DEVICE).float(), torch.from_numpy(target).to(DEVICE).long()))          
    return featuresAndGT
    #ok forget about batches for now, really not a priority    

#screw it, this function will get a single chord for the song specified 
def getGroundTruthChords(numFrames, songNumber, answerFolder, hopSize = 441):
    #features are just being passed in so we know the length of the song which sounds dumb but... well too late
    #print("in the func")
    fs = 44100
    #hopSize = 441
    modificationDict = {"dim":"min", "7":"maj","sus":"maj", "aug": "maj","hdi":"min", "9":"maj", "min":"min","maj":"maj"}
    stringToIndex = {"N":0}
    equivalence = [("A","Bbb","G##"), ("A#","Bb","Cbb"), ("B","Cb","A##"), ("C","B#","Dbb"), ("C#","Db","B##"), ("D","C##","Ebb"),("D#","Eb","Fbb"),("E","D##","Fb"),("F","E#","Gbb"),("F#","Gb","E##"),("G","F##","Abb"),("G#","Ab")]
    number = 1
    for item in equivalence:
        for note in item:
            stringToIndex[note+"min"] = number
            stringToIndex[note+"maj"] = number+1
        number += 2
    files = [answerFolder+"/"+file for file in sorted(os.listdir(answerFolder)) if file[-3:]=="lab"]
    print("found ", len(files), " files")
    answerFile = open(files[songNumber-1])
    print("file name is ", files[songNumber-1])
    chordGT = np.zeros(numFrames)
    lines = answerFile.readlines()
    print("found ", len(lines), " lines")
    answerFile.close()
    #print("num lines is ", len(lines))
    for line in lines:
        info = line.strip().split("\t")
        #info = line.strip().split(" ")
        #print("info is ", info)
        if info[2] == "N":
            category = 0
        else:
            s = info[2].split(":")
            note = s[0]
            if len(s) == 1:
                mod = "maj"
            else:
                if len(s[1]) == 1:
                    mod = s[1]
                else:
                    if s[1][0] == "7" or s[1][0] == "9":
                        mod = modificationDict[s[1][0]]
                    else:
                        mod = s[1][:3]
            if modificationDict.get(mod,"maj") == "N":
                newString = "N"
            else:
                s[0] = s[0].split("/")[0]
                newString = s[0] + modificationDict.get(mod,"maj")
            category = stringToIndex[newString]
            #print("parsed ", newString)
            #print("category ", category)
        startFrame = int(round(float(info[0])* fs / hopSize))
        endFrame = int(round(float(info[1]) * fs / hopSize))
        # print("start frame: ", startFrame)
        # print("end frame ", endFrame)
        chordGT[startFrame:min(endFrame+1, chordGT.shape[0])] = category
    # plt.plot(np.arange(numFrames), chordGT)
    # plt.show()
    return chordGT


        
def stackNFeatures(originalFeatureVec, N):
    #N is total and should be odd, we're doing 11
    toReturn = np.zeros((originalFeatureVec.shape[0], originalFeatureVec.shape[1]*N))
    paddedZeros = np.concatenate((np.concatenate((np.zeros((int(N/2), originalFeatureVec.shape[1])), originalFeatureVec), axis=0), np.zeros((int(N/2), originalFeatureVec.shape[1]))), axis=0) 
    for i in range(N):
            placeLeft = originalFeatureVec.shape[1]*i
            placeRight = originalFeatureVec.shape[1]*(1+i)
            toReturn[:,placeLeft:placeRight] = paddedZeros[i:i+originalFeatureVec.shape[0], :] 
    return toReturn


