import numpy as np
import matplotlib.pyplot as plt 


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.5f' % height,
                ha='center', va='bottom')


# # losses = []
# # for i in range(5):
# #     myFile = np.load("lossesTest"+str(i)+".npy")
# #     losses.append(min(myFile))
# losses = np.array([.8830290607, 0.8409609499, 0.7524548206, 0.8353407576,0.8556688712])
# print("average f is ", np.mean(losses))
# fig, ax = plt.subplots()
# rects1 = ax.bar(np.arange(5), losses, .35)

# plt.xlabel("Partition Set")
# plt.ylabel("F-Measure")
# plt.xticks(np.arange(5))
# plt.yticks(np.arange(20)/20)
# plt.ylim(0.6,1)
# autolabel(rects1)
# #plt.bar(np.arange(5),losses)

# # Make some labels.
# #labels = [str(i) for i in losses]

# plt.show()
folder = "../beatsChords1/"
lossesTraining = np.load(folder+"lossesTrainingK0.npy")
lossesTest = np.load(folder+"lossesTestK0.npy")
plt.xlabel("Epoch")
plt.ylabel("Loss (Average Chord and Beat)")
plt.plot(np.arange(len(lossesTraining)), lossesTraining)
plt.plot(np.arange(len(lossesTraining)), lossesTest)
plt.show()
