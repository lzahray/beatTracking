import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.display


audioFile = "/n/sd1/music/RWC-MDB/P/wav/RM-P001.wav"
x, fs = librosa.load(audioFile,sr=None)
hopSize = 441
windowSize = 1024
mel = librosa.feature.melspectrogram(x,sr=fs, hop_length = hopSize, n_fft = windowSize, n_mels = 30,fmax=17000, fmin=30)
spec = librosa.core.stft(x, n_fft=windowSize, hop_length=441)

mel = mel.transpose()

#First order difference:
#This time we're gonna try using the median difference from the 2011 paper, see if it does better 
#add an extra time line so our first order difference is correct dims
melAugmented = np.insert(mel, 0, 0, axis=0)
#take the difference 
firstOrderDiff = np.diff(melAugmented,axis=0)
#we only care about positive diffs
firstOrderDiff[firstOrderDiff<0] = 0
plt.imshow(librosa.core.power_to_db(firstOrderDiff), interpolation="nearest", origin="upper", aspect="auto", cmap="Greys")
plt.figure()
melDb = librosa.core.power_to_db(mel)
melAugmentedDb = np.insert(melDb, 0, 0, axis=0)
firstOrderDiffDb = np.diff(melAugmentedDb,axis=0)
firstOrderDiffDb[firstOrderDiffDb<0] = 0
plt.imshow(firstOrderDiffDb, interpolation="nearest", origin="upper", aspect="auto", cmap="Greys")

plt.figure()
tempogram = librosa.feature.tempogram(x, sr=fs)
librosa.display.specshow(tempogram, sr=fs, hop_length=512, x_axis='time', y_axis='tempo')
plt.figure()
plt.imshow(tempogram, interpolation="nearest", origin="upper", aspect="auto", cmap="Greys")
#plt.imshow(spec, interpolation="nearest", origin="upper", aspect="auto", cmap="Greys")
plt.show()
#librosa.display.specshow(mel, y_axis='mel',x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.figure()
# librosa.display.specshow(spec, y_axis='hz', x_axis='time')
# plt.title("Spectrogram")
# plt.tight_layout()
# plt.show()