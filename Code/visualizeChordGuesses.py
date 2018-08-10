import numpy as np
import matplotlib.pyplot as plt 

#first we do simple plot
song = 1
chordGuessesFolder = "../Results/Conv3DivC/ChordGuesses"
targetFolder = "../Features/mel128ChromaCQTWithTargets"
chord_guesses = np.load(chordGuessesFolder+"/chordGuess"+str(song)+".npy")
chord_targets = np.load(targetFolder+"/"+str(song)+"chordTargets.npy")
t_guess = np.arange(len(chord_guesses))

plt.plot(t_guess,chord_guesses)
plt.plot(t_guess, chord_targets )
plt.figure()
diff = chord_guesses - chord_targets
plt.plot(t_guess,diff)
plt.show()