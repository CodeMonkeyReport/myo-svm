from listener import Listener
import myo as libmyo
import numpy as np
from random import randint


libmyo.init(r'C:\Users\Michael\Documents\School\Myo\myo-sdk-win-0.9.0\bin\myo64.dll')
hub = libmyo.Hub()
listener = Listener(250)

letters = ['a', 'b', 'c', 'e', 'f', 'o']
record_count = 5

# Get data for each letter
for letter in letters:
    for i in range(record_count):
        print("recording instance {0} of letter {1}".format(i, letter))
        input("Press Enter to begin recording. . . ")
        hub.run(listener, 1500)
        data = np.array(listener.data)
        np.savetxt("data/{0}-{1}-{2}.csv".format(i, letter, randint(10000, 99999)), data, delimiter=',')

