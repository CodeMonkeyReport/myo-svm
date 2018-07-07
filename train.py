from sklearn.neural_network import MLPClassifier
import numpy as np
import myo as libmyo
from listener import Listener
import fnmatch
import os
import pandas as pd
from gtts import gTTS
from io import BytesIO
from sklearn.model_selection import train_test_split
from playsound import playsound


letters = ['a', 'b', 'c', 'e', 'f', 'o']

X = []
Y = []

def feature_extraction(input_array):
    """This function takes an input array and returns back for each column
       the std-dev, variance, min, max, mean, median"""

    mins = input_array.min(axis=0)
    maxs = input_array.max(axis=0)
    std = input_array.std(axis=0)
    var = input_array.var(axis=0)
    mean = input_array.mean(axis=0)
    median = np.median(input_array, axis=0)
    return np.concatenate((mean, median, var, std, maxs, mins), axis=0)

for letter in letters:
    for file_name in fnmatch.filter(os.listdir('data'), "[0-9]-{0}*.csv".format(letter)):
        X.append(feature_extraction(np.loadtxt('data/'+file_name, delimiter=',')))
        Y.append(letter)

nn_learner = MLPClassifier(
                        hidden_layer_sizes=(96, 96, ), 
                        activation='logistic', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size='auto', 
                        learning_rate='constant', 
                        learning_rate_init=0.1, 
                        power_t=0.5, 
                        max_iter=200, 
                        shuffle=True, 
                        random_state=None, 
                        tol=0.0001, 
                        verbose=False, 
                        warm_start=False, 
                        momentum=0.9, 
                        nesterovs_momentum=True, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        beta_1=0.9, 
                        beta_2=0.999, 
                        epsilon=1e-08
)



nn_learner.fit(X, Y)

libmyo.init(r'myo-sdk-win-0.9.0\bin\myo64.dll')
hub = libmyo.Hub()
listener = Listener(250)

# Loop untill we close
while True:
    input("Press Enter to predict another letter")
    hub.run(listener, 1500)
    data = feature_extraction(np.array(listener.data))
    res = nn_learner.predict(data.reshape(1, -1))[0]
    print(res)
    textToVoice = gTTS(text=res,lang='en')
    textToVoice.save('sound.mp3')
    playsound('sound.mp3')
