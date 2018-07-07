from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
import myo as libmyo
from listener import Listener
import fnmatch
import os
import pandas as pd
from gtts import gTTS
from io import BytesIO
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

svm_learner = SVC(    # Create the SVM learner
                    C=1.0,                           # Penalty parameter for the error
                    kernel='poly',                    # Type of kernal
                    degree=8,                          # Degree of polynomial kernal (ignored if not poly kernal)
                    gamma='auto',                       # Kernal coefficient for the rbf, poly, and sigmoid kernals, auto = 1/n features used
                    coef0=1.0,                          # Independent term (Used in poly and sigmoid)
                    shrinking=False,                    # Use shrinking huristic or not
                    probability=True,                   # Probability estimates
                    tol=0.00001,                        # Tolerance for stopping criterion
                    cache_size=1000,                    # size in MB
                    class_weight='balanced',            # supply class weights in case of unbalanced data, 'balanced' option uses input sizes
                    verbose=False,                      # Output all steps
                    max_iter=-1,                        # Limit number of itterations, -1 for inf
                    decision_function_shape='ovo',      # 
                    random_state=None                   # Seed for the random state 'None' for always random
)


# Split into training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                test_size=0.1, train_size=0.90, random_state=None)

svm_learner.fit(X_train, Y_train)
print("Score: ", svm_learner.score(X_test, Y_test))

Y_pred_test = svm_learner.predict(X_test)

print("Test set results:")
print("------------------------------------------------")
print(metrics.confusion_matrix(Y_test, Y_pred_test))
print('accuracy:    ', metrics.accuracy_score(Y_test, Y_pred_test))
print('f_1:         ', metrics.f1_score(Y_test, Y_pred_test, average=None))
print('recall:      ', metrics.recall_score(Y_test, Y_pred_test, average=None))
print('precision:   ', metrics.precision_score(Y_test, Y_pred_test, average=None))
print('kappa:       ', metrics.cohen_kappa_score(Y_test, Y_pred_test))

# libmyo.init(r'myo-sdk-win-0.9.0\bin\myo64.dll')
# hub = libmyo.Hub()
# listener = Listener(250)

# # Loop untill we close
# while True:
#     input("Press Enter to predict another letter")
#     hub.run(listener, 1500)
#     data = feature_extraction(np.array(listener.data))
#     res = svm_learner.predict(data.reshape(1, -1))[0]
#     proba = svm_learner.predict_proba(data.reshape(1, -1))
#     print("Proba: ", proba)
#     print(res)
#     textToVoice = gTTS(text=res,lang='en')
#     textToVoice.save('sound.mp3')
#     playsound('sound.mp3')
