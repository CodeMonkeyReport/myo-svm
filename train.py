from sklearn.svm import SVC
import numpy as np
import myo as libmyo
from listener import Listener
import os
import pandas as pd

letters = ['a', 'b', 'c']
record_count = 5

X = []
Y = []

def feature_extraction(input_array):
    """This function takes an input array and returns back for each column
       the std-dev, variance, abs-min, abs-max, mean, median"""

    mins = input_array.min(axis=0)
    maxs = input_array.max(axis=0)
    std = input_array.std(axis=0)
    var = input_array.var(axis=0)
    mean = input_array.mean(axis=0)
    median = np.median(input_array, axis=0)
    return np.concatenate((mean, median, var, std, maxs, mins), axis=0)

for letter in letters:
    for i in range(record_count):
        X.append(feature_extraction(np.loadtxt("{0}-{1}.csv".format(i, letter), delimiter=',')))
        Y.append(letter)

svm_learner = SVC(    # Create the SVM learner
                    C=1.7,                           # Penalty parameter for the error
                    kernel='linear',                    # Type of kernal
                    degree=32,                          # Degree of polynomial kernal (ignored if not poly kernal)
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



svm_learner.fit(X, Y)

libmyo.init(r'C:\Users\Michael\Documents\School\Myo\myo-sdk-win-0.9.0\bin\myo64.dll')
hub = libmyo.Hub()
listener = Listener(250)

# Loop untill we close
while True:
    input("Press Enter to predict another letter")
    hub.run(listener, 1500)
    data = feature_extraction(np.array(listener.data))
    print(svm_learner.predict(data.reshape(1, -1)))