from sklearn.svm import SVC
from sklearn import metrics
import myo as libmyo
import numpy as np
import pandas as pd
import threading
import collections

class Listener(libmyo.DeviceListener):
    """ Implementation of myo.DeviceListener
    This class stores the past *que_size* data points """
    def __init__(self, que_size=25):
        self._lock = threading.Lock()
        self.data_queue = collections.deque(maxlen=que_size)
        self.current_orientation = [0, 0, 0, 0]

    def on_connected(self, event):
        event.device.stream_emg(libmyo.StreamEmg.enabled)

    @property
    def data(self):
        with self._lock:
            return list(self.data_queue)
        
    def on_emg(self, event):
        with self._lock:
            self.data_queue.append(event.emg + self.current_orientation)

    def on_orientation(self, event):
        with self._lock:
            # self.orientation_data_queue.extend(event.orientation)
            self.current_orientation = list(event.orientation)

if __name__ == "__main__":

    libmyo.init(r'C:\Users\Michael\Documents\School\Myo\myo-sdk-win-0.9.0\bin\myo64.dll')

    hub = libmyo.Hub()
    listener = Listener(250)
    hub.run(listener, 1500)

    data = np.array(listener.data)
    np.savetxt("output.csv", data, delimiter=',')