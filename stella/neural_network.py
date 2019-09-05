import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from lightkurve.lightcurve import LightCurve as LC

import stella


__all__ = ['NeuralNetwork']

class NeuralNetwork(object):

    def __init__(self, training_dir=None, training_size=3000, cadences=128):

        if training_dir is None:
            self.training_dir = os.path.join(stella.__path__[0], 'training_set')
        else:
            self.training_dir = training_dir

        self.training_size = training_size
        self.cadences = cadences

        self.training_files()
        self.training_set()
        self.preparing_set()

    
    def training_files(self):
        """
        Grabs grabs the first n number of files from the training set, where n
            is the initialized training_size.

        Attributes
        ----------
        training_files : np.ndarray
        """
        files = os.listdir(self.training_dir)
        files = np.sort([os.path.join(self.training_dir, i) for i in files])
        self.training_files = files[0:int(self.training_size)]


    def training_set(self, window_length=21):
        """
        Appends a list of all the cadences from the training files.

        Parameters
        ----------
        window_length : int, optional
             Allows the user to specify the window_length used to
             detrend the light curve. Uses lightkurve.lightcurve.LightCurve.flatten.

        Attributes
        ----------
        training_data : np.ndarray
             A list of the flux values from all training files.
        training_labels : np.ndarray
             A list of the labels from all training files.
        training_data_detrended : np.ndarray
             A list of the detrended values from all training files.
        """

        # Finds how many cadences are in each simulated data file
        test = np.load(self.training_files[0], allow_pickle=True)
        cad_per_lc = int(len(test[0]))

        training_data   = np.zeros(cad_per_lc*len(self.training_files))
        training_labels = np.zeros(cad_per_lc*len(self.training_files))
        training_data_detrended = np.zeros(cad_per_lc*len(self.training_files))

        loc = 0

        for fn in self.training_files:
            d = np.load(fn, allow_pickle=True)
            detrend = LC(d[0], d[1]).flatten(window_length=window_length)

            # Loads in data from each simulated data file
            for i in range(len(d[0])):
                training_data[loc]  = d[1][i]
                training_labels[loc] = d[3][i]
                training_data_detrended[loc] = detrend.flux[i]
                loc += 1

        self.training_data   = training_data
        self.training_labels = training_labels
        self.training_data_detrended = training_data_detrended


    def preparing_set(self):
        """
        Chunks the data into n data points, where n is the number of cadences.

        Attributes
        ----------
        binned_data : np.ndarray
        binned_labels : np.ndarray
        binned_data_detrended : np.ndarray
        """
        
        def ranges(nums):
            """ Finds where the flares are originally labeled in the simulted data. """
            nums  = sorted(set(nums))
            gaps  = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            return np.array(list(zip(edges, edges)))


        def binning_to_cadences(data, start, stop):
            """ Bins the non-flare data to the same length. """
            j = 0
            while len(data) % self.cadences != 0:
                data = self.training_data[start:stop-j]
                j += 1
            return np.reshape(data, (int(len(data)/self.cadences), self.cadences))

        # Uses the simulated flare labels
        flare_inds = np.where(self.training_labels == 1)[0]
        flares     = ranges(flare_inds)

        print("** FLARES **")
        print(len(flares))

        # Centers the flare in each chunk of data to be fed into the network
        for i in range(len(flares)):
            flare_peak = flares[i][0]
            padding    = int(self.cadences/2)
            region     = [flares[i][0]-padding, flares[i][0]+padding]
            flares[i]  = np.array(region)

            
        binned_data   = []
        binned_labels = []
        binned_data_detrended = []

        for i in range(len(flares)):
            if i == 0:
                junk = binning_to_cadences(self.training_data[0:flares[i][0]], 0, flares[i][0])
                flat_junk = binning_to_cadences(self.training_data_detrended[0:flares[i][0]], 0, flares[i][0])
            else:
                junk = binning_to_cadences(self.training_data[flares[i-1][1]:flares[i][0]],
                                           flares[i-1][1], flares[i][0])
                flat_junk = binning_to_cadences(self.training_data_detrended[flares[i-1][1]:flares[i][0]],
                                                flares[i-1][1], flares[i][0])

            # Appends the binned flux
            for j in range(junk.shape[0]):
                binned_data.append(junk[0])
                binned_labels.append(0)
                binned_data_detrended.append(flat_junk[0])
                
            # Appends the flare
            f = self.training_data[flares[i][0]:flares[i][1]]
            binned_data.append(f)
            binned_labels.append(1)
            f = self.training_data_detrended[flares[i][0]:flares[i][1]]
            binned_data_detrended.append(f)

        self.binned_data   = np.array(binned_data)
        self.binned_labels = np.array(binned_labels)
        self.binned_data_detrended = np.array(binned_data_detrended)



    def network_model(self, layers=2, density=[32,128],
                      optimizer='adam', metrics=['accuracy'],
                      loss='sparse_categorical_crossentropy'):
        """
        Creates the neural network model based on layer and density specifications.

        Parameters
        ----------
        layers : int, optional
             Specifies how many layers in the neural network the user wants. Default = 2.
        density : list or np.ndarray, optional
             Specifices the density in each layer in the network. Default = [32, 128].
        optimizer : str, optional
             The type of optimizer. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
             for options. Default = 'adam'.
        metrics : list, optional
             The list of metrics to be evaluated by the model during training and testing.
             Default = ['accuracy'].
        loss : str, optional
             The objective function. SEe https://www.tensorflow.org/api_docs/python/tf/losses
             for options. Default = 'sparse_categotical_crossentropy'

        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """

        keras_layers = []

        for i in range(len(density)):
            keras_layers.append(keras.layers.Dense(density[i], activation=tf.nn.relu))
            
        keras_layers.append(keras.layers.Dense(2, activation=tf.nn.softmax))

        model = keras.Sequential(keras_layers)
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        
        self.model = model
                    

    def train(self, epochs=15, percentage=None, detrended=False):
        """
        Trains the neural network.

        Parameters
        ---------- 
        percentage : float, optional
             Gives the user the option to choose a certain number
             of binned data sets to use from the full set. Default is 0.66.
        detrended  : bool, optional
             Gives the user the option to train on detrended
             simulated light curves.
        """

        if percentage is None:
            bins = int( 0.66 * self.binned_data.shape[0])
        else:
            bins = int( percentage * self.binned_data.shape[0])

        if detrended is True:
            self.model.fit(self.binned_data_detrended[0:bins],
                           self.binned_labels[0:bins],
                           epochs=epochs)
        else:
            self.model.fit(self.binned_data[0:bins],
                           self.binned_labels[0:bins],
                           epochs=epochs)


    def predict(self, input_data):
        """
        Uses the trained neural network to predict data.

        Parameters
        ---------- 
        input_data : np.ndarray            
        
        Attributes
        ---------- 
        predictions : np.ndarray
             A 2D array of probabilities for each data set put in.
             Index 0 = Junk; Index 1 = Flare
        """
        predictions = self.model.predict(input_data)
        self.predictions = predictions
