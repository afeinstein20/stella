import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from lightkurve.lightcurve import LightCurve as LC

import stella


__all__ = ['NeuralNetwork']

class NeuralNetwork(object):

    def __init__(self, training_dir=None, training_size=None,
                 epochs=None, layers=None, density=None, cadences=None):

        if training_dir is None:
            self.training_dir = os.path.join(stella.__path__[0], 'training_set')
        else:
            self.training_dir = training_dir

        if training_size is None:
            self.training_size = 300000
        else:
            self.training_size = training_size

        if epochs is None:
            self.epochs = 20
        else:
            self.epochs = epochs

        if layers is None:
            self.layers = 2
        else:
            self.layers = layers

        if cadences is None:
            self.cadences = 128
        else:
            self.cadences = cadences

        if density is None:
            self.density = np.array([32, 128, 2])
        else:
            self.density = density

        if int(len(self.density)-1) != self.layers:
            raise ValueError("Please put in as many densities as you want layers.")

        self.training_files()
        self.training_set()
        self.preparing_set()
        self.network_model()

    
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
        window_length : int, opt
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
        self.binned_labels = np.array(binned_label)
        self.binned_data_detrended = np.array(binned_data_detrended)



    def network_model(self, optimizer='adam', metrics=['accuracy'],
                      loss='sparse_categorical_crossentropy'):
        """
        Creates the neural network model based on layer and density specifications.
        
        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """

        keras_layers = []

        for i in range(len(self.density)):
            if self.density[i] == 2:
                kl = keras.layers.Dense(self.density[i], activation=tf.nn.softmax)
            else:
                kl = keras.layers.Dense(self.density[i], activation=tf.nn.relu)
            keras_layers.append(kl)
            

        model = keras.Sequential(keras_layers)
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        
        self.model = model
                    
