import os 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from lightkurve.lightcurve import LightCurve as LC

__all__ = ['NeuralNetwork']

class NeuralNetwork(object):
    """
    A class that creates a neural network and trains on 
    a data set. The data set can either be created using
    stella.SimulateLightCurves or from a directory of files.
    """

    def __init__(self, slc):
        """
        Parameters
        ----------
        slc : stella.SimulatedLightCurve
             The simulated light curve data set to 
             train the neural network on.

        Attributes
        ---------- 
        slc : stella.SimulatedLightCurve
        """
        self.slc = slc
        

    def network_model(self, layers=2, density=[32,128],optimizer='adam',
                      metrics=['accuracy'], loss='sparse_categorical_crossentropy'):
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
             The objective function. See https://www.tensorflow.org/api_docs/python/tf/losses
             for options. Default = 'sparse_categotical_crossentropy'

        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """
        
        keras_layers = []

        for i in range(len(density)):
            keras_layers.append(keras.layers.Dense(density[i], activation=tf.nn.relu))

        # Appends a layer for 2 label classification
        keras_layers.append(keras.layers.Dense(2, activation=tf.nn.softmax))

        model = keras.Sequential(keras_layers)
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        self.model = model
        

    def train_data(self, epochs=15, detrended=False):
        """
        Trains the neural network on simulated data or inputted data files.

        Parameters
        ---------- 
        epochs : int, optional
             The number of epochs to train the data set on.
             Default = 15.
        detrended : bool, optional
             Gives the user the option to train on detrended simulated data.
             Default = False.
        """

        if detrended is True:
            training_set = self.slc.flare_fluxes_detrended
        else:
            training_set = self.slc.flare_fluxes

        self.model.fit(training_set,
                       self.slc.labels,
                       epochs=epochs)


    def predict(self, input_data):
        """
        Ues the trained neural network to predict data.

        Parameters
        ---------- 
        input_data : np.ndarray
             The data you want to predict using the neural network.

        Attributes
        ---------- 
        predictions : np.ndarray
             A 2D array of probabilities for each data set put in.
             Index 0 = Junk; Index 1 = Flare.
        """
        input_data = np.array(input_data)
        data_shape = input_data.shape

        predictions = self.model.predict(input_data)
        self.predictions=predictions


    def data_slider(self):
        """
        Assigns a probability to each data point in the set.
        """

            
