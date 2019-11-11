import numpy as np
from tqdm import tqdm
from tensorflow import keras

__all__ = ['ConvNN']

class ConvNN(object):
    """
    Creates and trains the convolutional
    neural network.
    """

    def __init__(self, TD, layers=None, optimizer='adam',
                 loss='binary_crossentropy', metrics=['accuracy'],
                 epochs=15):
        """
        Creates and trains a Tensorflow keras model
        with either layers that have been passed in
        by the user or with default layers used in
        Feinstein et al. (2020; in prep.).

        Parameters
        ----------
        TD : stella.TrainingData object
        layers : np.array, optional
             An array of keras.layers for the ConvNN.
        optimizer : str, optional
             Optimizer used to compile keras model. Default is 'adam'.
        loss : str, optional
             Loss function used to compile keras model. Default is
             'binary_crossentropy'.
        metrics: np.array, optional
             Metrics used to train the keras model on. Default is 
             ['accuracy'].
        epochs : int, optional
             Number of epochs to train the keras model on. Default is 15.

        Attributes
        ----------
        layers : np.array
        optimizer : str
        loss : str
        metrics : np.array
        epochs : int
        training_matrix : stella.TrainingData.training_matrix
        labels : stella.TrainingData.labels
        """
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.training_matrix = TD.training_matrix
        self.labels = TD.labels

        self.create_model()
        self.train_model()

        
    def create_model(self):
        """
        Creates the Tensorflow keras model with appropriate layers.
        
        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """
        # Makes sure the backend is clean
        keras.backend.clear_session()

        model = keras.models.Sequential()

        if self.layers is None:
            model.add(keras.layers.Conv1D(32, 3, activation='relu'))
            model.add(keras.layers.MaxPooling1D((2)))
            model.add(keras.layers.Conv1D(64, 3, activation='relu'))

            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(2, activation='sigmoid'))
            
        else:
            for l in self.layers:
                model.add(l)
                
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        self.model = model


    def train_model(self):
        """
        Trains the model using the training set from stella.TrainingData.
        """

        self.model.fit(self.training_matrix,
                       self.labels, epochs=self.epochs)
