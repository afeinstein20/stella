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
        image_fmt : stella.TrainingData.image_fmt
        """
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.training_matrix = TD.training_matrix
        self.labels = TD.labels
        self.image_fmt = TD.image_fmt

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


    def predict(self, times, fluxes):
        """
        Takes in arrays of time and flux and predicts where the flares
        are based on the keras model created and trained. This function
        also reshapes the data into a way for the model to understand.

        Parameters
        ----------
        times : np.ndarray
             Array of times to predict flares in.
        fluxes : np.ndarray
             Array of fluxes to predict flares in.

        Attributes
        ----------
        time_data : time
        flux_data : flux
        predictions : np.nparray
             Predicted labels for the light curves.
        """
        predictions = []

        bins = self.image_fmt
        cadences = self.image_fmt[0]

        for j in tqdm(range(len(times))):
            time = times[j]
            lc   = fluxes[j]
    
            reshaped_data = np.zeros((len(lc), bins[1], bins[0]))
            padding       = np.nanmedian(lc)
            std           = np.std(lc)
            cadence_pad   = int(cadences/2)
    
            for i in range(len(lc)):
                if i <= cadences/2:
                    fill_length   = int(cadence_pad-i)
                    padding_array = np.zeros( (fill_length,))
                    f = np.append(padding_array, lc[0:int(i+cadence_pad)])
            
                    tsteps = np.std(np.diff(time)) * np.arange(0,fill_length,1)
                    tstep_padding = np.flip(time[i] - tsteps)
                    t = np.append(tstep_padding, time[0:int(i+cadence_pad)])
                    
                elif i >= (len(lc)-cadence_pad):
                    loc = [int(i-cadence_pad), int(len(lc))]
                    fill_length   = int(np.abs(cadences - len(lc[loc[0]:loc[1]])))
                    padding_array = np.zeros( (fill_length,))
                    f = np.append(lc[loc[0]:loc[1]], padding_array)
                    
                    tsteps = np.std(np.diff(time)) * np.arange(0,fill_length,1)
                    tstep_padding = time[i] - tsteps
                    t = np.append(time[loc[0]:loc[1]], tstep_padding)
                    
                else:
                    loc = [int(i-cadence_pad), int(i+cadence_pad)]
                    f = lc[loc[0]:loc[1]]
                    t = np.append(time[loc[0]:loc[1]], tstep_padding)
            
                data = np.histogram2d(t, (f - np.nanmax(f)) / np.nanstd(f), bins=bins)
                data = np.rot90(data[0])
                
                reshaped_data[i] = data

            preds = self.model.predict(reshaped_data)
            predictions.append(preds)

        self.time_data   = times
        self.flux_data   = fluxes
        self.predictions = np.array(predictions)
                                  
