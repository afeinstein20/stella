import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

__all__ = ['ConvNN']

class ConvNN(object):
    """
    Creates and trains the convolutional
    neural network.
    """

    def __init__(self, ts, training=0.80, validation=0.90,
                 layers=None, optimizer='adam',
                 loss='binary_crossentropy', metrics=['accuracy']):
        """
        Creates and trains a Tensorflow keras model
        with either layers that have been passed in
        by the user or with default layers used in
        Feinstein et al. (2020; in prep.).

        Parameters
        ----------
        ts : stella.TrainingSet object
        training : float, optional
             Assigns the percentage of training set data for training.
             Default is 80%.
        validation : float, optional
             Assigns the percentage of training set data for validation.
             Default is 10%.
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
        training_matrix : stella.TrainingSet.training_matrix
        labels : stella.TrainingSet.labels
        image_fmt : stella.TrainingSet.cadences
        """
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.training_matrix = ts.training_matrix
        self.labels = ts.labels
        self.cadences = ts.cadences

        self.train_cutoff = int(training * len(self.labels))
        self.val_cutoff   = int(validation * len(self.labels))

        self.create_model()


    def create_model(self):
        """
        Creates the Tensorflow keras model with appropriate layers.
        
        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """
        # INITIALIZE CLEAN MODEL
        keras.backend.clear_session()

        model = keras.models.Sequential()

        # DEFAULT NETWORK MODEL FROM FEINSTEIN ET AL. (in prep)
        if self.layers is None:
            filter1 = 16
            filter2 = 64
            dense   = 32
            dropout = 0.1

            # CONVOLUTIONAL LAYERS
            model.add(tf.keras.layers.Conv1D(filters=filter1, kernel_size=3, 
                                             activation='relu', padding='same', 
                                             input_shape=(self.cadences, 1)))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Conv1D(filters=filter2, kernel_size=3, 
                                             activation='relu', padding='same'))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            model.add(tf.keras.layers.Dropout(dropout))
            
            # DENSE LAYERS AND SOFTMAX OUTPUT
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(dense, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            
        else:
            for l in self.layers:
                model.add(l)
                
        # COMPILE MODEL AND SET OPTIMIZER, LOSS, METRICS
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        self.model = model
        
        # PRINTS THE MODEL SUMMARY FOR THE USER
        model.summary()


    def train_model(self, epochs=500, batch_size=64):
        """
        Trains the model using the training set from stella.TrainingData.

        Parameters
        ---------- 
        epochs : int, optional 
             The number of epochs to train for.
             Default is 500.
        batch_size : int, optional
             The batch size fro training.
             Default is 64.

        Attributes
        ---------- 
        history : tensorflow.python.keras.callbacks.History
        test_data : np.array
             The remaining data in training_matrix that is used
             for testing.
        test_labels : np.array
             The labels for the testing data set.
        """
        x_train = self.training_matrix[0:self.train_cutoff]
        y_train = self.labels[0:self.train_cutoff]

        x_val = self.training_matrix[self.train_cutoff:self.val_cutoff]
        y_val = self.labels[self.train_cutoff:self.val_cutoff]

        x_test = self.training_matrix[self.val_cutoff:] 
        y_test = self.labels[self.val_cutoff:]

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val   = x_val.reshape(x_val.shape[0], x_train.shape[1], 1)
        x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        self.test_data = x_test
        self.test_labels = y_test

        history = self.model.fit(x_train, y_train, epochs=epochs, 
                                 batch_size=batch_size, shuffle=True, 
                                 validation_data=(x_val, y_val))
        self.history = history

        
    def loss_acc(self):
        """
        Plots the loss and accuracy for the training and validation 
        data sets. Keyword accuracy will change based on what metrics
        were used.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        epochs = np.arange(0,len(self.history.history['loss']),1)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,4))

        ax1.plot(epochs, self.history.history['loss'], c='k',
                 linewidth=2, label='Training')
        ax1.plot(epochs, self.history.history['val_loss'], c='darkorange',
                 linewidth=2, label='Validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        ax2.plot(epochs, self.history.history['acc'], c='k',
                 linewidth=2)
        ax2.plot(epochs, self.history.history['val_acc'], c='darkorange',
                 linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')

        return fig


    def predict(self, ids, times, fluxes, flux_errs, injected=False):
        """
        Takes in arrays of time and flux and predicts where the flares 
        are based on the keras model created and trained.

        Parameters
        ----------
        ids : np.array
             Array of light curve identifiers (e.g. list of TICs).
        times : np.ndarray
             Array of times to predict flares in.
        fluxes : np.ndarray
             Array of fluxes to predict flares in.
        flux_errs : np.ndarray
             Array of flux errors for predicted flares.
        injected : bool, optional
             Returns predictions instead of setting attribute. Used
             for injection-recovery. Default is False.
             
        Attributes
        ----------
        predict_ids : np.array
             The input target IDs.
        predict_times : np.ndarray
             The input times array.
        predict_fluxes : np.ndarray
             The input fluxes array.
        predict_errs : np.ndarray
             The input flux errors array.
        predictions : np.ndarray
             An array of predictions from the model.
        """
        predictions = []

        cadences = self.cadences

        # FOR THE LOADING BAR
        if injected is True:
            disable = True
        else:
            disable = False

        for j in tqdm(range(len(times)), disable=disable):
            time = times[j]
            lc   = fluxes[j]
        
            reshaped_data = np.zeros((len(lc), cadences))
        
            padding       = np.nanmedian(lc)
            std           = np.std(lc)/3.
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

                reshaped_data[i] = f
                
            reshaped_data = reshaped_data.reshape(reshaped_data.shape[0], 
                                                  reshaped_data.shape[1], 1)
            preds = self.model.predict(reshaped_data)
            predictions.append(np.reshape(preds, len(preds)))

        if injected is False:
            self.predict_ids = np.array(ids)
            self.predict_times = np.array(times)
            self.predict_fluxes = np.array(fluxes)
            self.predict_errs = np.array(flux_errs)
            self.predictions = np.array(predictions)
        else:
            return np.array(predictions)
