import os 
import wotan
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from astropy.stats import sigma_clip
from lightkurve.lightcurve import LightCurve as LC

__all__ = ['NeuralNetwork']

class NeuralNetwork(object):
    """
    A class that creates a neural network and trains on 
    a data set. The data set can either be created using
    stella.SimulateLightCurves or from a directory of files.
    """

    def __init__(self, slc=None, training_dir=None):
        """
        Parameters
        ----------
        slc : stella.SimulatedLightCurve
             The simulated light curve data set to 
             train the neural network on.
        training_dir : path
             If the training set has been saved in
             .npy files, this will load in the set.

        Attributes
        ---------- 
        cadences : np.ndarray
        training_time : np.ndarray
        training_fluxes : np.ndarray
        training_labels : np.ndarray
        training_detrended : np.ndarray
        """
        if slc is not None:
            self.training_time   = slc.time
            self.training_fluxes = slc.fluxes
            self.training_detrended = slc.detrended
            self.training_labels = slc.labels
            self.cadences = slc.cadences

        else:
            if training_dir is None:
                raise ValueError("Please initialize with either Stella.Simulate() or with training_set directory.")
            files = os.listdir(training_dir)
            files = [os.path.join(training_dir, i) for i in files if i.endswith('.npy')]
            
            for i in tqdm(range(len(files))):
                d = np.load(files[i], allow_pickle=True)
                if i == 0:
                    training_time   = np.zeros((len(files), len(d[0])))
                    training_fluxes = np.zeros((len(files), len(d[0])))
                    training_labels = np.zeros((len(files), ))
                    training_detrended = np.zeros((len(files), len(d[0])))
                training_time[i] = d[0]
                training_fluxes[i] = d[1]
                training_detrended[i] = d[2]
                training_labels[i] = d[3]

            self.training_time = training_time
            self.training_fluxes = training_fluxes
            self.training_labels = training_labels
            self.training_detrended = training_detrended
            self.cadences = len(training_time[0])
                                               
        self.flux        = None
        self.predictions = None

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
            training_set = self.training_detrended
        else:
            training_set = self.training_fluxes

        self.model.fit(training_set,
                       self.training_labels,
                       epochs=epochs)


    def predict(self, time, flux, flux_err, detrending=True,
                injection=False, detrend_method='biweight', window_length=0.1,
                polyorder=2, edge_cutoff=0.5, break_tolerance=0.5, cval=5.0,
                sigma=6, niters=5):
        """
        Assigns a probability of being a flare to each point in the 
        input data set.

        Parameters
        ----------  
        time : np.ndarray
             A time array for the data you want to predict.
        flux : np.ndarray
             An array of the data you want to predict using the 
             neural network.
        flux_err : np.ndarray
             An array of errors on the flux.
        detrending : bool, optional
             Allows the user to specify if they want the light curve
             detrended before identifying flares. Default is True.
        injection : bool, optional
             If injection == True, return the predictions instead of 
             overwriting self.predictions. Default = False.
        detrend_method : str, optional
             The type of detrending to use on the light curve. Default
             is 'biweight'. Other options include: 'savgol'.
        window_length : str, optional
             The window length used for the robust estimator. Default = 0.1.
             For the biweight filter, window_length is given in unit days.
             For the savgol filter, window_length is given in unit cadences.
        polyorder : int, optional
             The order of the polynomial used to fit the samples. 
             Default = 2.
        sigma : float, optional
             Number of sigma above which to remove outliers during detrending.
             Default = 6.
        niters : int, optional
             Number of iterations to iteratively sigma clip and flatten. Default = 5.
        edge_cutoff : float, optional
             Trims data near the edges of the light curve. Default = 0.5. 
             Only available for the following detrend_methods: biweight.
        break_tolerance : float, optional
             Breaks the light curve into segments to help alleviate gap issues.
             Default = 0.5.
             Only available for the following detrend_methods: biweight.  
        cval : float, optional
             Tuning parameter for robust M-estimators. Defined as multiples in units
             of median absolute deviation from central location. Default = 5.0.
             Only available for the following detrend_methods: biweight.  

        Attributes
        ---------- 
        predictions : np.ndarray
             A 2D array of probabilities for each data set put in.
             Index 0 = Junk; Index 1 = Flare.
        """
        def detrend_poly(x, y):
            coefs = poly.polyfit(x, y, 2)
            return poly.polyval(x, coefs)

        cadences    = self.cadences
        predictions = []

        detrended_flux = np.copy(flux)

        if injection is False:
            self.detrend_method = detrend_method
            self.window_length  = window_length

        for i, lc in enumerate(flux):
            detrend = np.array([])
            if detrending is True:
                if detrend_method == 'savgol':
                    detrend = LC(time[i], lc).flatten(window_length=window_length,
                                                      polyorder=polyorder,
                                                      sigma=sigma,
                                                      niters=niters).flux
                if detrend_method == 'biweight':
                    detrend = wotan.flatten(time[i], lc, window_length=window_length,
                                            method=detrend_method, edge_cutoff=edge_cutoff,
                                            break_tolerance=break_tolerance, cval=cval)

                detrended_flux[i] = detrend
                lc = detrend

            # Centers each point in the input light curve and pads
            # with same number of cadences as used in the training set
            reshaped_data = np.zeros((len(lc),cadences))
            padding       = np.nanmedian(lc)
            std           = np.std(lc)
            cadence_pad   = int(cadences/2)

            for i in range(len(lc)):
                if i <= cadences/2:
                    fill_length   = int(cadence_pad-i)
                    padding_array = np.full( (fill_length,), padding ) + np.random.normal(0, std, fill_length)
                    reshaped_data[i] = np.append(padding_array, lc[0:int(i+cadence_pad)])

                elif i >= (len(lc)-cadence_pad):
                    loc = [int(i-cadence_pad), int(len(lc))]
                    fill_length   = int(np.abs(cadences - len(lc[loc[0]:loc[1]])))
                    padding_array = np.full( (fill_length,), padding ) + np.random.normal(0, std, fill_length)
                    reshaped_data[i] = np.append(lc[loc[0]:loc[1]], padding_array)

                else:
                    loc = [int(i-cadence_pad), int(i+cadence_pad)]
                    reshaped_data[i] = lc[loc[0]:loc[1]]
                    
            predictions.append(self.model.predict(reshaped_data))


        if injection is False:
            self.predictions    = np.array(predictions)
            self.time           = time
            self.flux           = flux
            self.flux_err       = flux_err
            self.detrended_flux = np.array(detrended_flux)

        else:
            return np.array(detrended_flux), np.array(predictions)


    def p_detrending(self, window_length=101):
        """
        Uses a Gaussian process to detrend the rotation period of the star.
        First it flattens using a Savitsky-Golay filter to complete a sigma clipping.
        Then it fits a GP to the sigma clipped flux.

        Parameters
        ----------  
        window_length : int, optional
             The window length applied to a Savitsky-Golay filter.
             Default = 101.
        """
        
        if self.flux is None:
            raise ValueError("Please input data into stella.NeuralNetwork.predict and set detrending=True.")

        return detrended_data
