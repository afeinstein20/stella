import os, glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
from astropy.table import Table, Column

__all__ = ['ConvNN']

class ConvNN(object):
    """
    Creates and trains the convolutional
    neural network.
    """

    def __init__(self, output_dir, ds=None,
                 layers=None, optimizer='adam',
                 loss='binary_crossentropy', 
                 metrics=None):
        """
        Creates and trains a Tensorflow keras model
        with either layers that have been passed in
        by the user or with default layers used in
        Feinstein et al. (2020; in prep.).

        Parameters
        ----------
        ds : stella.DataSet object
        output_dir : str
             Path to a given output directory for files.
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
             Metrics used to train the keras model on. If None, metrics are
             [accuracy, precision, recall].
        epochs : int, optional
             Number of epochs to train the keras model on. Default is 15.
        seed : int, optional
             Sets random seed for reproducable results. Default is 2.
        output_dir : path, optional
             The path to save models/histories/predictions to. Default is
             to create a hidden ~/.stella directory.

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
        self.ds = ds
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        if ds is not None:
            self.training_matrix = np.copy(ds.training_matrix)
            self.labels = np.copy(ds.labels)
            self.cadences = np.copy(ds.cadences)

            self.frac_balance = ds.frac_balance + 0.0

            self.tpeaks = ds.training_peaks
            self.training_ids = ds.training_ids

        else:
            print("WARNING: No stella.DataSet object passed in.")
            print("Can only use stella.ConvNN.predict().")

        self.prec_recall_curve = None
        self.history = None
        self.history_table = None

        self.output_dir = output_dir


    def create_model(self, seed):
        """
        Creates the Tensorflow keras model with appropriate layers.
        
        Attributes
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
        """
        # SETS RANDOM SEED FOR REPRODUCABLE RESULTS
        np.random.seed(seed)
        tf.random.set_seed(seed)

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
            model.add(tf.keras.layers.Conv1D(filters=filter1, kernel_size=7, 
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
        if self.metrics is None:
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=['accuracy', tf.keras.metrics.Precision(), 
                                   tf.keras.metrics.Recall()])
        else:
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

        self.model = model

        # PRINTS MODEL SUMMARY
        model.summary()


    def load_model(self, modelname, mode='validation'):
        """
        Loads an already created model. 

        Parameters
        ----------
        modelname : str
        mode : str, optional
        """
        model = keras.models.load_model(modelname)
        self.model = model
        
        if mode == 'test':
            pred = model.predict(self.ds.test_data)
        elif mode == 'validation':
            pred = model.predict(self.ds.val_data)
        pred = np.reshape(pred, len(pred))
        
        ## Calculate metrics from here
        return 
        

    def train_models(self, seeds=[2], epochs=350, batch_size=64, shuffle=False,
                     pred_test=False, save=False):
        """
        Runs n number of models with given initial random seeds of
        length n. Also saves each model run to a hidden ~/.stella 
        directory. 

        Parameters
        ----------
        seeds : np.array
             Array of random seed starters of length n, where
             n is the number of models you want to run.
        epochs : int, optional
             Number of epochs to train for. Default is 350.
        batch_size : int, optional
             Setting the batch size for the training. Default
             is 64.
        shuffle : bool, optional
             Allows for shuffling of the training set when fitting
             the model. Default is False.
        pred_test : bool, optional
             Allows for predictions on the test set. DO NOT SET TO
             TRUE UNTIL YOU'VE DECIDED ON YOUR FINAL MODEL. Default
             is False.
        save : bool, optional
             Saves the predictions and histories of from each model
             in an ascii table to the specified output directory.
             Default is False.

        Attributes
        ----------
        history_table : Astropy.table.Table
             Saves the metric values for each model run.
        val_pred_table : Astropy.table.Table
             Predictions on the validation set from each run.
        test_pred_table : Astropy.table.Table
             Predictions on the test set from each run. Must set
             pred_test = True, or else it is an empty table.
        """

        if type(seeds) == int or type(seeds) == float or type(seeds) == np.int64: 
            seeds = np.array([seeds])

        self.epochs = epochs

        # CREATES TABLES FOR SAVING DATA
        table = Table()
        val_table  = Table([self.ds.val_ids, self.ds.val_labels, self.ds.val_tpeaks],
                           names=['tic', 'gt', 'tpeak'])
        test_table = Table([self.ds.test_ids, self.ds.test_labels, self.ds.test_tpeaks],
                           names=['tic', 'gt', 'tpeak'])
        
        
        for seed in seeds:
            
            fmt_tail = '_s{0:04d}_i{1:04d}_b{2}'.format(int(seed), int(epochs), self.frac_balance)
            model_fmt = 'ensemble' + fmt_tail + '.h5'

            keras.backend.clear_session()
            
            # CREATES MODEL BASED ON GIVEN RANDOM SEED
            self.create_model(seed)
            self.history = self.model.fit(self.ds.train_data, self.ds.train_labels, 
                                          epochs=epochs,
                                          batch_size=batch_size, shuffle=shuffle,
                                          validation_data=(self.ds.val_data, self.ds.val_labels))

            col_names = list(self.history.history.keys())
            for cn in col_names:
                col = Column(self.history.history[cn], name=cn+'_s{0:04d}'.format(int(seed)))
                table.add_column(col)

            # SAVES THE MODEL TO OUTPUT DIRECTORY
            self.model.save(os.path.join(self.output_dir, model_fmt))

            # GETS PREDICTIONS FOR EACH VALIDATION SET LIGHT CURVE
            val_preds = self.model.predict(self.ds.val_data)
            val_preds = np.reshape(val_preds, len(val_preds))
            val_table.add_column(Column(val_preds, name='pred_s{0:04d}'.format(int(seed))))
            

            # GETS PREDICTIONS FOR EACH TEST SET LIGHT CURVE IF PRED_TEST IS TRUE
            if pred_test is True:
                test_preds = self.model.predict(self.ds.test_data)
                test_preds = np.reshape(test_preds, len(test_preds))
                test_table.add_column(Column(test_preds, name='pred_s{0:04d}'.format(int(seed))))
                
        # SETS TABLE ATTRIBUTES
        self.history_table = table
        self.val_pred_table = val_table
        self.test_pred_table = test_table

        # SAVES TABLE IS SAVE IS TRUE
        if save is True:
            fmt_table = '_i{0:04d}_b{1}.txt'.format(int(epochs), self.frac_balance)
            hist_fmt = 'ensemble_histories' + fmt_table
            pred_fmt = 'ensemble_predval' + fmt_table

            table.write(os.path.join(self.output_dir, hist_fmt), format='ascii')
            val_table.write(os.path.join(self.output_dir, pred_fmt), format='ascii',
                            fast_writer=False)

            if pred_test is True:
                test_fmt = 'ensemble_predtest' + fmt_table
                test_table.write(os.path.join(self.output_dir, test_fmt), format='ascii',
                                 fast_writer=False)


    def cross_validation(self, seed=2, epochs=350, batch_size=64,
                         n_splits=5, shuffle=False, pred_test=False, save=False):
        """
        Performs cross validation for a given number of K-folds.
        Reassigns the training and validation sets for each fold.

        Parameters
        ----------
        seed : int, optional
             Sets random seed for creating CNN model. Default is 2.
        epochs : int, optional
             Number of epochs to run each folded model on. Default is 350.
        batch_size : int, optional
             The batch size for training. Default is 64.
        n_splits : int, optional
             Number of folds to perform. Default is 5.
        shuffle : bool, optional
             Allows for shuffling in scikitlearn.model_slection.KFold.
             Default is False.
        pred_test : bool, optional
             Allows for predicting on the test set. DO NOT SET TO TRUE UNTIL
             YOU ARE HAPPY WITH YOUR FINAL MODEL. Default is False.
        save : bool, optional
             Allows the user to save the kfolds table of predictions.
             Defaul it False.

        Attributes
        ----------
        crossval_predval : astropy.table.Table
             Table of predictions on the validation set from each fold.
        crossval_predtest : astropy.table.Table
             Table of predictions on the test set from each fold. ONLY 
             EXISTS IF PRED_TEST IS TRUE.
        crossval_histories : astropy.table.Table
             Table of history values from the model run on each fold.
        """

        from sklearn.model_selection import KFold
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score

        num_flares = len(self.labels)
        trainval_cutoff = int(0.90 * num_flares)

        tab = Table()
        predtab = Table()

        x_trainval = self.training_matrix[0:trainval_cutoff]
        y_trainval = self.labels[0:trainval_cutoff]
        p_trainval = self.tpeaks[0:trainval_cutoff]
        t_trainval = self.training_ids[0:trainval_cutoff]

        kf = KFold(n_splits=n_splits, shuffle=shuffle)

        if pred_test is True:
            pred_test_table = Table()

        i = 0
        for ti, vi in kf.split(y_trainval):
            # CREATES TRAINING AND VALIDATION SETS
            x_train   = x_trainval[ti]
            y_train = y_trainval[ti]
            x_val   = x_trainval[vi]
            y_val = y_trainval[vi]

            p_val = p_trainval[vi]
            t_val = t_trainval[vi]
            
            # REFORMAT TO ADD ADDITIONAL CHANNEL TO DATA
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            
            # CREATES MODEL AND RUNS ON REFOLDED TRAINING AND VALIDATION SETS
            self.create_model(seed)
            history = self.model.fit(x_train, y_train,
                                     epochs=epochs,
                                     batch_size=batch_size, shuffle=shuffle,
                                     validation_data=(x_val, y_val))

            # SAVES THE MODEL BY DEFAULT
            self.model.save(os.path.join(self.output_dir, 'crossval_s{0:04d}_i{1:04d}_b{2}_f{3:04d}.h5'.format(int(seed),
                                                                                                               int(epochs),
                                                                                                               self.frac_balance,
                                                                                                               i)))
            

            # CALCULATE METRICS FOR VALIDATION SET
            pred_val = self.model.predict(x_val)
            pred_val = np.reshape(pred_val, len(pred_val))

            # SAVES PREDS FOR VALIDATION SET
            tab_names = ['id', 'gt', 'peak', 'pred']
            data = [t_val, y_val, p_val, pred_val]
            for j, tn in enumerate(tab_names):
                col = Column(data[j], name=tn+'_f{0:03d}'.format(i))
                predtab.add_column(col)

            # PREDICTS ON TEST SET IF PRED_TEST IS TRUE
            if pred_test is True:
                preds = self.model.predict(self.ds.test_data)
                preds = np.reshape(preds, len(preds))
                data = [self.ds.test_ids, self.ds.test_labels, self.ds.test_tpeaks, 
                        np.reshape(preds, len(preds))]
                for j, tn in enumerate(tab_names):
                    col = Column(data[j], name=tn+'_f{0:03d}'.format(i))
                    pred_test_table.add_column(col)
                self.crossval_predtest = pred_test_table

            precision, recall, _ = precision_recall_curve(y_val, pred_val)
            ap_final = average_precision_score(y_val, pred_val, average=None)

            # SAVES HISTORIES TO A TABLE
            col_names = list(history.history.keys())
            for cn in col_names:
                col = Column(history.history[cn], name=cn+'_f{0:03d}'.format(i))
                tab.add_column(col)

            # KEEPS TRACK OF WHICH FOLD
            i += 1

        # SETS TABLES AS ATTRIBUTES
        self.crossval_predval = predtab
        self.crossval_histories = tab

        # IF SAVE IS TRUE, SAVES TABLES TO OUTPUT DIRECTORY
        if save is True:
            fmt = 'crossval_{0}_s{1:04d}_i{2:04d}_b{3}.txt'
            predtab.write(os.path.join(self.output_dir, fmt.format('predval', int(seed),
                                                                   int(epochs), self.frac_balance)), format='ascii',
                          fast_writer=False)
            tab.write(os.path.join(self.output_dir, fmt.format('histories', int(seed),
                                                               int(epochs), self.frac_balance)), format='ascii',
                      fast_writer=False)

            # SAVES TEST SET PREDICTIONS IF TRUE
            if pred_test is True:
                pred_test_table.write(os.path.join(self.output_dir, fmt.format('predtest', int(seed),
                                                                               int(epochs), self.frac_balance)),
                                      format='ascii', fast_writer=False)


    def calibration(self, df, metric_threshold):
        """
        Transforming the rankings output by the CNN into actual probabilities.
        This can only be run for an ensemble of models.

        Parameters
        ----------
        df : astropy.Table.table
             Table of output predictions from the validation set.
        metric_threshold : float
             Defines ranking above which something is considered
             a flares.
        """
        # ADD COLUMN TO TABLE THAT CALCULATES THE FRACTION OF MODELS
        # THAT SAY SOMETHING IS A FLARE
        names= [i for i in df.colnames if 's' in i]
        flare_frac = np.zeros(len(df))

        for i, val in enumerate(len(df)):
            preds = np.array(list(df[names][i]))
            flare_frac[i] = len(preds[preds >= threshold]) / len(preds)

        df.add_column(Column(flare_frac, name='flare_frac'))
        
        # !! WORK IN PROGRESS !!

        return df
                       
        
    def predict(self, modelname, times, fluxes, errs, 
                multi_models=False, injected=False):
        """
        Takes in arrays of time and flux and predicts where the flares 
        are based on the keras model created and trained.

        Parameters
        ----------
        modelname : str
             Path and filename of a model to load.
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
        model : tensorflow.python.keras.engine.sequential.Sequential
             The model input with modelname.
        predict_time : np.ndarray
             The input times array.
        predict_flux : np.ndarray
             The input fluxes array.
        predict_err : np.ndarray
             The input flux errors array.
        predictions : np.ndarray
             An array of predictions from the model.
        """

        def identify_gaps(t):
            """
            Identifies which cadences can be predicted on given
            locations of gaps in the data. Will always stay 
            cadences/2 away from the gaps.

            Returns lists of good indices to predict on.
            """
            nonlocal cad_pad

            # SETS ALL CADENCES AVAILABLE
            all_inds = np.arange(0, len(t), 1, dtype=int)

            # REMOVES BEGINNING AND ENDS
            bad_inds = np.arange(0,cad_pad,1,dtype=int)
            bad_inds = np.append(bad_inds, np.arange(len(t)-cad_pad,
                                                     len(t), 1, dtype=int))

            diff = np.diff(t)
            med, std = np.nanmedian(diff), np.nanstd(diff)
            
            bad = np.where(np.abs(diff) >= med + 1.5*std)[0]
            for b in bad:
                bad_inds = np.append(bad_inds, np.arange(b-cad_pad,
                                                         b+cad_pad,
                                                         1, dtype=int))
            bad_inds = np.sort(bad_inds)
            return np.delete(all_inds, bad_inds)

        model = keras.models.load_model(modelname)

        self.model = model

        # GETS REQUIRED INPUT SHAPE FROM MODEL
        cadences = model.input.shape[1]
        cad_pad  = cadences/2

        # REFORMATS FOR A SINGLE LIGHT CURVE PASSED IN
        try:
            times[0][0]
        except:
            times  = [times]
            fluxes = [fluxes]
            errs   = [errs]
        

        predictions = []
        pred_t, pred_f, pred_e = [], [], []
    
        for j in tqdm(range(len(times))):
            time = times[j] + 0.0
            lc   = fluxes[j] / np.nanmedian(fluxes[j]) # MUST BE NORMALIZED
            err  = errs[j] + 0.0

            q = ( (np.isnan(time) == False) & (np.isnan(lc) == False))
            time, lc, err = time[q], lc[q], err[q]
            
            # APPENDS MASKED LIGHT CURVES TO KEEP TRACK OF
            pred_t.append(time)
            pred_f.append(lc)
            pred_e.append(err)

            good_inds = identify_gaps(time)
                
            reshaped_data = np.zeros((len(lc), cadences))

            for i in good_inds:
                loc = [int(i-cad_pad), int(i+cad_pad)]
                f = lc[loc[0]:loc[1]]
                t = time[loc[0]:loc[1]]                    
                reshaped_data[i] = f
            
            reshaped_data = reshaped_data.reshape(reshaped_data.shape[0], 
                                                  reshaped_data.shape[1], 1)
            

            preds = model.predict(reshaped_data)
            preds = np.reshape(preds, (len(preds),))
            predictions.append(preds)
            
        self.predict_time = np.array(pred_t)
        self.predict_flux = np.array(pred_f)
        self.predict_err  = np.array(pred_e)
        self.predictions  = np.array(predictions)
