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

    def __init__(self, ts, training=0.80, validation=0.90,
                 layers=None, optimizer='adam',
                 loss='binary_crossentropy', 
                 metrics=None, seed=2, output_dir=None):
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
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.training_matrix = np.copy(ts.training_matrix)
        self.labels = np.copy(ts.labels)
        self.cadences = np.copy(ts.cadences)
        self.seed = seed
        self.frac_balance = ts.frac_balance + 0.0

        self.tpeaks = ts.training_peaks
        self.training_ids = ts.training_ids
        self.prec_recall_curve = None
        self.history = None
        self.history_table = None

        if output_dir is None:
            self.fetch_dir()
        else:
            self.output_dir = output_dir

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
        # SETS RANDOM SEED FOR REPRODUCABLE RESULTS
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

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
        
        # PRINTS THE MODEL SUMMARY FOR THE USER
        model.summary()


    def train_model(self, epochs=350, batch_size=64, shuffle=True):
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
        shuffle : bool, optional
             Whether or not to shuffle the training set batches.
             Default is True.

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

        self.train_data = x_train
        self.train_labels = y_train
        
        self.test_data = x_test
        self.test_labels = y_test
        self.test_ids = self.training_ids[self.val_cutoff:]
        self.test_tpeaks = self.tpeaks[self.val_cutoff:]

        self.val_data = x_val
        self.val_labels = y_val
        self.val_ids = self.training_ids[self.train_cutoff:self.val_cutoff]
        self.val_tpeaks = self.tpeaks[self.train_cutoff:self.val_cutoff]

        self.history = self.model.fit(x_train, y_train, epochs=epochs, 
                                      batch_size=batch_size, shuffle=shuffle,
                                      validation_data=(x_val, y_val))


    def train_multi_models(self, n=5, seeds=None, epochs=150, batch_size=64,
                           metric_threshold=0.5):
        """
        Runs n number of models with given initial random seeds of
        length n. Also saves each model run to a hidden ~/.stella 
        directory. 

        Parameters
        ----------
        n : int, optional
             Number of models to loop through. Default is 5.
        seeds : np.array, optional
             Array of random seed starters of length n, where
             n is the number of models you want to run.
        save : bool, optional
             Tells whether or not to save the model histories
             to a .txt file. Default is False.
        metric_threshold : float, optional
             Defines the threshold for positive vs. negative cases. Default
             is 0.5 (50%). 

        Attributes
        ----------
        history_table : Astropy.table.Table
             Saves the metric values for each model run.
        multi_predictions : np.ndarray
             Array of all the predictions from each model run.
        """
        self.predval_fmt = 'predval_s{0:04d}_i{1:04d}_b{2}.txt'
        self.predtest_fmt = 'predtest_s{0:04d}_i{1:04d}_b{2}.txt'
        self.epochs = epochs

        if len(seeds) != n:
            print("Please input {}-random seeds. You put in {}.".format(n, 
                                                                        len(seeds)))
            return

        else:
            table = Table()
            all_predictions = []
            
            pred_fn = os.path.join(self.output_dir,'{0:09d}_seed{1:03d}.npy')

            for seed in seeds:
                self.seed = seed
                keras.backend.clear_session()

                # CREATES MODEL BASED ON GIVEN RANDOM SEED
                self.create_model()
                self.train_model(epochs=epochs, batch_size=batch_size)

                col_names = list(self.history.history.keys())
                for cn in col_names:
                    col = Column(self.history.history[cn], name=cn+'{0:04d}'.format(seed))
                    table.add_column(col)

                # SAVES THE MODEL TO OUTPUT DIRECTORY
                self.model.save(os.path.join(self.output_dir, 'model_{0:04d}.h5'.format(seed)))

                # GETS PREDICTIONS FOR EACH LIGHT CURVE
                val_preds = self.model.predict(self.val_data)

                np.savetxt(os.path.join(self.output_dir, self.predval_fmt.format(int(seed), 
                                                                                 int(epochs),
                                                                                 self.frac_balance)),
                           np.column_stack((self.val_ids, val_preds, self.val_labels,
                                            self.val_tpeaks)),
                           fmt=['%.0f', '%0.6f', '%0.6f', '%.10f'], delimiter=',', 
                           header='tic,pred,gt,tpeak')

                test_preds = self.model.predict(self.test_data)

                np.savetxt(os.path.join(self.output_dir, self.predtest_fmt.format(int(seed), 
                                                                                  int(epochs),
                                                                                  self.frac_balance)),
                           np.column_stack((self.test_ids, test_preds, self.test_labels,
                                            self.test_tpeaks)),
                           fmt=['%.0f','%0.6f', '%0.6f', '%.10f'], delimiter=',',
                           header='tic,pred,gt,tpeak')

            self.history_table = table
            self.multi_predictions = all_predictions

            table.write(os.path.join(self.output_dir, 'model_histories.txt'), format='ascii')

            self.ensemble_metrics(metric_threshold)
            

    def create_df(self, threshold, mode='metrics'):
        """
        Creates an astropy.Table.table for the ensemble metrics.
        """
        files = np.sort(glob.glob(os.path.join(self.output_dir,
                                               "predval*i{0:04d}*_b{1}.txt".format(self.epochs,
                                                                                   self.frac_balance))))
        if mode is 'metrics' and len(files) < 2:
            raise ValueError("Can only calculate metrics for multiple models.")
        
        
        df= Table()
        mean_arr = []
        
        for i, val in enumerate(files):
            r = Table.read(val, format="csv")
            
            df.add_column(Column(np.round(r['pred'].data, 3),
                                 name='s'+(val.split('_s')[-1]).split('_')[0]))
            
            mean_arr.append(np.round(r['pred'].data, 3))
            
        df.add_column(Column(np.nanmean(mean_arr, axis=0), name='pred'))
        
        df.add_column(Column(r['gt'].data, name='gt'), index=0)
        df.add_column(Column(r['tpeak'].data, name='peak'), index=0)
        df.add_column(Column(r['# tic'].data, name='tic'), index=0)

        pred_round = np.zeros(len(df))
        pred_round[df['pred'] >= threshold] = 1
        pred_round[df['pred'] < threshold]  = 0
        df.add_column(Column(pred_round, name='pred_round'), index=0)

        return df

    def ensemble_metrics(self, threshold):
        """
        Calculates the metrics and average metrics when ensemble training.

        Parameters
        ----------
        threshold : float
             Defines the threshold for positive vs. negative cases.

        Attributes
        ----------
        average_precision : float
        accuracy : float
        recall_score : float
        precision_score : float
        prec_recall_curve : np.array
             2D array of precision and recall for plotting purposes.
        """
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        df = self.create_df(threshold)
        ap, ac = [], []

        rs, ps = [], []
        for i, val in enumerate(df.columns[4:]):
    
            # CALCULATES AVERAGE PRECISION SCORE
            ap.append(np.round(average_precision_score(df['gt'].data, 
                                                       df[val].data, average=None), 4))
            # ROUNDED BASED ON THRESHOLD
            arr = np.copy(df[val].data)
            arr[arr >= threshold] = 1.0
            arr[arr < threshold] = 0.0
        
            # CALCULATES ACCURACY
            ac.append(np.round(np.sum(arr == df['gt'].data) / len(df), 4))

        # CALCULATES RECALL SCORE
        rs = np.round(recall_score(df['gt'], df['pred_round']), 4)
        
        # CALCULATES PRECISION SCORE
        ps = np.round(precision_score(df['gt'], df['pred_round']), 4)

        # PRECISION RECALL CURVE
        prec_curve, rec_curve, _ = precision_recall_curve(df['gt'], df['pred_round'])

        self.average_precision = ap[-1]
        self.accuracy = ac[-1]
        self.recall_score = rs
        self.precision_score = ps
        self.prec_recall_curve = np.array([rec_curve, prec_curve])
                        
        
    def predict(self, times, fluxes, errs, injected=False):
        """
        Takes in arrays of time and flux and predicts where the flares 
        are based on the keras model created and trained.

        Parameters
        ----------
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
        predict_times : np.ndarray
             The input times array.
        predict_fluxes : np.ndarray
             The input fluxes array.
        predict_errs : np.ndarray
             The input flux errors array.
        predictions : np.ndarray
             An array of predictions from the model.
        """
        def fill_in_sample(t, f, e, sigma=2.5):
            # FILLS IN GAPS IN THE DATA FOR CHUNKING TO FIND FLARES
            t, f = np.array(t), np.array(f)
            
            diff = np.diff(t)
            
            diff_ind = np.where( diff >= (np.nanmedian(diff) + sigma*np.nanstd(diff)) )[0]
            avg_noise = np.nanstd(f) / 2.0
            
            if len(diff_ind) > 0:
                for i in diff_ind:
                    start = i
                    stop  = int(i + 2)
                    func = interp1d(t[start:stop], f[start:stop])
                    new_time = np.arange(t[start], 
                                         t[int(start+1)],
                                         np.nanmedian(diff))
                    new_flux = func(new_time) + np.random.normal(0, avg_noise,
                                                                 len(new_time))
                    t = np.insert(t, i, new_time)
                    f = np.insert(f, i, new_flux)
                    e = np.insert(e, i,
                                  np.full(len(new_time), avg_noise))
            t, f = zip(*sorted(zip(t, f)))
            return t, f, e


        predictions = []

        cadences = self.cadences + 0
        
        new_time = []
        new_flux = []
    
        for j in tqdm(range(len(times))):
            q = np.isnan(fluxes[j]) == False
            time = times[j]#[q]
            lc   = fluxes[j]#[q]
            err  = errs[j]#[q]
            
            time, lc, err = fill_in_sample(time, lc, err)
        
            # LIGHT CURVE MUST BE NORMALIZED
            lc = lc/np.nanmedian(lc)
    
            new_time.append(time)
            new_flux.append(lc)
            
            reshaped_data = np.zeros((len(lc), cadences))
            
            padding       = np.nanmedian(lc)
            std           = np.std(lc)/4.5
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
            preds = np.reshape(preds, (len(preds),))
            predictions.append(preds)
            
        return predictions


    def fetch_dir(self):
        """
        Returns the default path to the directory where files will be saved
        or loaded.
        By default, this method will return "~/.stella" and create
        this directory if it does not exist.  If the directory cannot be
        access or created, then it returns the local directory (".").
        
        Attributes
        -------
        output_dir : str
        Path to location of saved CNN models.
        """
        download_dir    = os.path.join(os.path.expanduser('~'), '.stella')
        if os.path.isdir(download_dir):
            self.output_dir = download_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(download_dir)
                # downloads locally if OS error occurs
            except OSError:
                download_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Saving models to the current '
                              'working directory instead.'.format(download_dir))
                    
            self.output_dir = download_dir
