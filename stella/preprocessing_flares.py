import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from scipy.interpolate import interp1d

from .utils import break_rest, do_the_shuffle, split_data

__all__ = ['FlareDataSet']


class FlareDataSet(object):
    """
    Given a directory of files, reformat data to
    create a training set for the convolutional
    neural network.
    Files must be in '.npy' file format and contain
    at minimum the following indices:
         - 0th index = array of time
         - 1st index = array of flux
         - 2nd index = array of flux errors
    All other indices in the files are ignored.
    This class additionally requires a catalog of flare
    start times for labeling. The flare catalog can be
    in either '.txt' or '.csv' file format. This class will
    be passed into the stella.neural_network() class to 
    create and train the neural network.
    """

    def __init__(self, fn_dir=None, catalog=None, 
                 downloadSet=None,
                 cadences=200, frac_balance=0.73,
                 training=0.80, validation=0.90):
        """
        Loads in time, flux, flux error data. Reshapes
        arrays into `cadences`-sized bins and labels
        flares vs. non-flares using the input catalog.

        Parameters
        ----------
        fn_dir : str, optional
             The path to where the files for the training
             set are stored.
        catalog : str, optional
             The path and filename of the catalog with 
             marked flare start times
        downloadSet : stella.DownloadSets, optional
             The stella.DownloadSets class, which contains the 
             flare catalog name and directory where light curves
             and the catalog are saved.
        cadences : int, optional
             The size of each training set. Default is 200.
        frac_balance : float, optional 
             The amount of the negative class to remove.
             Default is 0.75.
        training : float, optional
             Assigns the percentage of training set data for the
             model. Default is 80%
        validation : float, optionl
             Assigns the percentage of validation and testing set
             data for the model. Default is 90%.
        """
        if fn_dir is not None:
            self.fn_dir   = fn_dir
        if catalog is not None:
            self.catalog  = Table.read(catalog, format='ascii')

        if downloadSet is not None:
            self.fn_dir = downloadSet.fn_dir
            self.catalog = downloadSet.flare_table

        self.cadences = cadences

        self.frac_balance = frac_balance
        self.load_files()
        self.reformat_data()

        misc = split_data(self.labels, self.training_matrix, 
                          self.training_ids, self.training_peaks,
                          training, validation)

        self.train_data   = misc[0]
        self.train_labels = misc[1]

        self.val_data = misc[2]
        self.val_labels = misc[3]
        self.val_ids = misc[4]
        self.val_tpeaks = misc[5]

        self.test_data = misc[6]
        self.test_labels = misc[7]

        self.test_ids = misc[8]
        self.test_tpeaks = misc[9]


    def load_files(self, id_keyword='TIC', ft_keyword='tpeak',
                   time_offset=2457000.0):
        """
        Loads in light curves from the assigned training set
        directory. Files must be formatted such that the ID 
        of each star is first and followed by '_' 
        (e.g. 123456789_sector09.npy).

        Attributes
        ----------
        times : np.ndarray
             An n-dimensional array of times, where n is the 
             number of training set files.
        fluxes : np.ndarray
             An n-dimensional array of fluxes, where n is the
             number of training set files.
        flux_errs : np.ndarray
             An n-dimensional array of flux errors, where n is
             the number of training set files.
        ids : np.array
             An array of light curve IDs for each time/flux/flux_err.
             This is essential for labeling flare events.
        id_keyword : str, optional
             The column header in catalog to identify target ID. 
             Default is 'tic_id'.
        ft_keyword : str, optional
             The column header in catalog to identify flare peak time.
             Default is 'tpeak'.
        time_offset : float, optional 
             Time correction from flare catalog to light curve and is
             necessary when using Max Guenther's catalog. 
             Default is 2457000.0
        """

        print("Reading in training set files.")

        files = os.listdir(self.fn_dir)
        
        files = np.sort([i for i in files if i.endswith('.npy') and 'sector' in i])
    
        tics, time, flux, err, tpeaks = [], [], [], [], []
        
        for fn in files:
            data = np.load(os.path.join(self.fn_dir, fn), allow_pickle=True)
            split_fn = fn.split('_')
            tic = int(split_fn[0])
            tics.append(tic)
            sector = int(split_fn[1].split('r')[1][0:2])
            time.append(data[0])
            flux.append(data[1])
            err.append( data[2])
        
            peaks = self.catalog[(self.catalog[id_keyword] == tic)][ft_keyword].data 
#                            (self.catalog['sector'] == sector)][ft_keyword].data
            peaks = peaks - time_offset
            tpeaks.append(peaks)

        self.ids      = np.array(tics)
        self.time     = np.array(time, dtype=np.ndarray)   # in TBJD
        self.flux     = np.array(flux, dtype=np.ndarray)
        self.flux_err = np.array(err,  dtype=np.ndarray)
        self.tpeaks   = tpeaks # in TBJD

    def reformat_data(self, random_seed=321):
        """
        Reformats the data into `cadences`-sized array and assigns
        a label based on flare times defined in the catalog.

        Parameters
        ----------
        random_seed : int, optional
             A random seed to set for randomizing the order of the
             training_matrix after it is constructed. Default is 321.

        Attributes
        ----------
        training_matrix : np.ndarray
             An n x `cadences`-sized array used as the training data.
        labels : np.array
             An n-sized array of labels for each row in the training
             data.
        """
        ss = 240000

        training_matrix = np.zeros((ss, self.cadences))
        training_labels = np.zeros(ss, dtype=int)
        training_peaks  = np.zeros(ss)
        training_ids    = np.zeros(ss)
        
        x = 0
        
        for i in tqdm(range(len(self.time))):
            flares = np.array([], dtype=int)
            
            for peak in self.tpeaks[i]:
                arg = np.where((self.time[i]>(peak-0.02)) & (self.time[i]<(peak+0.02)))[0]
                # DOESN'T LIKE FLARES AT THE VERY END OF THE LIGHT CURVE 
                # (AND NEITHER DO I)
                if len(arg) > 0:
                    closest = arg[np.argmin(np.abs(peak - self.time[i][arg]))]
                    start = int(closest-self.cadences/2)
                    end   = int(closest+self.cadences/2)
                    if start < 0:
                        start = 0
                        end = self.cadences
                    if end > len(self.time[i]):
                        start = start - (end - len(self.time[i]))
                        end = len(self.time[i])
                    flare_region = np.arange(start, end,1,dtype=int)
                    flares = np.append(flares, flare_region)
                
                    # ADD FLARE TO TRAINING MATRIX & LABEL PROPERLY
                    training_peaks[x]  = self.time[i][closest] + 0.0
                    training_ids[x]   = self.ids[i] + 0.0 
                    training_matrix[x] = self.flux[i][flare_region]
                    training_labels[x] = 1
                    x += 1
                
            time_removed = np.delete(self.time[i], flares)
            flux_removed = np.delete(self.flux[i], flares)
            flux_err_removed = np.delete(self.flux_err[i], flares)

            nontime, nonflux, nonerr = break_rest(time_removed, flux_removed, 
                                                  flux_err_removed, self.cadences)
            for j in range(len(nonflux)):
                if x >= ss:
                    break
                else:
                    training_ids[x] = self.ids[i] + 0.0
                    training_peaks[x] = nontime[j][int(self.cadences/2)]
                    training_matrix[x] = nonflux[j]
                    training_labels[x] = 0
                    x += 1

        # DELETE EXTRA END OF TRAINING MATRIX AND LABELS
        training_matrix = np.delete(training_matrix, np.arange(x, ss, 1, dtype=int), axis=0)
        labels          = np.delete(training_labels, np.arange(x, ss, 1, dtype=int))
        training_peaks  = np.delete(training_peaks, np.arange(x, ss, 1, dtype=int))
        training_ids    = np.delete(training_ids, np.arange(x, ss, 1, dtype=int))

        ids, matrix, label, peaks = do_the_shuffle(training_matrix, labels, training_peaks, 
                                                   training_ids, self.frac_balance)
        self.labels = label
        self.training_peaks  = peaks
        self.training_ids    = ids
        self.training_matrix = matrix
        
