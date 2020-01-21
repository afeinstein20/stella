import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from scipy.interpolate import interp1d

from .utils import fill_in

__all__ = ['TrainingSet']


class TrainingSet(object):
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
    be passed into the stella.neural_networ() class to 
    create and train the neural network.
    """

    def __init__(self, fn_dir, catalog, cadences=200, frac_balance=0.75):
        """
        Loads in time, flux, flux error data. Reshapes
        arrays into `cadences`-sized bins and labels
        flares vs. non-flares using the input catalog.

        Parameters
        ----------
        fn_dir : str
             The path to where the files for the training
             set are stored.
        catalog : str
             The path and filename of the catalog with 
             marked flare start times
        cadences : int, optional
             The size of each training set. Default is 200.
        frac_balance : float, optional 
             The amount of the negative class to remove.
             Default is 0.75.
        """

        self.fn_dir   = fn_dir
        self.catalog  = Table.read(catalog, format='ascii')
        self.cadences = cadences

        self.frac_balance = frac_balance


    def load_files(self, id_keyword='tic_id', ft_keyword='tpeak',
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
        
        files = [i for i in files if i.endswith('.npy') and 'sector' in i]
    
        tics, time, flux, err, tpeaks = [], [], [], [], []
        
        for fn in files:
            data = np.load(os.path.join(self.fn_dir, fn))
            split_fn = fn.split('_')
            tic = int(split_fn[0])
            tics.append(tic)
            sector = int(split_fn[1].split('r')[1][0:2])
            time.append(data[0])
            flux.append(data[1])
            err.append( data[2])
        
            peaks = self.catalog[(self.catalog[id_keyword] == tic) & 
                            (self.catalog['sector'] == sector)][ft_keyword].data
            peaks = peaks - time_offset
            tpeaks.append(peaks)

        self.ids      = np.array(tics)
        self.time     = np.array(time)   # in TBJD
        self.flux     = np.array(flux)
        self.flux_err = np.array(err)
        self.tpeaks   = np.array(tpeaks) # in TBJD


    def break_rest(self, time, flux, flux_err):
        """
        Breaks up the non-flare cases into bite-sized cadence-length chunks.
        """
        diff = np.diff(time)
        breaking_points = np.where(diff > (np.nanmedian(diff) + 2.0*np.nanstd(diff)))[0]
        
        tot = 200
        ss  = 1000
        nonflare_time = np.zeros((ss, self.cadences))
        nonflare_flux = np.zeros((ss, self.cadences))
        nonflare_err = np.zeros((ss,  self.cadences))
    
        x = 0
        for j in range(len(breaking_points)+1):
            if j == 0:
                start = 0
                end = breaking_points[j]
            elif j < len(breaking_points):
                start = breaking_points[j-1]
                end = breaking_points[j]
            else:
                start = breaking_points[-1]
                end = len(time)

            if (end-start) > self.cadences:
                # DIVIDES LIGHTCURVE INTO EVEN BINS
                c = 0
                while (len(time) - c) % self.cadences != 0:
                    c += 1

                # REMOVING CADENCES TO BIN EVENLY INTO CADENCES
                temp_time = np.delete(time, np.arange(len(time)-c, 
                                                      len(time), 1, dtype=int) )
                temp_flux = np.delete(flux, np.arange(len(flux)-c, 
                                                      len(flux), 1, dtype=int) )
                temp_err = np.delete(flux_err, np.arange(len(flux_err)-c, 
                                                         len(flux_err), 1, dtype=int) )
        
                # RESHAPE ARRAY FOR INPUT INTO MATRIX
                temp_time = np.reshape(temp_time, 
                                       (int(len(temp_time) / self.cadences), self.cadences) )
                temp_flux = np.reshape(temp_flux, 
                                       (int(len(temp_flux) / self.cadences), self.cadences) )
                temp_err  = np.reshape(temp_err, 
                                       (int(len(temp_err) /  self.cadences), self.cadences) )

                # APPENDS TO BIGGER MATRIX 
                for f in range(len(temp_flux)):
                    if x >= ss:
                        break
                    else:
                        nonflare_time[x] = temp_time[f]
                        nonflare_flux[x] = temp_flux[f]
                        nonflare_err[x] = temp_err[f]
                        x += 1
    
        np.random.seed(101)
        rand = np.random.randint(0,x,tot)
        nonflare_time = np.delete(nonflare_time, np.arange(x, ss, 1, dtype=int), axis=0)
        nonflare_flux = np.delete(nonflare_flux, np.arange(x, ss, 1, dtype=int), axis=0)
        nonflare_err  = np.delete(nonflare_err,  np.arange(x, ss, 1, dtype=int), axis=0)
        
        return nonflare_time[rand], nonflare_flux[rand], nonflare_err[rand]


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
                    flares = np.append(flares,flare_region)
                
                    # ADD FLARE TO TRAINING MATRIX & LABEL PROPERLY
                    training_peaks[x]  = self.time[i][closest] + 0.0
                    training_ids[x]   = self.ids[i] + 0.0 
                    training_matrix[x] = self.flux[i][flare_region]
                    training_labels[x] = 1
                    x += 1
                
            time_removed = np.delete(self.time[i], flares)
            flux_removed = np.delete(self.flux[i], flares)
            flux_err_removed = np.delete(self.flux_err[i], flares)
        
            nontime, nonflux, nonerr = self.break_rest(time_removed, flux_removed, 
                                                       flux_err_removed)
            for j in range(len(nonflux)):
                if x >= ss:
                    break
                else:
                    training_matrix[x] = nonflux[j]
                    training_labels[x] = 0
                    x += 1

        # DELETE EXTRA END OF TRAINING MATRIX AND LABELS
        training_matrix = np.delete(training_matrix, np.arange(x, ss, 1, dtype=int), axis=0)
        labels          = np.delete(training_labels, np.arange(x, ss, 1, dtype=int))
        training_peaks  = np.delete(training_peaks, np.arange(x, ss, 1, dtype=int))
        training_ids    = np.delete(training_ids, np.arange(x, ss, 1, dtype=int))

        self.do_the_shuffle(training_matrix, labels, training_peaks, 
                            training_ids, random_seed)
        

    def do_the_shuffle(self, training_matrix, labels, training_peaks, training_ids, seed):
        """
        Shuffles the data in a random order and fixes data inbalance based on
        frac_balance.
        """
        np.random.seed(seed)

        ind_shuffle = np.random.permutation(training_matrix.shape[0])

        labels2 = np.copy(labels[ind_shuffle])
        matrix2 = np.copy(training_matrix[ind_shuffle])
        peaks2  = np.copy(training_peaks[ind_shuffle])
        ids2    = np.copy(training_ids[ind_shuffle])

        # INDEX OF NEGATIVE CLASS
        ind_nc = np.where(labels2 == 0)
        
        # RANDOMIZE INDEXES
        random_seed = 123
        np.random.seed(random_seed)
        ind_nc_rand = np.random.permutation(ind_nc[0])

        # REMOVE FRAC_BALANCE% OF NEGATIVE CLASS
        length = int(self.frac_balance * len(ind_nc_rand))

        self.labels = np.delete(labels2, ind_nc_rand[0:length])
        self.training_peaks  = np.delete(peaks2 , ind_nc_rand[0:length])
        self.training_ids    = np.delete(ids2   , ind_nc_rand[0:length])
        self.training_matrix = np.delete(matrix2, ind_nc_rand[0:length], axis=0)
        
