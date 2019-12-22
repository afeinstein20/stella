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

    def __init__(self, fn_dir, catalog, cadences=200):
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
        """

        self.fn_dir   = fn_dir
        self.catalog  = catalog
        self.cadences = cadences

        self.load_files()


    def load_files(self):
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
        """
        ids = []
        time, flux, flux_err = [], [], []

        for fn in np.sort(os.listdir(self.fn_dir)):
            if fn.endswith('.npy'):
                data = np.load(os.path.join(self.fn_dir, fn), 
                               allow_pickle=True)
                time.append(data[0])
                flux.append(data[1])
                flux_err.append(data[2])
                ids.append(int(fn.split('_')[0]))

        self.ids = np.array(ids)
        self.times = np.array(time)
        self.fluxes = np.array(flux)
        self.flux_errs = np.array(flux_err)


    def reformat_data(self, id_keyword='tic_id', ft_keyword='tpeak',
                      time_offset=2457000, random_seed=321):
        """
        Reformats the data into `cadences`-sized array and assigns
        a label based on flare times defined in the catalog.

        Parameters
        ----------
        id_keyword : str, optional
             The column name of the IDs for each light curve
             in the catalog. Default is 'tic_id'.
        ft_keyword : str, optional
             The column name of the flare peak times in the catalog.
             Default is 'tpeak'. 
        time_offset, float, optional
             Time offset if there is a difference between times saved
             in the catalog and the times of the light curve. Default 
             is 2457000 (correction for TESS time to BJD).
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
        catalog = Table.read(self.catalog, format='ascii')

        # SETUP EMPTY TRAINING MATRIX
        ss = 240000
        training_matrix = np.zeros((ss, self.cadences))
        labels = np.zeros(ss, dtype=int)

        # LOOP THROUGH EACH LIGHT CURVE
        x = 0

        for i in tqdm(range(len(self.times))):
            
            # RENAME LIGHT CURVE VARIABLES
            time = self.times[i]
            flux = self.fluxes[i]
            flux_err = self.flux_errs[i]

            # FIND FLARES IN CATALOG & LOOP THROUGH
            peaks = catalog[ft_keyword][catalog[id_keyword] == 
                                        self.ids[i]].data - time_offset

            flare_time, flare_flux = [], []
            flare_errs = []
            taken = np.array([], dtype=int)

            for p in peaks:
                # INDEX START/END OF DATA CENTERED ON FLARE PEAK
                pi = np.nanmedian(np.where( (time >= p-0.001) &
                                            (time <= p+0.001) )[0])
                if pi > 0:
                    if pi + self.cadences/2 > len(time):
                        end = len(time)
                    else: 
                        end = int(pi + self.cadences/2)

                    if pi - self.cadences/2 < 0:
                        start = 0
                    else:
                        start = int(pi - self.cadences/2)

                    # RECORD FLARE INDEXES AND TIME/FLUX VALUES
                    reg = np.arange(start, end, 1, dtype=int)
                    taken = np.append(taken, reg)
                    
                    flare_time.append(time[reg])
                    flare_flux.append(flux[reg])
                    flare_errs.append(flux_err[reg])
                
            # DIVIDE LIGHT CURVE IMTO EVEN BINS
            c = 0
            while (len(time) - c) % self.cadences != 0:
                c += 1

            time = np.delete(time, np.arange(len(time)-c, len(time), 1, dtype=int))
            flux = np.delete(flux, np.arange(len(flux)-c, len(flux), 1, dtype=int))
            flux_err = np.delete(flux_err, np.arange(len(flux_err)-c, len(flux_err),
                                                     1, dtype=int))

            # RESHAPE DATA 
            time = np.reshape(time, (int(len(time) / self.cadences), self.cadences))
            flux = np.reshape(flux, (int(len(flux) / self.cadences), self.cadences))
            flux_err = np.reshape(flux_err, (int(len(flux_err) / self.cadences),
                                             self.cadences))

            # PUT DATA INTO TRAINING MATRIX AND ASSIGN LABELS
            for j in range(int(len(flare_flux)) + int(len(time))):

                # PUT IN FLARES FIRST
                if j < len(flare_flux):
                    data = flare_flux[j]
                    labels[x] = 1
                else:
                    j = j - len(flare_flux)
                    t, f, e = fill_in(time[j], flux[j], flux_err[j])
                    data = f[0:self.cadences]

                if len(data) == self.cadences:
                    training_matrix[x] = data
                    x += 1

        # DELETE EXTRA END OF TRAINING MATRIX AND LABELS
        training_matrix = np.delete(training_matrix, np.arange(x, ss, 1, dtype=int),
                                    axis=0)
        labels = np.delete(labels, np.arange(x, ss, 1, dtype=int))
        
        # RANDOMIZES THE ORDER OF THE TRAINING_MATRIX 
        np.random.seed(random_seed)
        ind_shuffle = np.random.permutation(training_matrix.shape[0])

        self.training_matrix = training_matrix[ind_shuffle]
        self.labels = labels[ind_shuffle]
