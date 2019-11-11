import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table


__all__ = ['TrainingData']


class TrainingData(object):
    """
    A class that takes in a directory of testing
    light curves and reformats the data to be fed
    into the neural network.
    """

    def __init__(self, fn_dir=None, cadences=200,
                 bins=40, catalog=None):
        """
        Creates a reformatted training set to feed 
        into the convolutional neural network. This
        takes a file directory path and reads in 
        light curves from there. Acceptable file 
        extensions are: `.npy` and `.txt`. Files must
        contain the following information: time, 
        flux, and flux error. File names must contain
        the TIC ID. 
        
        Parameters
        ----------
        fn_dir : str
             The path where the light curves are.
        cadences : int, optional
             The number of cadences used in each 
             training set.
        bins : int, optional
             The number of horizontal bins in the 2D 
             direction.
        catalog : str
             The catalog with flares. Flare times must
             be in a column named "tpeak".
        
        Attributes
        ----------
        fn_dir : str
        image_fmt : np.array
        """

        self.fn_dir     = fn_dir
        self.image_fmt  = np.array([cadences, bins])
        self.catalog    = Table.read(catalog, format='ascii')
        
        self.load_files()
        self.reformat_data()



    def load_files(self):
        """
        Reads in the files from the appropriate
        file directory.
        
        Attributes
        ----------
        tics : np.ndarray
             An array of TIC IDs according to the filename.
        times : np.ndarray
             An array of times.
        fluxes : np.ndarray
             An array of fluxes.
        flux_errs : np.ndarray
             An array of flux errors.
        """
        
        tics, times, fluxes, flux_errs = [], [], [], []

        for fn in np.sort(os.listdir(self.fn_dir)):
            
            tics.append(int(fn.split('_')[0]))

            if fn.endswith('.npy'):
                data = np.load(os.path.join(self.fn_dir, fn), 
                               allow_pickle=True)
                times.append(data[0])
                fluxes.append(data[1])
                flux_errs.append(data[2])

            elif fn.endswith('.txt'):
                try:
                    data = np.loadtxt(os.path.join(self.fn_dir, fn))
                except:
                    # Attempts to catch if there is a header in file
                    data = np.loadtxt(os.path.join(self.fn_dir, fn),
                                      skiprows=1)
                times.append(data[:,0])
                fluxes.append(data[:,1])
                flux_errs.append(data[:,2])

        self.tics     = np.array(tics)
        self.times    = np.array(times)
        self.fluxes   = np.array(fluxes)
        self.flux_err = np.array(flux_errs)
        return


    def reformat_data(self):
        """
        Reformats the data into 2D images and assigns labels.

        Attributes
        ----------
        training_matrix : np.ndarray
             The array of 2D training set data.
        labels : np.array
             An array of labels for each set in training_matrix.
        """
        
        def two_d_image(t, f):
            """
            Turns the data into a 2D image.
            """
            data = np.histogram2d(t, (f - np.nanmax(f)) / np.nanstd(f), bins=self.image_fmt)
            data = np.rot90(data[0])
            return data

        ss = 240000
        training_matrix = np.zeros((ss, self.image_fmt[1], self.image_fmt[0]))
        labels = np.zeros(ss, dtype=int)
        
        x = 0

        for i in tqdm(range(len(self.times))):
            
            subcat = self.catalog[self.catalog['tic_id'] == self.tics[i]]
            peaks  = subcat['tpeak'].data - 2457000

            flare_time, flare_flux = [], []
            taken = np.array([], dtype=int)

            # Find regions around the peaks and set aside
            for p in peaks:
                pi = np.nanmedian(np.where( (self.times[i] >= p - 0.001) & 
                                            (self.times[i] <= p + 0.001))[0])
                # Sets the beginning and end of each chunk of data
                if pi > 0:
                    if pi + self.image_fmt[0]/2 > len(self.times[i]):
                        end = len(self.times[i])
                    else:
                        end = int(pi + self.image_fmt[0]/2)
                    if pi - self.image_fmt[0]/2 < 0:
                        start = 0
                    else:
                        start = int(pi - self.image_fmt[0]/2)

                    reg = np.arange(start, end, 1, dtype=int)
                    taken = np.append(taken, reg)
                    flare_time.append(self.times[i][reg])
                    flare_flux.append(self.fluxes[i][reg])

            total = np.arange(0, len(self.times[i]), 1, dtype=int)
            total = np.delete(total, taken)
            time  = self.times[i][total]
            flux  = self.fluxes[i][total]

            # Divides into even bins
            c = 0
            while (len(time) - c) % self.image_fmt[0] != 0:
                c += 1

            time = np.delete(time, np.arange(len(time)-c, len(time), 1, dtype=int) )
            flux = np.delete(flux, np.arange(len(flux)-c, len(flux), 1, dtype=int) )

            time = np.reshape(time, (int(len(time)/self.image_fmt[0]), self.image_fmt[0]) )
            flux = np.reshape(flux, (int(len(flux)/self.image_fmt[0]), self.image_fmt[0]) )

            
            for j in range(int(len(time) + len(flare_flux))):
                if j < len(flare_flux):
                    data = two_d_image(flare_time[j], flare_flux[j])
                    labels[x] = 1
                else:
                    j = j - len(flare_flux)
                    data = two_d_image(time[j], flux[j])
                training_matrix[x] = data
                x += 1

        training_matrix = np.delete(training_matrix, np.arange(x, ss, 1, dtype=int), axis=0)
        labels = np.delete(labels, np.arange(x, ss, 1, dtype=int))
            
        self.training_matrix = training_matrix
        self.labels = labels
