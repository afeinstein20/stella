import os
import warnings
import numpy as np
from tqdm import tqdm
from astropy.table import Table, Row
import numpy.polynomial.polynomial as poly

from .utils import *

__all__ = ['SimulateLightCurves']

class SimulateLightCurves(object):
    """
    A class to create simulated flares for neural network training.
    """

    def __init__(self, sample_size=3000, output_dir=None, cadences=128,
                 activity=True, activity_amp=[0.02,0.1],
                 activity_freq=[1,10], noise=[0.001, 0.025],
                 flare_amp=[0.003, 0.09], ratio=3):
        """
        Parameters 
        ---------- 
        sample_size : int, optional
             The number of light curves to generate.
             Default = 8000.
        output_dir : str, optional
             Path to location where to save the simulated
             flares. Default = '~/.stella'.
        cadences : int, optional
             The number of cadences used in each training set.
             Default = 128.
        activity : bool, optional
             Whether or not to include sinusoidal modulation
             in the training set. Default = True.
        activity_amp : list, optional
             The minimum and maximum amplitude of sinusoidal 
             modulation. Default = [0.1, 0.8].
        activity_freq : list, optional
             The minimum and maximum frequency of sinusoidal 
             modulation. Default = [100,10000].
        noise : list, optional
             The minimum and maximum noise to add on top of the 
             light curve. Noise is drawn from a Gaussian. Defaul = 
             [0.001, 0.01].
        flare_amp : list, optional
             The minimum and maximum flare amplitudes to inject.
             Default = [0.01, 0.08].
        ratio : int, optional
             The ratio of non-flares to flares to create the training
             set out of. Default = 3.

        Attributes
        ----------
        time : np.ndarray
             A simulated time array.
        sample_size : int
             The number of light curves in the simulated set.
        cadences : int
             The number of cadences in each simulated light curve.
        """
        self.cadences = cadences
        self.sample_size = sample_size

        time = np.full((self.sample_size, self.cadences), np.linspace(0, 2.0/1440.0*self.cadences, self.cadences))
        remove = np.arange(self.sample_size*self.cadences, len(time), 1)
        time = np.delete(time, remove)
        self.time = np.reshape(time, (self.sample_size, self.cadences))

        if output_dir is None:
            self.fetch_dir()
        else:
            self.output_dir = output_dir

        if activity is True:
            self.induce_activity(activity_amp, activity_freq)
        
        self.inject_flares(flare_amp, ratio, noise)


    def induce_activity(self, activity_amp, activity_freq):
        """
        Puts in stellar activity modulation to the training set.
        
        Parameters
        ----------
        activity_amp : list
             The minimum and maximum amplitude of sinusoidal
             modulation. 
        activity_freq : list
             The minimum and maximum frequency of sinusoidal 
             modulation.

        Attributes
        ---------- 
        fluxes : np.ndarray
        """

        amps  = np.random.uniform(activity_amp[0], activity_amp[1],
                                 self.sample_size)
        freqs = 1/np.random.uniform(activity_freq[0], activity_freq[1],
                                    self.sample_size)
        phase = np.random.uniform(0, 2*np.pi, self.sample_size)

        fluxes = np.zeros((self.sample_size, self.cadences))

        for i in range(self.sample_size):
            fluxes[i] = amps[i] * np.sin( 2*np.pi*freqs[i]*self.time[i] + phase[i] )

        self.fluxes = fluxes
        self.detrended = np.zeros((self.sample_size, self.cadences))
        return


    def inject_flares(self, flare_amp, ratio, noise):
        """
        Injects flares into the light curves.

        Parameters
        ----------
        flare_amp : list
             The minimum and maximum flare amplitudes to inject.
        ratio : int
             The ratio of non-flares to flares to create the training
             set out of.
        noise : list
             The minimum and maximum noise to add on top of the 
             light curve. Noise is drawn from a Gaussian.

        Attributes
        ---------- 
        simulate_params : astropy.Table

        """
        flare_table = Table(names=["flare_number", "t0", "amp", "dur",
                                   "rise", "decay", "noise_lvl"])

        num  = int(self.sample_size/ratio)
        rand = np.random.randint(0, self.sample_size, num)

        # Sets the labels for each flare
        labels = np.zeros(self.sample_size, dtype=int)
        labels[rand] = 1
        self.labels = labels
        
        # Gets the distribution of flare parameters
        dist = flare_parameters(len(rand), self.cadences,
                                flare_amp, [0.001, 0.006])

        r = 0
        models = np.zeros((len(rand), self.cadences))

        # Adds the flare to the light curve
        for i in tqdm(range(self.sample_size)):

            noise_lvl = np.random.uniform(noise[0], noise[1], 1)[0]
            add_noise = np.random.normal(0, np.abs(noise_lvl), self.cadences)

            if i in rand:   
                amp = np.abs(dist[1][r])
                if amp <= 4*np.std(add_noise):
                    amp += np.max(add_noise)*2
                    amp += 0.02

                flare, row = flare_lightcurve(self.time[i],
                                              amp,
                                              int(dist[0][r]),
                                              np.abs(dist[2][r]),
                                              np.abs(dist[3][r]),
                                              binned=True)
                flare_flux = self.fluxes[i] + flare

                q = np.round(flare,3) < 0
                flare_flux[q] += add_noise[q]

                models[r]  = flare
                self.fluxes[i] = flare_flux

                row = np.append(i, row)
                row = np.append(row, noise_lvl)
                flare_table.add_row(row)
                r += 1
                
#            else:
                # remove tophat
                # remove amplitude relation
#                if i % 2 == 0:
#                    length = np.random.uniform(4,10,1)[0]
#                    amp    = np.random.randint(0, len(dist[1]), 1)[0]
#                    tophat = np.arange(63-length/2, 63+length/2, 1, dtype=int)
#                    self.fluxes[i][tophat] += np.abs(dist[1][amp])
                                       
            self.fluxes[i] += add_noise
            
            self.fluxes[i] = self.fluxes[i] - np.nanmedian(self.fluxes[i]) + 1

            inreg = np.arange(60,70,1,dtype=int)
            outof = np.delete(np.arange(0,len(self.time[i]),1,dtype=int),
                              np.arange(60,70,1,dtype=int))

            coefs = poly.polyfit(self.time[i][outof], self.fluxes[i][outof], 2)
            ffit = poly.polyval(self.time[i], coefs)
            self.detrended[i] = self.fluxes[i]-ffit + np.nanmedian(self.fluxes[i])
            
        self.simulate_params = flare_table
        self.models = models

        return


    def save_training_set(self, output_format='sim{0:04d}.npy'):
        """
        Saves the simulated light curves to npy files. The files are
        saved to the given (or default) output directory.

        Parameters
        ---------- 
        output_format : str, optional
             The naming convention for saving the simulated light curves.
             Default = 'sim{0:04d}.npy'.format(simulation_number).
        """
        for i in tqdm(self.sample_size):
            path = os.path.join(self.output_dir, output_format.format(i))
            data = [self.time[i], self.fluxes[i], self.labels[i]]
            np.save(path, data)
        return


    def save_table(self, output_name="simulate_flare_table.txt"):
        """
        Saves a table of simulated data information.
        """
        self.simulate_params.write(os.path.join(self.output_dir, output_name),
                                   format="ascii")
        return


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
            Path to location of where simulated light curves will be saved to.
        """

        default_path    = os.path.join(os.path.expanduser('~'), '.stella')
        if os.path.isdir(default_path):
            self.output_dir = default_path
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(default_path)
                self.output_dir = default_path
            # downloads locally if OS error occurs
            except OSError:
                output_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Saving simulated light curves to '
                              'working directory instead.'.format(default_path))

                self.output_dir = '.'
