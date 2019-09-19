import numpy as np
from tqdm import tqdm
from astropy.table import Table

from .utils import *

__all__ = ['FlareCharacterization']

class FlareCharacterization(object):
    """
    A class that classifies flares in a given data set.
    """

    def __init__(self, nn=None, time=None, flux=None, flux_err=None, labels=None):
        """
        Parameters
        ---------- 
        nn : stella.NeuralNetwork, optional
             The neural network and labels for specified input data.
             This only works if stella.NeuralNetwork.predict() was called.
        time : np.ndarray, optional
             Time array.
        flux : np.ndarray, optional
             Flux array.
        labels : np.nparray, optional
             An array of same shape as time and flux that consists of values
             between 0 and 1, as probabilities that data point is part of 
             a flare.
        """

        if nn is not None:
            if nn.predictions is None:
                raise ValueError("Please call stella.NeuralNetwork.predict first.")
            else:
                time     = nn.time
                flux     = nn.flux
                flux_err = nn.flux_err
                labels   = nn.predictions

        self.nn       = nn
        self.time     = time
        self.flux     = flux
        self.labels   = labels
        self.flux_err = flux_err


    def section_by_flare(self, time, flux, flux_err, labels, region_size):
        """
        Splits the data into just the regions around the flare event.

        Parameters
        ---------- 
        region_size : int
             The number of cadences to be taken on both sides of the 
             peak of the flare.
        """
        flare = np.where(labels[:,1]>prob_accept)[0]
        flare_regions = np.where(np.diff(flares)>1)[0]

        time_peaks, flux_peaks = [], []
        flux_err_peaks = []

        for i in range(len(flare_regions)):
            if i == 0:
                r = (flares <= flares[flare_regions[i]])
            else:
                r = ((flares > flares[flare_regions[i-1]]) & (flares <= flares[flare_regions[i]]))

            median_ind = np.argsort(flares[r])[len(flares[r])//2]
            peak       = flares[r][median_ind]
            time_peaks.append( time[int((peak-region_size/1.5)): int(peak+region_size)] )
            flux_peaks.append( flux[int((peak-region_size/1.5)): int(peak+region_size)] )
            flux_err_peaks.append( flux_err[int((peak-region_size/1.5)): int(peak+region_size)] )

        return np.array(time_peaks), np.array(flux_peaks), np.array(flux_err_peaks)



    def flares(self, prob_accept=0.75, region_size=16):
        """
        Uses labels to identify flares and recover the following parameters:
        equivalent duration, amplitude, time of flare peak.

        Parameters
        ----------
        prob_accept : float, optional
             The label probability for which to find flares
             above this threshold. Default = 0.75.
        region_size : int
             The number of cadences to be taken on both sides of the
             peak of the flare.    

        Attributes
        ----------
        flare_table : astropy.Table
             A table of flare parameters.
        """
        t = Table(names=["ed", "ed_err", "amp", "amp_err", "Peak [time]"])

        try:
            self.flux.shape[1]
            for i in range(len(self.flux)):
                tp, fp, fep = self.section_by_flare(self.time[i], self.flux[i],
                                                    self.flux_err[i],
                                                    self.labels[i], region_size)
        except:
            time_section, flux_section, flux_err_section = self.section_by_flare(self.time,
                                                                                 self.flux,
                                                                                 self.flux_err,
                                                                                 self.labels,
                                                                                 region_size)


        for i in tqdm(range(len(time_section))):
            peak = np.argmax(flux_section[i])

            if self.labels[i][:,1][peak] > prob_accept:

                t0  = self.time[i][peak]

                amp = np.nanmax(flux_section[i])
                random_flux = np.zeros((len(self.flux[i]),200))
                for j in range(len(self.flux[i])):
                    random_flux[i] = np.random.normal(self.flux[i][j], self.flux_err[i][j], 200)
                amp_err = np.std(np.nanmax(random_flux, axis=1))

                dur     = np.abs(np.sum(self.flux[i][:-1]    * np.diff(self.time[i]) ))
                dur_err = np.abs(np.sum(self.flux_err[i][:-1]* np.diff(self.time[i]) ))

                t.add_row([dur, dur_err, amp, amp_err, t0])

        self.flare_table = t


    def completeness(self, trials=100, amplitudes=[0.01,0.08],
                     decays=[0.05,0.15], rises=[0.001,0.005]):
        """
        Injection and recovery of fake flares into your input data. 

        Parameters
        ----------
        trials : int, optional
             The number of injections completed. Default = 100.
        amplitudes : list, optional 
             List of mean and std of flare amplitudes to draw from a
             normal distritbution. Default = [0.01, 0.08].
        decays : list, optional
             List of minimum and maximum exponential decay rate to draw from a
             uniform distribution. Default = [0.05, 0.15].
        rises : list, optional
             List of minimum and maximum Gaussian rise rate to draw from a
             uniform distribution. Default = [0.001, 0.006] 
        """
        models = []
        t = Table(names=['flare', 't0', 'amplitude',
                            'duration', 'rise', 'decay'])

        for i in range(self.time.shape[0]):
            flare_params = flare_parameters(trials, len(self.time[i]),
                                            128, amplitudes, rises, decays)
            models = []
            for j in range(trials):
                m, row = flare_lightcurve(self.time[i], np.abs(flare_params[1][j]), 
                                          flare_params[0][j],
                                          flare_params[2][j], 
                                          flare_params[3][j])
                row = np.append([j], row)
                t.add_row(row)

                pred = self.nn.predict(self.time[i], m+self.flux[i], 
                                       self.flux_err[i], injection=True)

                models.append(pred)
        return t, models
