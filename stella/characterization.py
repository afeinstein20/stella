import numpy as np
from tqdm import tqdm
import more_itertools as mit
from scipy import interpolate
from astropy.table import Table
from scipy.optimize import minimize
from astropy.stats import sigma_clip

from .utils import *

__all__ = ['FlareCharacterization']

class FlareCharacterization(object):
    """
    A class that classifies flares in a given data set.
    """

    def __init__(self, nn=None, time=None, flux=None, flux_err=None, labels=None,
                 prob_accept=0.75):
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
        prob_accept : float, optional
             The acceptance probability that something is a flare. Must be 
             between 0 and 1. Default = 0.75.
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
        self.prob_accept = prob_accept

        self.find_flares()
        self.fit_flares()


    def find_flares(self, injection=False):
        """
        Finds the time of flare peak for each flare above
        the accepted flare probability.

        Parameters
        ----------
        injection : bool, optional
             For injection is True, returns values rather than
             setting the attributes in the class. Default is 
             False.

        Attributes
        ---------- 
        all_flux_flares : np.ndarray
             An array of consecutive indices where flares occur.
        flare_t0s : np.ndarray
             The peak of each flare.
        flare_amps : np.ndarray
             The amplitude of each flare.
        """

        all_flux_flares = []
        t0s, amps = [], []

        for i in range(len(self.flux)):
            where = np.argwhere(self.labels[i][:,1] > self.prob_accept)
            l_int = np.array([], dtype=int)
            for w in where:
                l_int = np.append(l_int, w)
            flares = [list(group) for group in mit.consecutive_groups(l_int)]
            
            sub_t0, sub_amp = [], []

            for f in flares:
                peak = np.argmax(self.flux[i][f])
                sub_t0.append(self.time[i][f][peak])
                sub_amp.append(self.flux[i][f][peak])

            all_flux_flares.append(flares)
            t0s.append(sub_t0)
            amps.append(sub_amp)

        if injection is False:
            self.all_flux_flares = all_flux_flares
            self.flare_t0s = t0s
            self.flare_amps=amps
                
        else:
            return all_flux_flares, t0s, amps


    def fit_flares(self):
        """
        Fits a a model of stellar activity and flare around each
        peak idenitfied in the light curve.
        """

        tab = Table(names=["t0", "amp", "rise", "decay", "amp_fit"])
        
        def chisquare(var, x, y, yerr):
            # Fits the flare and an underlying polynomial to extract
            # flare parameters.
            amp, t0, rise, fall = var

            sig_mask = sigma_clip(y, sigma=2.5).mask
            
            poly_fit = np.polyfit(x[~sig_mask], y[~sig_mask], deg=2)
            poly_model = np.poly1d(poly_fit)

            m = flare_lightcurve(x, amp, int(t0), rise, fall, y=poly_model(x))[0]+np.nanmedian(y)
            return np.sum( (y-m)**2/yerr**2 )


        models = []
        for i in range(len(self.flux)):
            submodels = []

            for t in range(len(self.flare_t0s[i])):

                # amplitude, t0, rise, decay, order fit
                init_guess = [np.abs(self.flare_amps[i][t]-1),
                              int(self.flare_t0s[i][t]),
                              0.005, 0.008]

                mini = minimize(chisquare, x0=init_guess,
                                args=(self.time[i], self.flux[i]-np.nanmedian(self.flux[i]),
                                      self.flux_err[i]),
                                method='l-bfgs-b') # try different minimizers, what is best for discrete

                if mini.success is True:
                    row = [self.flare_t0s[i][t], self.flare_amps[i][t]-1, mini.x[2], 
                           mini.x[3], mini.x[0]]
                    tab.add_row(row)

                model = flare_lightcurve(self.time[i], mini.x[0], int(mini.x[1]),
                                         mini.x[2], mini.x[3])[0] + np.nanmedian(self.flux[i])
                submodels.append(model)
            models.append(submodels)
            
        self.models = models
        self.parameters = tab


    def injection_recovery(self, n=100):
        """
        Completes injection & recovery for each light curve.

        Parameters
        ----------
        n : int, optional
             The number of flares you wish to inject. Default is 100.
        """

        for i in range(len(self.flux)):
            p = []
            print("Injecting flares into flare {}".format(i))
            q = self.labels[i][:,1] < self.prob_accept

            interpolation = interpolate.interp1d(self.time[i][q], self.flux[i][q])
            cleaned_flux  = interpolation(self.time[i])

            t0s, amps, rises, decays = flare_parameters(n,
                                                        len(self.time[i]),
                                                        self.nn.slc.cadences,
                                                        [0.001, 0.015],
                                                        [0.0001, 0.002],
                                                        [0.0001, 0.008])

            for j in range(len(t0s)):
                m = flare_lightcurve(self.time[i],
                                     np.abs(amps[j]),
                                     t0s[j],
                                     rises[j],
                                     decays[j])[0]

                preds = self.nn.predict(self.time[i], cleaned_flux+m,
                                        self.flux_err[i], injection=True)
                p.append(preds)                   

        return p, cleaned_flux+m, cleaned_flux, m
