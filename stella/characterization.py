import numpy as np
from tqdm import tqdm
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.optimize import minimize
from astropy.stats import sigma_clip
from lightkurve.lightcurve import LightCurve as LC

from .utils import *

__all__ = ['FlareCharacterization']


class FlareCharacterization(object):
    """
    A class that classifies flares in a given data set.
    """

    def __init__(self, nn, prob_accept=0.85):
        """
        Uses information predicted by the neural network
        to identify and characterize flares in the data.

        Parameters
        ----------
        nn : stella.ConvNN
        prob_accept : float, optional
             Probability threshold for accepting as a 
             potential flare. Default is 0.85.

        Attributes
        ----------
        time : stella.ConvNN.time_data
        flux : stella.ConvNN.flux_data
        prob_accept : float
        predictions : stella.ConvNN.predictions
        """

        self.time = nn.time_data
        self.flux = nn.flux_data
        self.cadences = nn.image_fmt[0]
        self.prob_accept = prob_accept
        self.predictions = nn.predictions

        self.find_flares()


    def find_flares(self):
        """
        Loops through light curves and finds groupings of
        points above the prob_accept value. 

        Attributes
        ----------
        flare_table : astropy.table.Table
        """
        
        def group_sequence(lst):
            """
            Identifies groupings of data points that constitute
            one flare. Any point within 3 of another with a high
            probability is grouped together. This does not affect
            characterizing nearby flares.
            
            Returns
            ----------
            res : np.ndarray
                 The groups for a given list.
            """
            res = [[lst[0]]]
            for i in range(1, len(lst)):
                if np.abs(lst[i-1] - lst[i]) <= 3:
                    res[-1].append(lst[i])
                else:
                    res.append([lst[i]])
            return res

        flare_t0s = []
        flare_flux = []

        for i in tqdm(range(len(self.time))):
            q = self.predictions[i][:,1] > self.prob_accept
            inds = np.where(q==True)[0]
            
            t0s = np.array([])
            flare_fluxes  = []
            flare_detrend = []
            flare_times   = []

            if len(inds) > 0:
                grp  = group_sequence(inds)
            
                for g in grp:
                    if len(g) > 2:
                        padding = 50#int((self.cadences - len(g))/2)
                        g = np.append(np.arange(g[0]-padding, g[0], 1, dtype=int), g)
                        g = np.append(g, np.arange(g[-1], g[-1]+padding, 1, dtype=int))

                        # Local detrending
                        poly  = np.polyfit(self.time[i][g], self.flux[i][g], 6)
                        fit   = np.poly1d(poly)
                        model = fit(self.time[i][g])
                        detrended_flux = self.flux[i][g]/model

                        lk, trend = LC(self.time[i][g], self.flux[i][g]).flatten(window_length=21,
                                                                                 return_trend=True)
                        detrended_flux = self.flux[i][g]/trend.flux

                        med = np.nanmedian(detrended_flux)
                        std = np.nanstd(detrended_flux)

                        # Flare signal finding
                        signal, _ = find_peaks(detrended_flux,
                                               height=(med+1.0*std, med+100*std))

                        for peak in signal:
                            # Makes sure the peak is at least 1.5 sigma above the local noise
                            if detrended_flux[peak] > (med + 1.3*std):
                                diff_pre  = detrended_flux[peak] - detrended_flux[int(peak-5):peak]
                                diff_post = detrended_flux[peak] - detrended_flux[int(peak+1):int(peak+5)]

                                # Makes sure the peak is greater than the points before it
                                # Makes sure the peak is greater than the next 2 data points
                                if (len(np.where(diff_post > 0)[0]) >= 3) and (len(np.where(diff_pre > 0)[0]) >= 2):

                                    flare_fluxes.append(self.flux[i][g])
                                    flare_times.append(self.time[i][g])
                                    flare_detrend.append(detrended_flux)
                                    t0s = np.append(t0s, self.time[i][g][peak])

            flare_t0s.append(np.unique(t0s))

        self.flare_t0s = np.array(flare_t0s)
        self.flare_fluxes = np.array(flare_fluxes)
        self.flare_times  = np.array(flare_times)
        self.flare_detrend = np.array(flare_detrend)


    def model_flares(self):
        """
        Models the stellar activity + flare in order to extract the
        most accurate flare parameters.

        Attributes
        ---------- 
        flare_table : astropy.table.Table
             Parameters of identified and fitted flares.
        """
        
        
        def chiSquare(var, x, y, yerr):
            # Fits the flare and an underlying polynomial
            # to mimic stellar activity and find flare params.
            amp, t0, rise, fall = var
            
            sig_mask = sigma_clip(y, sigma=1.5).mask
            
            fit = np.polyfit(x[~sig_mask], y[~sig_mask], deg=6)
            model = np.poly1d(poly_fit)

            m = flare_lightcurve(x, amp, int(t0), rise, fall,
                                 y=model(x))[0] + np.nanmedian(y)

            return np.sum( (y-m)**2/yerr**2 )
            

        
