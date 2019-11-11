import numpy as np
from tqdm import tqdm
from astropy.table import Table
from scipy.signal import find_peaks


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

        for i in tqdm(range(len(self.time))):
            q = self.predictions[i][:,1] > 0.3
            inds = np.where(q==True)[0]
            
            t0s = np.array([])

            if len(inds) > 0:
                grp  = group_sequence(inds)
            
                for g in grp:
                    if len(g) > 2:
                        padding = int((self.cadences - len(g))/2)
                        g = np.append(np.arange(g[0]-padding, g[0], 1, dtype=int), g)
                        g = np.append(g, np.arange(g[-1], g[-1]+padding, 1, dtype=int))
                        
                        temp_peak = np.argmax(self.flux[i][g])
                        
                        if (temp_peak < len(g)-1) and (temp_peak >= 1):
                            # Makes sure the peak is at least 1.5 sigma above the local noise
                            if self.flux[i][g][temp_peak] > (np.nanmean(self.flux[i][g]) + 1.5*np.std(self.flux[i][g])):
                                diff_pre  = self.flux[i][g][temp_peak] - self.flux[i][g][int(temp_peak-4):temp_peak]
                                diff_post = np.diff(self.flux[i][g][temp_peak:int(temp_peak+4)])
                                # Makes sure the peak is greater than the points before it
                                # Makes sure the peak is greater than the next 2 data points
                                if (len(np.where(diff_post < 0)[0]) >= 2) and (len(np.where(diff_pre > 0)[0]) >= 2):
                                
                                    # Local detrending
                                    poly  = np.polyfit(self.time[i][g], self.flux[i][g], 3)
                                    fit   = np.poly1d(poly)
                                    model = fit(self.time[i][g])

                                    # Finding the actual flare, not just argmax
                                    med = np.nanmedian(self.flux[i][g]/model)
                                    std = np.nanstd(self.flux[i][g]/model)

                                    signal, _ = find_peaks(self.flux[i][g]/model,
                                                           height=(med+2.0*std, med+100*std))
                                    for sig in signal:
                                        t0s = np.append(t0s, self.time[i][g][sig])
                        
            flare_t0s.append(t0s)

        self.flare_t0s = np.array(flare_t0s)
        
