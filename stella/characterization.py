import numpy as np
from tqdm import tqdm
from astropy.table import Table, Row


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

        try:
            time.shape[1]
            self.time     = time
            self.flux     = flux
            self.labels   = labels
            self.flux_err = flux_err
        except:
            self.time     = np.array([time])
            self.flux     = np.array([flux])
            self.labels   = np.array([labels])
            self.flux_err = np.array([flux_err])

    def flares(self, prob_accept=0.75):
        """
        Uses labels to identify flares and recover the following parameters:
        equivalent duration, amplitude, time of flare peak

        Parameters
        ----------
        prob_accept : float, optional
             The label probability for which to find flares
             above this threshold. Default = 0.75.

        Attributes
        ----------
        flare_table : astropy.Table
             A table of flare parameters.
        """
        t = Table(names=["ed", "ed_err", "amp", "amp_err", "Peak [time]"])

        for i in tqdm(range(len(self.flux))):
            peak = np.argmax(self.flux[i])
            if self.labels[i][:,1][peak] > prob_accept:
                t0  = self.time[i][peak]

                amp = np.nanmax(self.flux[i])
                random_flux = np.zeros((len(self.flux[i]),200))
                for j in range(len(self.flux[i])):
                    random_flux[i] = np.random.normal(self.flux[i][j], self.flux_err[i][j], 200)
                amp_err = np.std(np.nanmax(random_flux, axis=1))

                dur     = np.abs(np.sum(self.flux[i][:-1]    * np.diff(self.time[i]) ))
                dur_err = np.abs(np.sum(self.flux_err[i][:-1]* np.diff(self.time[i]) ))

                t.add_row([dur, dur_err, amp, amp_err, t0])

        self.flare_table = t


