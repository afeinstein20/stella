import numpy as np
from scipy.stats import binned_statistic

def flare_lightcurve(time, amp, t0, rise, fall, y=None, binned=False,
                     bins=16, statistic='mean'):
    """
    Generates a simple flare model with a Gaussian rise and an 
    exponential decay.

    Parameters
    ----------
    time : np.ndarray
         A time array.
    amp : float
         The amplitude of the flare.
    t0 : int
         The index in the time array where the flare will occur.
    rise : float
         The Gaussian rise of the flare.
    fall : float
         The exponential decay of the flare.
    y : np.ndarray, optional
         Underlying stellar activity. Default if None.
    binned : bool, optional
         Bins the flare to a lower resolution. Default is False.
    bins : int, optional
         The number of bins if binned = True. Default is 16.
    statistic : str, optional
         The metric on which to bin the data. Default is True. 
         For more statistic options, see 
         https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html.

    Returns
    ----------
    flare_model : np.ndarray
         A light curve of zeros with an injected flare of given parameters
    row : np.ndarray
         The parameters of the injected flare. Returns - 
         [t0, amplitude, duration, gauss_rise, exp_decay].
    """
    def gauss_rise(time, flux, amp, t0, rise):
        return amp * np.exp( -(time - t0)**2.0 / (2.0*rise**2.0) ) + flux
    
    def exp_decay(time, flux, amp, t0, fall):
        return amp * np.exp( -(time - t0) / fall ) + flux

    growth = np.where(time <= time[t0])[0]
    decay  = np.where(time >  time[t0])[0]

    if y is None:
        y = np.zeros(len(time))

    growth_model = gauss_rise(time[growth], y[growth], amp, time[t0], rise)
    decay_model  = exp_decay(time[decay]  , y[decay] , amp, time[t0], fall)

    model = np.append(growth_model, decay_model)
    dur = np.abs(np.sum(model[:-1] * np.diff(time) ))

    if binned is False:
        return model, np.array([time[t0], amp, dur, rise, fall])

    else:
        mid = int(len(time)/2)
        bin_means = binned_statistic(time, model,
                                     bins=bins, statistic=statistic)[0]
        peak = np.argmax(bin_means)
        flux = np.zeros(len(time))
        flux[mid-peak:mid+bins-peak] = bin_means
        return flux, np.array([time[mid], bin_means[peak], dur, rise, fall])


def flare_parameters(size, cadences, amps, rises):
    """
    Generates an array of random amplitudes at different times with
    different rise and decay properties.

    Parameters
    ----------
    size : int
         The number of flares to generate.
    cadences : int
         The number of cadences to scroll over.
    amps : list
         List of minimum and maximum of flare amplitudes to draw from a 
         normal distribution. 
    rises : list
         List of minimum and maximum Gaussian rise rate to draw from
         a uniform distribution. 
    
    Returns
    ----------
    flare_t0s : np.ndarray
         The distribution of flare start times.
    flare_amps : np.ndarray
         The distribution of flare amplitudes.
    flare_rises : np.ndarray
         The distribution of flare rise rates.
    flare_decays : np.ndarray
         The distribution of flare decays rates.
    """
    flare_t0s   = np.full(size, cadences/2)
    flare_amps  = np.random.uniform(amps[0], amps[1], size)
    flare_rises = np.random.uniform(rises[0],  rises[1],  size)

    # Relation between amplitude and decay time
#    flare_decays = 0.07*flare_amps + 0.08*flare_amps**2
    flare_decays = np.random.uniform(0.005, 0.018, size)

    return flare_t0s, flare_amps, flare_rises, flare_decays
