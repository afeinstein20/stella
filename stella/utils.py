import numpy as np
from scipy.interpolate import interp1d

def flare_lightcurve(time, t0, amp, rise, fall, y=None):
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

    return model, np.array([time[t0], amp, dur, rise, fall])


def flare_parameters(size, time, amps, cut_ends=30):
    """
    Generates an array of random amplitudes at different times with
    different rise and decay properties.

    Parameters
    ----------
    size : int
         The number of flares to generate.
    times : np.array
         Array of times where a random subset will be chosen for flare
         injection. 
    amps : list
         List of minimum and maximum of flare amplitudes to draw from a 
         normal distribution. 
    cut_ends : int, optional
         Number of cadences to cut from the ends of the light curve.
         Default is 30.
    
    Returns
    ----------
    flare_t0s : np.ndarray
         The distribution of flare start time indices.
    flare_amps : np.ndarray
         The distribution of flare amplitudes.
    flare_rises : np.ndarray
         The distribution of flare rise rates.
    flare_decays : np.ndarray
         The distribution of flare decays rates.
    """
    # CHOOSES UNIQUE TIMES FOR INJ-REC PURPOSES
    randtimes   = np.random.randint(cut_ends, len(time)-cut_ends, size*2)
    randtimes   = np.unique(randtimes)
    randind     = np.random.randint(0, len(randtimes), size)
    randtimes   = randtimes[randind]

    flare_amps  = np.random.uniform(amps[0], amps[1], size)
    flare_rises = np.random.uniform(0.00005,  0.0002,  size)

    # Relation between amplitude and decay time
    flare_decays = np.random.uniform(0.0003, 0.004, size)

    return randtimes, flare_amps, flare_rises, flare_decays


def fill_in(time, flux, flux_err, sigma=2.5):
    """
    Fills in any gaps in the data with the standard deviation of
    the light curve. Looks for differences in time greater than 
    some defined sigma threshold.

    Parameters
    ----------
    time : np.array
         Array of time from one light curve.
    flux : np.array
         Array of flux from one light curve.
    flux_err : np.array
         Array of flux errors from one light curve.
    sigma : float, optional
         The sigma-outlier difference to find time
         gaps. Default is 2.5.
    
    Returns
    -------
    time : np.array
    flux : np.array
    flux_err : np.array
    """
    t, f, e = np.array(time), np.array(flux), np.array(flux_err)

    diff = np.diff(t)
    diff_ind = np.where(diff >= (np.nanmedian(diff) + 
                                 sigma*np.nanstd(diff)) )[0]
    avg_noise = np.nanstd(f) / 2.0

    if len(diff_ind) > 0:
        for i in diff_ind:
            start = i
            stop  = int(i+2)
            
            func = interp1d(t[start:stop], f[start:stop])

            new_time = np.arange(t[start],
                                 t[int(start+1)],
                                 np.nanmean(diff))
            noise = np.random.normal(0, avg_noise, len(new_time))
            new_flux = func(new_time) + noise

            t = np.insert(t, i, new_time)
            f = np.insert(f, i, new_flux)
            e = np.insert(e, i, noise)

    t, f, e = zip(*sorted(zip(t,f,e)))

    return np.array(t), np.array(f), np.array(e)
    
                            
