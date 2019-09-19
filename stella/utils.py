import numpy as np

def flare_lightcurve(time, amp, t0, rise, fall):
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

    Returns
    ----------
    flare_model : np.ndarray
         A light curve of zeros with an injected flare of given parameters
    row : np.ndarray
         The parameters of the injected flare. Returns - 
         [t0, amplitude, duration, gauss_rise, exp_decay].
    """
    def gauss_rise(time, amp, t0, rise):
        return amp * np.exp( -(time - t0)**2.0 / (2.0*rise**2.0) )
    
    def exp_decay(time, amp, t0, fall):
        return amp * np.exp( -(time - t0) / fall )

    growth = np.where(time <= time[t0])[0]
    decay  = np.where(time >  time[t0])[0]

    growth_model = gauss_rise(time[growth], amp, time[t0], rise)
    decay_model  = exp_decay(time[decay]  , amp, time[t0], fall)

    model = np.append(growth_model, decay_model)
    dur = np.abs(np.sum(model[:-1] * np.diff(time) ))

    return model, np.array([time[t0], amp, dur, rise, fall])


def flare_parameters(size, time_length, cadences,
                     amps, rises, decays):
    """
    Generates an array of random amplitudes at different times with
    different rise and decay properties.

    Parameters
    ----------
    size : int
         The number of flares to generate.
    time_length : int
         The size of the time array.
    cadences : int
         The number of cadences to scroll over.
    amps : list
         List of mean and stf of flare amplitudes to draw from a 
         normal distribution. 
    rises : list
         List of minimum and maximum Gaussian rise rate to draw from
         a uniform distribution. 
    decays : list
         List of minimum and maximum exponential decay rate to draw
         from a normal distribution.
    
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

    flare_t0s   = np.random.randint(cadences/2, 
                                    time_length-cadences/2,
                                    size)
    flare_amps  = np.random.normal(amps[0],    amps[1],   size)
    flare_decays= np.random.uniform(decays[0], decays[1], size)
    flare_rises = np.random.uniform(rises[0],  rises[1],  size)

    return flare_t0s, flare_amps, flare_rises, flare_decays
