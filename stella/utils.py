import wotan
import batman
import numpy as np
from scipy.stats import binned_statistic

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
    randtimes   = np.random.randint(cut_ends, len(time)-cut_ends, size)
    flare_amps  = np.random.uniform(amps[0], amps[1], size)
    flare_rises = np.random.uniform(0.00005,  0.0002,  size)

    # Relation between amplitude and decay time
    flare_decays = np.random.uniform(0.0003, 0.001, size)

    return randtimes, flare_amps, flare_rises, flare_decays


def batman_model(time, p):
    """
    Creates a batman transit model to inject into the data
    as not-a-flare noise.

    Parameters
    ----------    
    time : np.ndarray
    params : parameters for the batman model. params is an 
         array of [t0, period, rp/r_star, a/r_star].

    Returns
    ----------    
    flux : batman modeled flux with transit injected.
    """
    params = batman.TransitParams()
    params.t0 = p[0]
    params.per = p[1]
    params.rp = p[2]
    params.a = p[3]
    params.inc = 90.
    params.ecc = 0.
    params.w = 90.                       
    params.u = [0.1, 0.3]                
    params.limb_dark = "quadratic" 
    
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)  
    return flux
