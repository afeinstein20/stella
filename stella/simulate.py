import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic
from lightkurve.lightcurve import LightCurve as LC

__all__ = ['SimulateLightCurves']

class SimulateLightCurves(object):
    """
    A class to create simulated flares for neural
    network training.
    """

    def __init__(self, sample_size=8000, output_dir=None):
        """
        Parameters
        ----------
        sample_size : int, optional
             The number of light curves to generate.
             Default = 8000.
        output_dir : path, optional
             The path where to save the simulated flares.
             Default = '~/.stella/training_set/'.'

        Attributes
        ----------
        sample_size : int
             The number of light curves to generate.
        time : np.ndarray
             An array of time.
        """

        self.sample_size = sample_size
        self.time = np.arange(0,8000,1)


    def flare_model(self, uptime=10, statistic='mean'):
        """
        Generates a simple flare model with a Gaussian rise
        and an exponential decay.

        Parameters
        ----------
        uptime : int, optional
             The number of points to bin. Default = 10.
        statistic : str, optional
             How to bin the points. Default = 'mean'.
             For more options, see 
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html.

        Returns
        ----------
        flare model : np.ndarray
             A light curve of 0s with an injected flare of given parameters.
        """
        
        def gauss_rise(time, amp, t0, rise):
            return amp * np.exp( -(time - t0)**2.0 / (2.0*rise**2.0) )
        
        def exp_decay(time, amp, t0, decay):
            return amp * np.exp( -(time - t0) / decay )


        dt = np.nanmedian(np.diff(self.time))
        timeup = np.linspace(np.nanmin(self.time) - dt,
                             np.nanmax(self.time) + dt,
                             self.time.size*uptime)

        up_t0 = timeup[ np.where(timeip >= self.time[t0])[0][0] ]
        rise  = np.where(timeup <= up_t0)[0]
        fall  = np.where(timeup >  up_t0)[0]

        rise_model = gauss_rise(timeup[rise], amp, up_t0, growth_factor)
        fall_model = exp_decay( timeup[fall], amp, up_t0, decay_factory)

        model = np.append(rise_model, fall_model)
        
        return binned_statistic(timeup, model, statistic=statistic,
                                bins=len(self.time))[0]


    def sine_wave(self, amplitude=np.random.uniform(0.02, 0.15, self.sample_size), 
                  frequency=1.0/np.random.uniform(3456, 21600, self.sample_size), 
                  phase=np.random.uniform(0, 2*np.pi, self.sample_size),
                  noise=[0.01, 0.04]):
        """
        Creates a sine wave to simulate stellar activity
        with Gaussian noise.

        Parameters
        ----------
        amplitude : np.ndarray, optional
             An array of different amplitudes for each light curve.
             Default = np.random.uniform(0.02, 0.15, sample_size).
        frequency : np.ndarray, optional
             An array of frequencies for each light curve.
             Default = 1 / np.random.uniform(3456, 21600, sample_size).
        phase : np.ndarray, optional
             An array of phases for each light curve.
             Default = np.random.uniform(0, 2*np.pi, sample_size).
        noise : np.ndarray, optional
             List of minimum and maximum noise levels to use for a uniform
             distribution to draw from.
             Default = [0.01, 0.04].

        Attributes
        ----------
        fluxes : np.ndarray
             An array of simulated fluxes of shape (sample_size, len(time)).
        """

        def model(time, amp, freq, phase):
            return amp * np.sin( 2*np.pi*freq*time + phase )

        fluxes = np.zeros( (self.sample_size, self.time) )

        for i in range(self.sample_size):
            noise_lvl = np.random.uniform(noise[0], noise[1], 1)
            noise     = np.random.normal(0, noise_lvl, len(self.time))
            fluxes[i] = model(self.time, amplitude[i], frequency[i], phase[i]) + noise

        self.fluxes = fluxes
        

    def inject_flare(self):
