import os
import warnings
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
        output_dir : str, optional
             Path to location where to save the simulated flares.
             Default = '~/.stella/training_set/'.'

        Attributes
        ----------
        sample_size : int
             The number of light curves to generate.
        time : np.ndarray
             An array of time.
        output_dir : str
             Path to location where to save the simulated light curves to.
        """

        self.sample_size = sample_size
        self.time = np.arange(0,8000,1)

        if output_dir is None:
            self.output_dir = self.fetch_dir()
        else:
            self.output_dir = output_dir



    def flare_model(self, amp, t0, rise, fall,
                    uptime=10, statistic='mean'):
        """
        Generates a simple flare model with a Gaussian rise
        and an exponential decay.

        Parameters
        ----------
        amp : float
             The amplitude of the flare.
        t0 : int
             The index in the time array where the flare will occur.
        rise : float
             The Gaussian rise of the flare.
        decay : float
             The exponential decay of the flare.
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
        dur : float
             The duration of the flare.
        """
        
        def gauss_rise(time, amp, t0, rise):
            return amp * np.exp( -(time - t0)**2.0 / (2.0*rise**2.0) )
        
        def exp_decay(time, amp, t0, decay):
            return amp * np.exp( -(time - t0) / decay )


        dt = np.nanmedian(np.diff(self.time))
        timeup = np.linspace(np.nanmin(self.time) - dt,
                             np.nanmax(self.time) + dt,
                             self.time.size*uptime)

        up_t0 = timeup[ np.where(timeup >= self.time[t0])[0][0] ]
        growth  = np.where(timeup <= up_t0)[0]
        decay   = np.where(timeup >  up_t0)[0]

        rise_model = gauss_rise(timeup[growth], amp, up_t0, rise)
        fall_model = exp_decay( timeup[decay],  amp, up_t0 , fall)

        model = np.append(rise_model, fall_model)
        flare =  binned_statistic(timeup, model, statistic=statistic,
                                  bins=len(self.time))[0]
        dur = len(np.where(flare != 0.0)[0])
        return flare, dur


    def sine_wave(self, amplitude=[0.02,0.15], frequency=[300,10000],
                  noise=[0.02, 0.045]):
        """
        Creates a sine wave to simulate stellar activity
        with Gaussian noise.

        Parameters
        ----------
        amplitude : np.ndarray, optional
             List of minimum and maximum amplitude to use for a uniform
             distribution to draw from.
             Default = [0.02, 0.15].
        frequency : np.ndarray, optional
             List of minimum and maximum frequencies to use for a uniform
             distribution to draw from.
             Default = [300, 10000].
        noise : np.ndarray, optional
             List of minimum and maximum noise levels to use for a uniform
             distribution to draw from.
             Default = [0.01, 0.04].

        Attributes
        ----------
        fluxes : np.ndarray
             An array of simulated fluxes of shape (sample_size, len(time)).
        """

        amplitude = np.random.uniform(amplitude[0], amplitude[1], self.sample_size)
        frequency = 1.0/np.random.uniform(frequency[0], frequency[1], self.sample_size)
        phase     = np.random.uniform(0, 2*np.pi, self.sample_size)

        def model(time, amp, freq, phase):
            return amp * np.sin( 2*np.pi*freq*time + phase )

        fluxes = np.zeros( (self.sample_size, len(self.time) ))

        for i in range(self.sample_size):
            noise_lvl = np.random.uniform(noise[0], noise[1], 1)
            noise     = np.random.normal(0, np.abs(noise_lvl), len(self.time))
            fluxes[i] = model(self.time, amplitude[i], frequency[i], phase[i]) + noise

        self.fluxes = fluxes
        

    def inject_flares(self, number_per=[0,20],
                     amplitudes=[1.0,0.1], decays=[0.05,0.15],
                     rises=[0.001,0.006], window_length=101):
        """
        Injects flares of given parameters into a light curve.

        Parameters
        ----------
        number_per : np.ndarray, optional
             List of minimum and maximum number of flares to inject into a 
             single light curve.
             Default = [0, 20].
        amplitudes : list, optional
             List of mean and std of flare amplitudes to draw from a 
             normal distritbution. Default = [1.0, 0.1].
        decays : list, optional
             List of minimum and maximum exponential decay rate to draw from a 
             uniform distribution. Default = [0.05, 0.15].
        rises : list, optional
             List of minimum and maximum Gaussian rise rate to draw from a 
             uniform distribution. Default = [0.001, 0.006]
        window_length : int, optional
             The window length to use in the Savitsky-Golay filter to detrend
             simulated light curves. Default = 101.

        Attributes
        ----------
        total_flares : int
             The total number of injected flares.
        flare_fluxes : np.ndarray
             An array of simulated fluxes with injected flares.
        flare_fluxes_detrended : np.ndarray
             An array of detrended simulated fluxes with injected flares.
        flare_durs : np.ndarray
             An array of the durations of the injected flares.
        labels : np.ndarray
             An array of labels for each flare-injected flux.
        """

        number_per = np.random.randint(number_per[0], number_per[1], self.sample_size)
        self.total_flares = np.sum(number_per)

        self.flare_amps  = np.random.normal(amplitudes[0], amplitudes[1], self.total_flares)
        self.flare_decays= np.random.uniform(decays[0]   , decays[1]    , self.total_flares)
        self.flare_rises = np.random.uniform(rises[0]    , rises[1]     , self.total_flares)
        self.flare_t0s   = np.random.randint(0, len(self.time)          , self.total_flares)

        flare_fluxes = np.zeros( (self.sample_size, len(self.time) ))
        flare_fluxes_detrended = np.zeros( (self.sample_size, len(self.time) ))
        labels = np.zeros( (self.sample_size, len(self.time) ), dtype=int)

        durations = np.zeros(self.total_flares)

        loc = 0
        for i in tqdm(range(self.sample_size)):
            # Loops through each injected flare per light curve
            if number_per[i] == 0:
                flare_fluxes[i] = self.fluxes[i]
                
            else:
                for n in range(number_per[i]):
                    flare, dur = self.flare_model(self.flare_amps[loc],
                                                  self.flare_t0s[loc],
                                                  self.flare_rises[loc],
                                                  self.flare_decays[loc])
                    durations[loc] = dur

                    if n == 0:
                        flare_flux = self.fluxes[i] + flare
                    else:
                        flare_flux += flare

                    loc += 1

            q = flare_flux > (np.nanmedian(flare_flux)+(0.001*np.std(flare_flux)) )

            labels[i] = q*1
            flare_fluxes[i] = flare_flux + 1
            flare_fluxes_detrended[i] = LC(self.time, flare_fluxes[i]).flatten(window_length=
                                                                              window_length).flux
            

        self.flare_fluxes = flare_fluxes 
        self.labels       = labels
        self.flare_durs   = durations
        self.flare_fluxes_detrended = flare_fluxes_detrended


    def save(self, output_fn_format='sim{0:04d}.npy'):
        """
        A function that allows the user to save the simulated light curves.
        Saves as .npy files in the given output directory.

        Parameters
        ------- 
        output_fn_format : str, optional
             The naming convention used for saving the simulated light curves to.
             Default = 'sim{0:04d}.npy'.format(simulation_number).
        """

        self.output_fn_format = output_fn_format

        for i in tqdm(range(len(self.flare_fluxes))):
            path = os.path.join(self.output_dir, self.output_fn_format.format(i))
            data = [self.time, self.flare_fluxes[i], self.flare_fluxes_detrended[i], self.labels[i]]
            np.save(path, data)


    def fetch_dir(self):
        """
        Returns the default path to the directory where files will be saved
        or loaded.
        By default, this method will return "~/.stella" and create
        this directory if it does not exist.  If the directory cannot be
        access or created, then it returns the local directory (".").

        Attributes
        -------
        output_dir : str
            Path to location of where simulated light curves will be saved to.
        """

        output_dir    = os.path.join(os.path.expanduser('~'), '.stella/training_set')
        if os.path.isdir(output_dir):
            return output_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(output_dir)
            # downloads locally if OS error occurs
            except OSError:
                output_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Saving simulated light curves to '
                              'working directory instead.'.format(output_dir))

        self.output_dir = output_dir
        
