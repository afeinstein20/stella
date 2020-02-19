import numpy as np
from tqdm import tqdm
from astropy.table import Table, Column
from astropy.timeseries import LombScargle
from scipy.optimize import minimize, curve_fit

__all__ = ['MeasureRotations']

class MeasureRotations(object):
    """
    Used for measuring rotation periods.
    """
    
    def __init__(self, IDs, time, flux, flux_err):
        """
        Takes in light curve identifiers, time, flux, 
        and flux errors.
        """
        self.IDs  = IDs
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

        self.run_LS()


    def gauss_curve(self, x, std, scale, mu):
        """ Fits a Gaussian to the peak of the LS
            periodogram.

        Parameters
        ----------
        x : np.array
        std : float
             Standard deviation of gaussian.
        scale : float
             Scaling for gaussian.
        mu : float
             Mean to fit around.

        Returns
        -------
        Gaussian curve.
        """
        term1 = 1.0 / (std * np.sqrt(2 * np.pi) )
        term2 = np.exp(-0.5 * ((x-mu)/std)**2)
        return term1 * term2 * scale


    def chiSquare(self, var, mu, x, y, yerr):
        """ Calculates chi-square for fitting a Gaussian
            to the peak of the LS periodogram.

        Parameters
        ----------
        var : list
             Variables to fit (std and scale for Gaussian curve).
        mu : float
             Mean to fit around.
        x : np.array
        y : np.array
        yerr : np.array

        Returns
        -------
        chi-square value.
        """
        m = self.gauss(x, var[0], var[1], mu)
        return np.sum( (y-m)**2 / yerr**2 )

    
    def fit_LS_peak(self, period, power, arg):
        """ Fits the LS periodogram at the peak power. 

        Parameters
        ----------
        period : np.array
             Array of periods from Lomb Scargle routine.
        power : np.array
             Array of powers from the Lomb Scargle routine.
        arg : int
             Argmax of the power in the periodogram.

        Returns
        -------
        popt : np.array
             Array of best fit values for Gaussian fit.
        """
        def fitting_routine():
            popt, pcov = curve_fit(self.gauss_curve, period[m], power[m],
                                   p0 = [(np.nanmax(period[subm]) - np.nanmin(period[subm]))/2.0,
                                         0.02,
                                         period[arg]],
                                   maxfev = 5000)
            return popt

        m = np.arange(arg-40, arg+40, 1, dtype=int)
        subm = np.arange(arg-20, arg+20, 1, dtype=int)

        try:
            popt = fitting_routine()
        except RuntimeError:
            popt = np.full(3, np.nan)

        # TRIES TO READJUST FITTING WINDOW IF RANGE IS LARGER THAN PERIOD ARRAY
        except IndexError:
            if np.min(m) <= 0:
                m = np.arange(0,arg+40,1,dtype=int)
                subm = np.arange(0,arg+20,1, dtype=int)
            elif np.max(m) > len(period):
                diff = np.max(m) - len(period)
                m = np.arange(arg-40-diff, len(period)-diff, 1, dtype=int)
                subm = np.arange(arg-20-diff, len(period)-diff-20, 1, dtype=int)
            popt = fitting_routine()

        return popt

    
    def run_LS(self, minf=1/20., maxf=1/0.1, spp=50):
        """ Runs LS fit for each light curve. 

        Parameters
        ----------
        minf : float, optional
             The minimum frequency to search in the LS routine. Default = 1/20.
        maxf : float, optional
             The maximum frequency to search in the LS routine. Default = 1/0.1.
        spp : int, optional
             The number of samples per peak. Default = 50.

        Attributes
        ----------
        LS_results : astropy.table.Table
        """
        tab = Table()

        periods = np.zeros(len(self.time))
        stds = np.zeros(len(self.time))
        peak_power = np.zeros(len(self.time))

        periods2 = np.zeros(len(self.time))
        stds2 = np.zeros(len(self.time))
        peak_power2 = np.zeros(len(self.time))


        for i in tqdm(range(len(self.flux))):

            time, flux, flux_err = self.time[i], self.flux[i], self.flux_err[i]
            
            freq, power = LombScargle(time, flux, dy=flux_err).autopower(minimum_frequency=minf,
                                                                         maximum_frequency=maxf,
                                                                         method='fast',
                                                                         samples_per_peak=spp)
            arg = np.argmax(power)
            
            period = 1.0/freq

            popt = self.fit_LS_peak(period, power, arg)
            
            periods[i] = period[arg]
            stds[i] = popt[0]
            peak_power[i] = power[arg]

            ## SEARCHES & MASKS RESONANCES OF THE BEST-FIT PERIOD
            perlist = period[arg] * np.array([0.5, 1.0, 2.0, 4.0, 8.0])
            remove_res = np.zeros(len(period))
            for p in perlist:
                where = np.where( (period >= p-0.01) & (period <= p+0.01))[0]
                if len(where) > 0:
                    ind = int(np.nanmedian(where))
                    remove_res[int(ind-spp):int(ind+spp)] = 1

            rr = remove_res == 0

            arg2 = np.argmax(power[rr])

            ## REDOS PERIOD ROUTINE FOR SECOND HIGHEST PEAK
            popt2 = self.fit_LS_peak(period[rr], power[rr], arg2)
            
            periods2[i] = period[rr][arg2]
            stds2[i] = popt2[0]
            peak_power2[i] = power[rr][arg2]

        tab.add_column(Column(self.IDs, 'Target_ID'))
        tab.add_column(Column(periods, name='period_days'))
        tab.add_column(Column(stds, name='gauss_width'))
        tab.add_column(Column(peak_power, name='max_power'))
        tab.add_column(Column(stds2, name='secondary_gauss_width'))
        tab.add_column(Column(periods2, name='secondary_period_days'))
        tab.add_column(Column(peak_power2, name='secondary_max_power'))

        tab = self.averaged_per_sector(tab)

        self.LS_results = tab
            
            
    def averaged_per_sector(self, tab):
        """ Looks at targets observed in different sectors and determines
            which period measured is likely the best period. Adds a column
            to MeasureRotations.LS_results of 'true_period_days' for the 
            results.
        """
        def assign_flag(per, pow, width, avg, secpow):
            """ Assigns a flag in the table for which periods are reliable.
            """
            if ((pow > 0.02) and (width < (per*0.25)) and 
                (avg > 0) and (secpow < (0.95*pow))):
                return 1
            else:
                return 0

        averaged_periods = np.zeros(len(tab))
        flagging = np.zeros(len(tab), dtype=int)

        for i in tqdm(np.unique(self.IDs)):
            
            subind = np.where(tab['Target_ID'] == i)[0]

            # IF ONLY OBSERVED IN 1 SECTOR
            if len(subind) == 1:
                averaged_periods[subind[0]] = tab['period_days'].data[subind]
                flagging[subind[0]] = assign_flag(tab['period_days'].data[subind],
                                                  tab['max_power'].data[subind],
                                                  tab['gauss_width'].data[subind],
                                                  tab['period_days'].data[subind],
                                                  tab['secondary_max_power'].data[subind])
                
            # IF OBSERVED IN MULTIPLE SECTORS
            elif len(subind) > 1:
                periods = tab['period_days'].data[subind]
                med_period = np.nanmedian(periods)
                std = np.nanstd(periods) * 2.0
                
                for si, p in enumerate(periods):
                    max_power = tab['max_power'].data[subind[si]]
                    width = tab['gauss_width'].data[subind[si]]
                    sec_power = tab['secondary_max_power'].data[subind[si]]
                    
                    if (p >= med_period - std) and (p <= med_period + std):
                        avg = p

                    # CHECKS TO SEE IF TWICE THE PERIOD IS A BETTER FIT
                    elif (p*2.0 >= med_period-std) and (p*2.0 <= med_period + std):
                        avg = p * 2.0

                    # CHECKS TO SEE IF HALF THE PERIOD IS A BETTER FIT
                    elif (p/2.0 >= med_period-std) and (p/2.0 <= med_period + std):
                        avg = p / 2.0

                    else:
                        avg = 0.0

                    averaged_periods[subind[si]] = avg
                    flagging[subind[si]] = assign_flag(p, max_power, width,
                                                       avg, sec_power)

        tab.add_column(Column(averaged_periods, name='avg_period_days'))
        tab.add_column(Column(flagging, name='flags'))
        return tab
