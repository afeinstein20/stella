import numpy as np
from tqdm import tqdm
from astropy import units as u
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from astropy.table import Table, Column
from astropy.timeseries import LombScargle

__all__ = ['FindTheSpots']

class FindTheSpots(object):
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

    
    def run_LS(self, minf=1/12.5, maxf=1/0.1, spp=50):
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

        periods = np.zeros(len(self.IDs))
        stds = np.zeros(len(self.IDs))
        peak_power = np.zeros(len(self.IDs))

        periods2 = np.zeros(len(self.IDs))
        stds2 = np.zeros(len(self.IDs))
        peak_power2 = np.zeros(len(self.IDs))


        for i in tqdm(range(len(self.flux)), desc="Finding most likely periods"):

            time, flux, flux_err = self.time[i], self.flux[i], self.flux_err[i]

            ls = LombScargle(time, flux)
            freq, power = ls.autopower(minimum_frequency=minf,
                                       maximum_frequency=maxf,
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
            maskreg = spp/2
            for p in perlist:
                where = np.where( (period <= p))[0]
                if len(where) > 0:
                    ind = int(where[0])
                    if ind-maskreg > 0 and ind<len(period)-maskreg:
                        remove_res[int(ind-maskreg):int(ind+maskreg)] = 1
                    elif ind < maskreg:
                        remove_res[0:int(maskreg)] = 1
                    elif ind > len(period)-maskreg:
                        remove_res[int(len(period)-maskreg):len(period)] = 1
            
            if perlist[1] == 1/minf:
                remove_res[0:int(spp/2)] = 1

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

        Returns
        -------
        astropy.table.Table
        """
        def assign_flag(per, pow, width, avg, secpow):
            """ Assigns a flag in the table for which periods are reliable.
            """
            if ((pow > 0.02) and (width < (per*0.35)) and 
                (avg > 0) and (np.abs(per-avg) < 1.0) and
                ((secpow < (0.96*pow)) or (secpow == pow)) and 
                (per < 12.5)):
                return 1
            else:
                return 0

        averaged_periods = np.zeros(len(tab))
        flagging = np.zeros(len(tab), dtype=int)

        for i in self.IDs:
            
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
                
                check_res = np.array([])
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

                    check_res = np.append(check_res, avg)
                averaged_periods[subind] = np.nanmedian(check_res)
                
                for si, p in enumerate(periods):
                    max_power = tab['max_power'].data[subind[si]]
                    width = tab['gauss_width'].data[subind[si]]
                    sec_power = tab['secondary_max_power'].data[subind[si]]
                    flagging[subind[si]] = assign_flag(p, max_power, width,
                                                       np.nanmedian(check_res), 
                                                       sec_power)

        tab.add_column(Column(averaged_periods, name='avg_period_days'))
        tab.add_column(Column(flagging, name='flags'))
        return tab


    def phase_lightcurve(self, table=None, trough=-0.5, peak=0.5, kernel_size=15):
        """ 
        Finds and creates a phase light curve that traces the spots.
        Uses only complete rotations and extrapolates outwards until the
        entire light curve is covered.

        Parameters
        ----------
        table : astropy.table.Table, optional
             Used for getting the periods of each light curve. Allows users
             to use already created tables. Default = None. Will search for 
             stella.FindTheSpots.LS_results.
        trough : float, optional
             Sets the phase value at the minimum. Default = -0.5.
        peak : float, optional
             Sets the phase value t the maximum. Default = 0.5.
        kernel_size : odd float, optional
             Sets kernel size for median filter smoothing. Default = 15.

        Attributes
        ----------
        phases : np.ndarray
        """
        def medfilt_phase(f, ks, phase_order, localmin):
            """ Finds minimum of filtered rotation curve. """
            mf = medfilt(f, kernel_size=25)
            minimum = np.argmin(mf[localmin:-localmin])
            p = np.append( np.linspace(phase_order[0], 0, minimum),
                           np.linspace(0, phase_order[1], len(f)-minimum) )
            return minimum, p

        if table is None:
            table = self.LS_results

        PHASES   = np.copy(self.flux)
        localmin = 20
        which_inds = []

        for i in tqdm(range(len(table['Target_ID'])), desc='Getting Phases'):
            prot = table['avg_period_days'].data[i]
            prot_cadences = int(((prot*u.day).to(u.min)/2).value)

            # MAKES SURE PROT PASSED PREVIOUS TESTS
            if table['flags'].data[i] == 1:

                which_inds.append(i)

                time = self.time[i]
                flux = self.flux[i]
                err  = self.flux_err[i]
                
                phase = np.zeros(len(flux))

                # CREATES LIST OF ITERATIONS OF ROTATION PERIODS AND FINDS
                # WHICH ONES ARE IN THE LIGHT CURVE
                start = np.argmax(flux[np.where(time<=time[0]+prot)[0]])
                rots  = np.arange(-1,200,1) * prot + time[start]
                inlc  = ((rots > time[0]) & (rots < time[-1]))
                rots  = rots[inlc]

                if len(rots) <= 1:
                    PHASES[i] = np.zeros(time.shape)
                    
                else:
                    start_ind = np.where( (time>=rots[0]) & (time<rots[1]))[0]
                    if ( (flux[int(np.nanmedian(start_ind)/2)] > flux[start_ind][0]) and
                         (flux[int(np.nanmedian(start_ind)/2)] > flux[start_ind][-1])):
                        phase_order = [trough, peak]
                    else:
                        phase_order = [peak, trough]

                    # FINDS ORBITAL BREAKS TO IGNORE
                    diff = np.diff(time)
                    orbit_break = np.where(diff > (np.nanmedian(diff) + 12*np.nanstd(diff)))[0]
                    lightcurve = np.diff( np.sort( np.append([0, len(time)], orbit_break)))

#                if len(np.where(prot_cadences*1.5-lightcurve>0)[0]) > 0:
#                    PHASES[i] = np.zeros(flux.shape)

#                else:
# FINDS WHICH PROT STARTS COMPLETE 1 FULL ROTATION
                    rfull, rall = np.array([]), np.array([])
                    for r in rots:
                        if len(time[((time>=r) & (time<=r+0.1))]) > 0:
                            rall = np.append(rall, r)
                            for o in orbit_break:
                                rall = np.append(rall, np.array([time[o], time[o+1]]))

                                # FOR FULL ROTATIONS
                                if ( ((r < time[o]-prot/5) or (r>time[o+1]+prot/5)) and
                                     (r<time[-1]-prot/5) and (r>time[0]+prot/5)):
                                    rfull = np.append(rfull, r)
                    rall = np.append(rall, np.array([time[0], time[-1]]))
                    rall, rfull = np.sort(np.unique(rall)), np.sort(np.unique(rfull))

                    if len(rfull) <= 1:
                        PHASES[i] = np.zeros(time.shape)
                    
                    else:
                        troughs, regions = np.array([], dtype=int), np.array([], dtype=int)

                        for r in range(len(rfull)):                    
                            region = np.where((time>=rfull[r]) & (time<rfull[r]+prot))[0]
                            if len(region) >= 50:
                                minimum, p = medfilt_phase(flux[region], kernel_size,
                                                           phase_order, localmin)
                                regions = np.append(regions, len(region))
                                troughs = np.append(troughs, minimum)
                                phase[region] = p
                        
                
                        # FINDS AN APPROXIMATE MINIMUM CADENCE
                        averaged_trough = int(np.round(np.nanmedian(troughs)/np.nanmedian(regions) * prot_cadences))
                        interpphase = np.append( np.linspace(phase_order[0], (trough+peak)/2, averaged_trough),
                                                 np.linspace((trough+peak)/2, phase_order[1], prot_cadences-averaged_trough))

                        # FINDS PHASES FOR NOT COMPLETE ROTATION PERIODS OR 
                        # FULL ONES TOO CLOSE TO LARGE GAPS
                        for r in range(len(rall)):
                            if r not in rfull and r != len(rall)-1:
                                region = np.where( (time >= rall[r]) & (time <= rall[r+1]))[0]
                            
                                if len(region) > prot_cadences:
                                    diff = len(region) - prot_cadences
                                    ip = np.zeros(len(region))
                                    ip[0:averaged_trough+diff] = np.linspace(phase_order[0], (trough+peak)/2,
                                                                             averaged_trough+diff)
                                    ip[averaged_trough+diff:]  = np.linspace((trough+peak)/2, phase_order[1],
                                                                             len(region)-averaged_trough-diff)
                                    phase[region] = ip
                                else:
                                    if flux[region][-1] > flux[region][0]:
                                        phase[region] = interpphase[prot_cadences-len(region):]
                                    else:
                                        phase[region] = interpphase[:len(region)]

                        PHASES[i] = phase
                
            else:
                PHASES[i] = np.zeros(self.flux[i].shape)

        self.phases = PHASES
