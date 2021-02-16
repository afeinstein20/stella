import numpy as np
from tqdm import tqdm
import statistics as stats
from astropy import units as u
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from astropy.table import Table, Column
from astropy.timeseries import LombScargle

__all__ = ['MeasureProt']

class MeasureProt(object):
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

        if arg-40 < 0:
            start = 0
        else:
            start = arg-40
        if arg+40 > len(period):
            end = len(period)-1
        else:
            end = arg+40

        m = np.arange(start, end, 1, dtype=int)

        if arg-20 < 0:
            start = 0
        else:
            start = arg-20
        if arg + 20 > len(period):
            end = len(period)-1
        else:
            end = arg+20

        subm = np.arange(start, end, 1, dtype=int)

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
        def per_orbit(t, f):
            nonlocal maxf, spp

            minf = 1/(t[-1]-t[0])
            if minf > 1/12.0:
                minf = 1/12.0

            freq, power = LombScargle(t, f).autopower(minimum_frequency=minf,
                                                      maximum_frequency=maxf,
                                                      samples_per_peak=spp)
            arg = np.argmax(power)
            per = 1.0/freq
            popt = self.fit_LS_peak(per, power, arg)
            
            ## SEARCHES & MASKS RESONANCES OF THE BEST-FIT PERIOD
            perlist = per[arg] * np.array([0.5, 1.0, 2.0, 4.0, 8.0])
            remove_res = np.zeros(len(per))
            maskreg = int(spp/1.5)
            for p in perlist:
                where = np.where( (per <= p))[0]
                if len(where) > 0:
                    ind = int(where[0])
                    if ind-maskreg > 0 and ind<len(per)-maskreg:
                        remove_res[int(ind-maskreg):int(ind+maskreg)] = 1
                    elif ind < maskreg:
                        remove_res[0:int(maskreg)] = 1
                    elif ind > len(per)-maskreg:
                        remove_res[int(len(per)-maskreg):len(per)] = 1
            if perlist[1] == 1/minf:
                remove_res[0:int(spp/2)] = 1

            rr = remove_res == 0
            arg1 = np.argmax(power[rr])
            ## REDOS PERIOD ROUTINE FOR SECOND HIGHEST PEAK 
            if arg1 == len(per[rr]):
                arg1 = int(arg1-3)

            popt2 = self.fit_LS_peak(per[rr], power[rr], arg1)
            
            maxpower = power[arg]
            secpower = power[rr][arg1]

            bestperiod = per[arg]
            secbperiod = per[rr][arg1]

            bestwidth = popt[0]

            return bestperiod, secbperiod, maxpower, secpower, bestwidth

        tab = Table()

        periods = np.zeros(len(self.IDs))
        stds = np.zeros(len(self.IDs))
        peak_power = np.zeros(len(self.IDs))

        periods2 = np.zeros(len(self.IDs))
        peak_power2 = np.zeros(len(self.IDs))

        orbit_flag = np.zeros(len(self.IDs))
        orbit_flag1 = np.zeros(len(self.IDs))
        orbit_flag2 = np.zeros(len(self.IDs))

        for i in tqdm(range(len(self.flux)), desc="Finding most likely periods"):

            time, flux, flux_err = self.time[i], self.flux[i], self.flux_err[i]
            
            # SPLITS BY ORBIT
            diff = np.diff(time)
            brk = np.where(diff >= np.nanmedian(diff)+14*np.nanstd(diff))[0]
            
            if len(brk) > 1:
                brk_diff = brk - (len(time)/2)
                try:
                    brk_diff = np.where(brk_diff<0)[0][-1]
                except IndexError:
                    brk_diff = np.argmin(brk_diff)
                brk = np.array([brk[brk_diff]], dtype=int)

            # DEFINITELY TRIMS OUT EARTHSHINE MOFO
            t1, f1 = time[:brk[0]], flux[:brk[0]]#[300:-500], flux[:brk[0]]#[300:-500]
            t2, f2 = time[brk[0]:], flux[brk[0]:]#[800:-200], flux[brk[0]:]#[800:-200]

            o1_params = per_orbit(t1, f1)
            o2_params = per_orbit(t2, f2)

            both = np.array([o1_params[0], o2_params[0]])
            avg_period = np.nanmedian(both)


            flag1 = self.assign_flag(o1_params[0], o1_params[2], o1_params[-1],
                                    avg_period, o1_params[-2], t1[-1]-t1[0])
            flag2 = self.assign_flag(o2_params[0], o2_params[2], o2_params[-1],
                                     avg_period, o2_params[-2], t2[-1]-t2[0])

            if np.abs(o1_params[1]-avg_period) < 0.5 and np.abs(o2_params[1]-avg_period)<0.5:
                flag1 = flag2 = 0.0

            if flag1 != 0 and flag2 != 0:
                orbit_flag[i] = 1.0
            else:
                orbit_flag[i] = 0.0
                
            periods[i] = np.nanmedian([o1_params[0], o2_params[0]])
            
            orbit_flag1[i] = flag1
            orbit_flag2[i] = flag2
                
            stds[i]    = o1_params[-1]
            peak_power[i] = o1_params[2]
            periods2[i] = o2_params[0]
            peak_power2[i] = o1_params[-2]

        tab.add_column(Column(self.IDs, 'Target_ID'))
        tab.add_column(Column(periods, name='period_days'))
        tab.add_column(Column(periods2, name='secondary_period_days'))
        tab.add_column(Column(stds, name='gauss_width'))
        tab.add_column(Column(peak_power, name='max_power'))
        tab.add_column(Column(peak_power2, name='secondary_max_power'))
        tab.add_column(Column(orbit_flag, name='orbit_flag'))
        tab.add_column(Column(orbit_flag1, name='oflag1'))
        tab.add_column(Column(orbit_flag2, name='oflag2'))

        tab = self.averaged_per_sector(tab)

        self.LS_results = tab


            
    def assign_flag(self, period, power, width, avg, secpow, 
                    maxperiod, orbit_flag=0):
        """ Assigns a flag in the table for which periods are reliable.
        """
        flag = 100
        if period > maxperiod:
            flag = 4
        if (period < maxperiod) and (power > 0.005):
            flag = 3
        if (period < maxperiod) and (width <= period*0.6) and (power > 0.005):
            flag = 2
        if ( (period < maxperiod) and (width <= period*0.6) and
             (secpow < 0.96*power) and (power > 0.005)):
            flag = 1
        if ( (period < maxperiod) and (width <= period*0.6) and 
             (secpow < 0.96*power) and (np.abs(period-avg)<1.0) and (power > 0.005)):
            flag = 0
        if flag == 100:
            flag = 5
        return flag

            
    def averaged_per_sector(self, tab):
        """ Looks at targets observed in different sectors and determines
            which period measured is likely the best period. Adds a column
            to MeasureRotations.LS_results of 'true_period_days' for the 
            results.

        Returns
        -------
        astropy.table.Table
        """
        def flag_em(val, mode, lim):
            if np.abs(val-mode) < lim:
                return 0
            else:
                return 1

        averaged_periods = np.zeros(len(tab))
        flagging = np.zeros(len(tab), dtype=int)

        limit = 0.3

        for tic in np.unique(self.IDs):
            inds = np.where(tab['Target_ID']==tic)[0]
            primary = tab['period_days'].data[inds]
            secondary = tab['secondary_period_days'].data[inds]
            all_periods = np.append(primary, secondary)

#            ind_flags = np.append(tab['oflag1'].data[inds],
#                                  tab['oflag2'].data[inds])
            avg = np.array([])
            tflags = np.array([])

            if len(inds) > 1:
                try:
                    mode = stats.mode(np.round(all_periods,2))
                    if mode > 11.5:
                        avg = np.full(np.nanmean(primary), len(inds))
                        tflags = np.full(2, len(inds))
                    else:
                        for i in range(len(inds)):
                            if np.abs(primary[i]-mode) < limit:
                                avg = np.append(avg, primary[i])
                                tflags = np.append(tflags,0)
                                
                            elif np.abs(secondary[i]-mode) < limit:
                                avg = np.append(avg, secondary[i])
                                tflags = np.append(tflags,1)
                                
                            elif np.abs(primary[i]/2.-mode) < limit:
                                avg = np.append(avg, primary[i]/2.)
                                tflags = np.append(tflags,0)

                            elif np.abs(secondary[i]/2.-mode) < limit:
                                avg = np.append(avg, secondary[i]/2.)
                                tflags = np.append(tflags,1)
                                
                            elif np.abs(primary[i]*2.-mode) < limit:
                                avg = np.append(avg, primary[i]*2.)
                                tflags = np.append(tflags,0)
                                
                            elif np.abs(secondary[i]*2.-mode) < limit:
                                avg = np.append(avg, secondary[i]*2.)
                                tflags = np.append(tflags,1)
                                
                            else:
                                tflags = np.append(tflags, 2)

                except:
                    for i in range(len(inds)):
                        if tab['oflag1'].data[inds[i]]==0 and tab['oflag2'].data[inds[i]]==0:
                            avg = np.append(avg, tab['period_days'].data[inds[i]])
                            tflags = np.append(tflags, 0)
                        else:
                            tflags = np.append(tflags,2)
                            
                    
            else:
                avg = np.nanmean(primary)
                if tab['oflag1'].data[inds] == 0 and tab['oflag2'].data[inds]==0:
                    tflags = 0
                else:
                    tflags = 2

            averaged_periods[inds] = np.nanmean(avg)
            flagging[inds] = tflags

                        
        tab.add_column(Column(flagging, 'Flags'))
        tab.add_column(Column(averaged_periods, 'avg_period_days'))
        return tab


    def phase_lightcurve(self, table=None, trough=-0.5, peak=0.5, kernel_size=101):
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
        def map_per_orbit(time, flux, kernel_size, cadences):
            mf = medfilt(flux, kernel_size=kernel_size)
            argmin = np.argmin(mf[:cadences])
            mapping = np.linspace(0.5,-0.5, cadences)
            phase = np.ones(len(flux))

            full = int(np.floor(len(time)/cadences))
            
            phase[0:argmin] = mapping[len(mapping)-argmin:]
            
            points = np.arange(argmin, cadences*(full+1)+argmin, cadences, dtype=int)
            for i in range(len(points)-1):
                try:
                    phase[points[i]:points[i+1]] = mapping            
                except:
                    pass
            remainder = len(np.where(phase==1.0)[0])            
            phase[len(phase)-remainder:] = mapping[0:remainder]
            return phase

        if table is None:
            table = self.LS_results

        PHASES = np.copy(self.flux)

        for i in tqdm(range(len(table)), desc="Mapping phases"):
            flag = table['Flags'].data[i]
            if flag == 0 or flag == 1:
                period = table['avg_period_days'].data[i] * u.day
                cadences = int(np.round((period.to(u.min)/2).value))

                all_time = self.time[i]
                all_flux = self.flux[i]
                
                diff = np.diff(all_time)
                gaptime = np.where(diff>=np.nanmedian(diff)+12*np.nanstd(diff))[0][0]
                
                t1, f1 = all_time[:gaptime+1], all_flux[:gaptime+1]
                t2, f2 = all_time[gaptime+1:], all_flux[gaptime+1:]
                
                o1map = map_per_orbit(t1, f1, kernel_size=101, cadences=cadences)
                o2map = map_per_orbit(t2, f2, kernel_size=101, cadences=cadences)
                
                phase = np.append(o1map, o2map)

            else:
                phase = np.zeros(len(self.flux[i]))
            
            PHASES[i] = phase

        self.phases = PHASES
