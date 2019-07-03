import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.coordinates import SkyCoord, Angle
from astropy.stats import sigma_clip, LombScargle
from astropy import units as u
from astropy.io import fits
from astropy import constants as c

from astroquery.mast import Catalogs

import eleanor
import exoplanet as xo
from lightkurve.lightcurve import LightCurve as LC
from altaipony.flarelc import FlareLightCurve

from .injection_recovery import *

import pymc3 as pm
import theano.tensor as tt

plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (14,8)

__all__ = ['YoungStars']

class YoungStars(object):
    
    def __init__(self, fn=None, fn_dir=None):
        if fn_dir is None:
            self.directory = '.'
        else:
            self.directory = fn_dir

        self.file = fn

        if (type(fn) == list) or (type(fn) == np.ndarray):
            if len(fn) > 1:
                self.multi = True
            else:
                self.multi = False
                self.file  = fn[0]
        else:
            self.multi = False

        self.load_data()
        self.query_information()
        self.normalize_lc()
        self.measure_rotation()
        self.age()
    
        self.gp_flux = None
        self.flares  = None
        self.flc     = None

        self.gp_model = None
        self.gp_soln  = None


    def load_data(self):
        """Allows for the option to pass in multiple files. 
        """
        if self.multi is True:
            self.star, self.data = [], []
            for fn in self.file:
                s = eleanor.Source(fn=fn, fn_dir=self.directory)
                d = eleanor.TargetData(s)
                self.star.append(s)
                self.data.append(d)
            self.star = np.array(self.star)
            self.data = np.array(self.data)
            self.tic    = self.star[0].tic
            self.coords = self.star[0].coords
        else:
            self.star = eleanor.Source(fn=self.file, fn_dir=self.directory)
            self.data = eleanor.TargetData(self.star)
            self.tic  = self.star.tic
            self.coords = self.star.coords
        return
            

    def find_breaks(self, time=None):
        """Finds gaps due to data downlink or other telescope issues.
        """
        if time is None:
            time = self.time
        diff = np.diff(time)
        ind  = np.where((diff >= 2.5*np.std(diff)+np.mean(diff)))[0]

        subsets = []
        for i in range(len(ind)):
            if i == 0:
                region = np.arange(0, ind[i]+1, 1)
            elif i > 0 and i < (len(ind)-1):
                region = np.arange(ind[i], ind[i+1]+1, 1)
            elif i == (len(ind)-1):
                region = np.arange(ind[i-1], len(time), 1)
            subsets.append(region)
        return np.array(subsets)


    def normalize_lc(self):
        """Normalizes light curve via chunks of data.
        """
        def normalized_subset(regions, t, flux, err, cads):
            time, norm_flux = np.array([]), np.array([])
            error, cadences = np.array([]), np.array([])
#            for i in range(len(ind)):
#                if i == 0:
#                    region = np.arange(0, ind[i]+1, 1)
#                elif i > 0 and i < (len(ind)-1):
#                    region = np.arange(ind[i], ind[i+1]+1, 1)
#                elif i == (len(ind)-1):
#                    region = np.arange(ind[i-1], len(flux), 1)
            for reg in regions:
                f = flux[reg]
                norm_flux = np.append( f/np.nanmedian(f), norm_flux)
                time      = np.append(t[reg], time)
                error     = np.append(err[reg], error)
                cadences  = np.append(cads[reg], cadences)
            return time, norm_flux, error, cadences


        self.time, self.norm_flux = np.array([]), np.array([])
        self.flux_err = np.array([])
        self.cadences = np.array([])

        if self.multi is True:
            for d in self.data:
                q = d.quality == 0
                t = d.time[q]
                f = d.corr_flux[q]
                err = d.flux_err[q]

                # Searches for breaks based on differences in time array
                regions = self.find_breaks(time=t)
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, t, f, err, d.ffiindex[q])
                self.time = np.append(sector_t, self.time)
                self.norm_flux = np.append(sector_f, self.norm_flux)
                self.flux_err  = np.append(sector_e, self.flux_err)
                self.cadences  = np.append(sector_c, self.cadences)
        else:
            q = self.data.quality == 0
            regions = self.find_breaks(time=self.data.time[q])
            sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, self.data.time[q], 
                                                                       self.data.corr_flux[q],
                                                                       self.data.flux_err[q],
                                                                       self.data.ffiindex[q])
            self.time = sector_t
            self.norm_flux = sector_f
            self.flux_err  = sector_e
            self.cadences  = sector_c
            
        self.time, self.norm_flux = zip(*sorted(zip(self.time, self.norm_flux)))
        self.time, self.norm_flux = np.array(self.time), np.array(self.norm_flux)
        self.cadences = np.sort(self.cadences)

    def query_information(self):        
        """Queries the TIC for basic stellar parameters. 
        """
        result = Catalogs.query_object('tic'+str(int(self.tic)),
                                       radius=0.0001,
                                       catalog="TIC")
        # APASS Magnitudes
        self.jmag = result['Jmag'][0]
        self.hmag = result['Hmag'][0]
        self.kmag = result['Kmag'][0]
        self.jmag_err = result['e_Jmag'][0]
        self.hmag_err = result['e_Hmag'][0]
        self.kmag_err = result['e_Kmag'][0]
        
        # 2MASS Magnitude
        self.vmag = result['Vmag'][0]
        self.vmag_err = result['e_Vmag'][0]

        # Gaia magnitudes
        self.gaia_bp = result['gaiabp'][0]
        self.gaia_rp = result['gaiarp'][0]
        self.gaia_g  = result['GAIAmag'][0]
        self.gaia_bp_err = result['e_gaiabp'][0]
        self.gaia_rp_err = result['e_gaiarp'][0]
        self.gaia_g_err  = result['e_GAIAmag'][0]

        # GAIA proper motions
        self.pmra  = result['pmRA'][0]
        self.pmdec = result['pmDEC'][0]
        
        # GAIA parallax
        self.plx = result['plx'][0]

        # GAIA temperature
        self.teff = result['Teff'][0]
        self.e_teff = result['e_Teff'][0]
        self.lum  = result['lum'][0]
        self.e_lum = result['e_lum'][0]


    def measure_rotation(self, fmin=1./100., fmax=1.0/0.1, cut=30):
        """Uses a Lomb-Scargle periodogram to measure rotation period.
        """
        freq, power = LombScargle(self.time, self.norm_flux).autopower(minimum_frequency=fmin,
                                                                       maximum_frequency=fmax)
        period = 1.0/freq
        cut_greater = period <= cut
        power  = power[cut_greater]
        period = period[cut_greater]
        pmax   = period[np.argmax(power)]
        
        self.LS_power   = power
        self.LS_period  = period
        self.p_rot      = pmax
        

    def age(self):
        """Determines the age (in Myr) using relation from Mamajek & Hillenbrand (2009).
        """
        denom = 0.407 * np.abs((self.gaia_bp-self.vmag) - 0.495)**0.325
        if self.p_rot is not None:
            self.age = (self.p_rot/denom)**(1.0/0.566)
        else:
            raise Exception("Please measure rotation period before getting age.")

                  
    
    def savitsky_golay(self, sigma=2.5, window_length=15, niters=5, 
                       fake=False, flux=None, flux_err=None):
        """Simple Savitsky-Golay filter for detrending.
        """
        if flux is None:
            flux = self.norm_flux
        if flux_err is None:
            flux_err = self.flux_err

        lc, trend = LC(self.time, flux, flux_err=flux_err).flatten(window_length=window_length,
                                                                   return_trend=True,
                                                                   niters=niters,
                                                                   sigma=sigma)
        if fake is False:
            self.sg_flux     = np.array(lc.flux)
            self.sg_flux_err = np.array(lc.flux_err)
            self.sg_trend    = np.array(trend.flux)
        else:
            return np.array(lc.flux), np.array(lc.flux_err)


    def gp_modeling(self, time=None, flux=None, flux_err=None,
                    mask=None, sigma=3, niters=8, iterative=False):
        """Applies GP model to trend normalized light curve.
        """
        if flux is None:
            flux = self.norm_flux
        if time is None:
            time = self.time
        if flux_err is None:
            flux_err = self.flux_err
        if mask is None:
            mask = np.zeros(len(time), dtype=bool)

        if (len(time) != len(flux)) or (len(time) != len(flux_err)):
            raise ValueError("Please ensure you're passing in arrays of the same length.")

        self.mask = mask

        x    = np.array(time)
        y    = np.array(flux)
        yerr = np.array(flux_err)
        
        x = np.array(x[~mask])
        y = np.array(y[~mask])
        yerr = np.array(yerr[~mask])
        
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        yerr = np.ascontiguousarray(yerr, dtype=np.float64)
        time = np.ascontiguousarray(self.time, dtype=np.float64)

        mu = np.nanmean(y)
        y = (y/mu - 1) * 1e3
        yerr = yerr * 1e3 / mu
        
        results   = xo.estimators.lomb_scargle_estimator(x, y, 
                                                         min_period=self.p_rot*0.5, 
                                                         max_period=self.p_rot*2)
        peak_per  = results['peaks'][0]['period']
        per_uncert= results['peaks'][0]['period_uncert']
        self.xo_LS_results = results

        peak = results["peaks"][0]
        freq, power = results["periodogram"]

        with pm.Model() as model:
            mean = pm.Normal("mean", mu=0.0, sd=5.0)

            # white noise
            logs2 = pm.Normal("logs2", mu=np.log(np.min(yerr)/2.0), sd=10.0)

            
            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)/2.0), sd=20.0)

            # Bounds on period
#            BoundedNormal = pm.Bound(pm.Normal, lower=np.log(peak_per*0.5), 
#                                     upper=np.log(peak_per*3))
#            logperiod = BoundedNormal("logperiod", mu=np.log(2*peak["period"]), sd=per_uncert)
        
            # Q from simple harmonic oscillator 
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)

            # TRY WITH NORMAL MU 0.5 SD LOW
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
#            period = pm.Deterministic("period", tt.exp(logperiod))

            # Set up the Gaussian Process model

            # TRY WITH SHOTERM INSTEAD OF ROTATIONTERM
            kernel = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=peak_per,
                log_Q0=logQ0,
                log_deltaQ=logdeltaQ,
                mix=mix
                )
            gp = xo.gp.GP(kernel, x, yerr**2 + tt.exp(logs2), J=4)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(y - mean))

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())
        
            # Fit mean model first
            # Fit period and amplitude together
            # Fit over Q
            # Fit over mean
            # Fit period and amplitude together again
            map_soln = xo.optimize(start=model.test_point)
            map_soln = xo.optimize(start=model.test_point, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[logamp])
#            map_soln = xo.optimize(start=map_soln, vars=[logperiod])
            map_soln = xo.optimize(start=map_soln, vars=[logQ0])
            map_soln = xo.optimize(start=map_soln, vars=[logdeltaQ])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[mix])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[logamp])#, logperiod])
            map_soln = xo.optimize(start=map_soln, vars=[mix])

            map_soln = xo.optimize(start=map_soln)

        with model:
            mu, var = xo.eval_in_model(gp.predict(time, return_var=True), map_soln)

        if iterative is False:
            self.gp_soln  = map_soln
            self.gp_model = mu
            self.gp_flux  = self.norm_flux - (mu+1)

        else:
            self.gp_it_soln  = map_soln
            self.gp_it_model = mu
            self.gp_it_glux  = self.norm_flux - (mu+1)


    def iterative_gp_modeling(self, sigma=3, niters=5):
        """Iteratively fits GP model after first one has been created.
        """
        if self.gp_model is None:
            raise Exception("Please call gp_model() before iteratively fitting.")

        else:
            filtered = sigma_clip(self.gp_flux, sigma=sigma, maxiters=niters)
            mask = filtered.mask

            self.gp_modeling(time=self.time, flux=self.norm_flux,
                             flux_err=self.flux_err, mask=filtered.mask,
                             iterative=True, sigma=sigma, niters=niters)


    def equivalent_duration(self, time, flux, error):
        """Calculates the equivalent width and error for a given flare.
        """
        x = time * 60.0 * 60.0 * 24.0
        residual = flux/np.nanmedian(flux)  - 1.0
        ed = np.sum(np.diff(x) * residual[:-1])
        err = np.sum( (residual / error)**2.0 / np.size(error))
        return ed, err


    def identify_flares(self, detrended_flux=None, detrended_flux_err=None, 
                        method="savitsky-golay", N1=3, N2=1, N3=2, sigma=2.5, minsep=3,
                        cut_ends=5, fake=False):
        """Identifies flare candidates using AltaiPony.
        """

        def tag_flares(flux, sig):
            mask = sigma_clip(flux, sigma=sig).mask
            median = np.nanmedian(flux[mask])
            isflare = np.where( (mask==True) & ( (flux-median) > 0.) &
                                (flux > (np.std(flux)+median) ))[0]
            return isflare

        if detrended_flux is None:
            if (self.gp_flux is not None) and (method.lower() == "gp"):
                detrended_flux = self.gp_flux
            elif (self.sg_flux is not None) and (method.lower() == "savitsky-golay"):
                detrended_flux = self.sg_flux
            elif (detrended_flux is None) and (self.sg_flux is None) and (self.gp_flux is None):
                raise Exception("Pleae either run a detrending method or pass in a 'detrend_flux' argument.")


        if detrended_flux_err is None:
            detrended_flux_err = self.flux_err

        columns = ['istart', 'istop', 'tstart', 'tstop',
                   'ed_rec_s', 'ed_rec_err', 'ampl_rec', 'energy_ergs']
        flares = pd.DataFrame(columns=columns)
        brks = self.find_breaks()

        istart, istop = np.array([], dtype=int), np.array([], dtype=int)
        
        for b in brks:
            time  = self.time[b]
            flux  = detrended_flux[b]
            error = detrended_flux_err[b]

            isflare = tag_flares(flux, sigma)
            candidates = isflare[isflare > 0]

            if len(candidates) < 1:
                if fake is False:
                    print("No flares found in ", np.min(time), " - ", np.max(time))
            else:
                # Find start & stop indices and combine neighboring candidates
                sep_cand = np.where(np.diff(candidates) > minsep)[0]
                istart_gap = candidates[ np.append([0], sep_cand + 1) ]
                istop_gap = candidates[ np.append(sep_cand,
                                                  [len(candidates) - 1]) ]

            # CUTS 5 DATA POINTS FROM EACH BREAK
            ends = ((istart_gap > cut_ends) & ( (istart_gap+np.min(b)) < (np.max(b)-cut_ends)) )
            istart = np.append(istart, istart_gap[ends] + np.min(b))
            istop  = np.append(istop , istop_gap[ends]  + np.min(b) + 1)

            ed_rec, ed_rec_err = np.array([]), np.array([])
            ampl_rec = np.array([])
            for i in range(len(istart)):
                time = self.time[istart[i]:istop[i]+1]
                flux = detrended_flux[istart[i]:istop[i]+1]
                err  = detrended_flux_err[istart[i]:istop[i]+1]
                ed, ed_err = self.equivalent_duration(time=time, flux=flux, error=err)
                ed_rec     = np.append(ed_rec, ed)
                ed_rec_err = np.append(ed_rec_err, ed_err)
                ampl_rec   = np.append(ampl_rec, np.max(flux))
                

        flares['istart']     = istart
        flares['istop']      = istop
        flares['ed_rec_s']   = ed_rec
        flares['ed_rec_err'] = ed_rec_err
        flares['ampl_rec']   = ampl_rec
        flares['tstart']     = self.time[istart]
        flares['tstop']      = self.time[istop]

        energy = (flares.ed_rec_s.values * u.s) * (self.lum * c.L_sun)
        energy = energy.to(u.erg)
        flares['energy_ergs'] = energy.value

        if fake is False:
            self.flares = flares
        else:
            return flares


    def flare_recovery(self, nflares=100, mode='uniform', ed=[0.5, 130.0], ampl=[1e-3, 0.1]):
        """Determines the flare recovery probability.
        """
        ir = InjectionRecovery(self, nflares=nflares)
        ir.generate_fake_flares(ed, ampl, mode=mode)
        ir.inject_flares()
        
        for model in ir.models:
            detrended_flux, detrended_flux_err = self.savitsky_golay(fake=True, flux=model)
            flares = self.identify_flares(fake=True, detrended_flux=detrended_flux,
                                          detrended_flux_err=detrended_flux_err)
            
                
        self.recovery_tests = ir


    def plot_flares(self, time=None, flux=None, high_amp=0.009, mask=None, flare_table=None):
        if time is None:
            time = self.time
        if flux is None:
            flux = self.flc.detrended_flux
        if mask is None:
            mask = np.zeros(len(time))

        if self.flc is None:
            return("Please call YoungStars.identify_flares() before calling this function.")
        if flare_table is None:
            flare_table = self.flares

        plt.figure(figsize=(12,6))
        plt.plot(time[mask], flux[mask], c='k', alpha=0.8)
        plt.title('TIC '+str(self.tic))

        for i,p in flare_table.iterrows():
            plt.plot(time[p.istart:p.istop+1], flux[p.istart:p.istop+1], '*',
                     ms=10, c='turquoise')
            if p.ampl_rec >= high_amp:
                plt.plot(time[p.istart:p.istop+1], flux[p.istart:p.istop+1], '*',
                         ms=10, c='darkorange')
        plt.ylim(np.min(flux[mask])-0.01, np.max(flux[mask])+0.01)
        plt.xlim(np.min(time[mask]), np.max(time[mask]))
        plt.ylabel('Noralized Flux')
        plt.xlabel('Time (BJD - 2457000)')
        plt.tight_layout()
        plt.show()


    def plot_periodogram(self, save=False):
        plt.plot(self.LS_period, self.LS_power, 'k', alpha=0.8)
        plt.xlabel('Period [days]')
        plt.ylabel('Lomb-Scargle Power')
        if save is False:
            plt.show()
        else:
            fn = '{}_periodogram.png'.format(self.tic)
            path = os.path.join(self.directory, fn)
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', dpi=200)


    def plot_residuals(self, x=None, y=None, model=None):
        if x is None:
            x = self.time
        if y is None:
            y = self.norm_flux
        if model is None:
            if (self.gp_flux is None) & (self.sg_flux is not None):
                model = self.sg_trend
                resid = y - model
            elif (self.sg_flux is None) & (self.gp_flux is not None):
                model = self.gp_model
                resid = self.sg_flux
            else:
                raise ValueError("You have no detrended flux to compare to. Try again.")
        else:
            resid = y - model
    
        plt.figure(figsize=(14,8))
        gs  = gridspec.GridSpec(3,3)
    
        ax1 = plt.subplot(gs[0:2,0:])
        ax1.set_xticks([])
        ax1.plot(x, y, 'k', linewidth=3, label='Raw', alpha=0.8)
        ax1.plot(x, model, c='orange', label='Model')
        plt.legend()
        ax2 = plt.subplot(gs[2, 0:])
        ax2.plot(x, resid, c='turquoise', linewidth=2)
        
        plt.show()
