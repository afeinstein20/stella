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
from astroquery.gaia import Gaia
import wotan

import eleanor
import exoplanet as xo
from lightkurve.lightcurve import LightCurve as LC
from altaipony.flarelc import FlareLightCurve

from .injection_recovery import *
from .identify import *
from .plotting import *

import pymc3 as pm
import theano.tensor as tt

import warnings
warnings.filterwarnings("ignore", module='astropy.io.votable.tree')

plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (14,8)

__all__ = ['YoungStars']

class YoungStars(object):
    
    def __init__(self, time=None, flux=None, flux_err=None, tic=None,
                 cadences=None, fn=None, fn_dir=None, quality=None):

        self.time = time
        self.flux = flux
        self.directory= fn_dir
        self.flux_err = flux_err
        self.cadences = cadences
        self.quality  = quality

        if (time is not None) and (flux is not None):
            self.multi = False

            if cadences is None:
                self.cadences = np.arange(0, len(time), 1, dtype=int)
            if flux_err is None:
                self.flux_err = np.zeros(len(flux))

            if tic is not None:
                self.tic = tic
                self.coords, self.tmag, _ = eleanor.mast.coords_from_tic(tic)
            else:
                self.tic = tic

            self.file = fn

        elif fn is not None:
            self.file = np.array(fn)
            # Sets the default directory to current working directory
            if fn_dir is None:
                fn_dir = '.'
            self.directory = fn_dir

            if len(self.file) > 1:
                self.multi = True
            else:
                self.multi = False
                self.file  = self.file[0]

            self.load_data()

        self.normalize_lc()
        self.measure_rotation()

        if self.tic is not None:
            self.query_information()
            self.age()
        else:
            self.lum = None
            print("No TIC was given. Cannot query magnitudes to estimate age or flare energies.")
    
        self.gp_flux = None
        self.sg_flux = None
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
            print("TIC", self.tic)
            self.coords = self.star.coords
        return
            

    def find_breaks(self, time=None):
        """Finds gaps due to data downlink or other telescope issues.
        """
        if time is None:
            time = self.time
        diff = np.diff(time)
        std  = np.std(diff)
        mean = np.nanmean(diff)
        ind  = np.where((diff > (2.5*std+mean)))[0]
        subsets = []
        for i in range(len(ind)):
            if i == 0:
                region = np.arange(0, ind[i], 1)
            elif i > 0 and i < (len(ind)-1):
                region = np.arange(ind[i-1], ind[i], 1)
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

            if len(regions) > 0:
                for reg in regions:
                    f = flux[reg]
                    norm_flux = np.append(norm_flux, f/np.nanmedian(f))
                    time      = np.append(time, t[reg])
                    error     = np.append(error, err[reg])
                    cadences  = np.append(cadences, cads[reg])
            else:
                time      = t
                norm_flux = flux/np.nanmedian(flux)
                error     = err
                cadences  = cads
            return time, norm_flux, error, cadences

        if self.file is not None:
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

        else:
            regions = self.find_breaks(time=self.time)
            
            self.time, self.norm_flux, self.flux_err, self.cadences = normalized_subset(regions, 
                                                                                        self.time,
                                                                                        self.flux,
                                                                                        self.flux_err,
                                                                                        self.cadences)
            
    def query_information(self):        
        """Queries the TIC for basic stellar parameters. 
        """
        result = Catalogs.query_object('tic'+str(int(self.tic)),
                                       radius=0.0001,
                                       catalog="TIC")
        self.tmag = result['Tmag'][0]

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
        self.bmag = result['Bmag'][0]
        self.bmag_err = result['e_Bmag'][0]


        coord = SkyCoord(self.coords[0], self.coords[1], unit=(u.deg, u.deg),
                         frame='icrs')
        radius = u.Quantity(22, u.arcsec)
        j = Gaia.cone_search_async(coord, radius)
        result = j.get_results()

        # Gaia magnitudes
        self.gaia_bp = np.round(result['phot_bp_mean_mag'][0],4)
        self.gaia_rp = np.round(result['phot_rp_mean_mag'][0],4)
        self.gaia_g  = np.round(result['phot_g_mean_mag'][0],4)

        # GAIA proper motions
        self.pmra  = result['pmra'][0]
        self.pmdec = result['pmdec'][0]
        self.pmra_err = result['pmra_error'][0]
        self.pmdec_err = result['pmdec_error'][0]

        # GAIA parallax
        self.plx = result['parallax'][0]
        self.plx_err = result['parallax_error'][0]

        # GAIA radial velocity
        self.rv = result['radial_velocity'][0]
        self.rv_err = result['radial_velocity_error'][0]

        # GAIA temperature
        self.teff = np.round(result['teff_val'][0],4)
        self.teff_err = [np.round(result['teff_percentile_lower'][0],4), 
                         np.round(result['teff_percentile_upper'][0],4)]
        self.lum  = np.round(result['lum_val'][0],4)
        self.lum_err = [np.round(result['lum_percentile_lower'][0],4), 
                        np.round(result['lum_percentile_upper'][0],4)]
        self.rad = np.round(result['radius_val'][0],4)
        self.rad_err = [np.round(result['radius_percentile_lower'][0],4), 
                        np.round(result['radius_percentile_upper'][0],4)]

        if (type(self.lum) == np.ma.core.MaskedConstant) and (type(self.teff) != np.ma.core.MaskedConstant):
            if type(self.rad) != np.ma.core.MaskedConstant:
                lum = 4 * np.pi * (self.rad * c.R_sun)**2 * (self.teff * u.K)**4 * c.sigma_sb
                self.lum = lum / c.L_sun
                


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
        denom = 0.407 * np.abs((self.bmag-self.vmag) - 0.495)**0.325
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


    def wotan(self, time=None, flux=None, flux_err=None,
              mask=None, sigma=3, niters=8, iterative=False, kernel='squared_exp',
              kernel_size=10.0, window_length=15, method='gp'):
        """Applies GP model to trend normalized light curve.
        """
        def gp(m, k, ks):
            nonlocal time, flux
            f, t = wotan.flatten(time, flux, kernel_size=ks, return_trend=True,
                                 method=m, kernel=k)
            return f, t

        def sg(wl):
            nonlocal time, flux
            f, t = wotan.flatten(time, flux, return_trend=True, method='savgol',
                                 window_length=wl)
            return f, t

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

        gp_flux, gp_trend = np.array([]), np.array([])

        for i,b in enumerate(self.brks):
            if method == 'savgol':
                flattened, trend = sg(window_length)
            if method == 'gp':
                flattened, trend = gp(method, kernel, kernel_size)

            gp_flux  = np.append(gp_flux, flattened[b])
            gp_trend = np.append(gp_trend, trend[b])
 
        self.gp_flux  = gp_flux
        self.gp_trend = gp_trend 


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
#        y = (y/mu - 1) * 1e3
#        yerr = yerr  * 1e3 / mu
        
        results   = xo.estimators.lomb_scargle_estimator(x, y, 
                                                         min_period=self.p_rot*0.5, 
                                                         max_period=self.p_rot*2)
        peak_per  = results['peaks'][0]['period']
        per_uncert= results['peaks'][0]['period_uncert']
        self.xo_LS_results = results

        peak = results["peaks"][0]
        freq, power = results["periodogram"]

        with pm.Model() as model:
            mean = pm.Normal("mean", mu=1.0, sd=5.0)

            # white noise
            logs2 = pm.Normal("logs2", mu=np.log(np.nanmin(yerr)/2.0), sd=10.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)/2.0), sd=50.0)

            # Bounds on period
            BoundedNormal = pm.Bound(pm.Normal, lower=np.log(peak_per*0.5), 
                                     upper=np.log(peak_per*3))
            logperiod = BoundedNormal("logperiod", mu=np.log(2*peak["period"]), sd=per_uncert)
        
            # Q from simple harmonic oscillator 
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)

            # TRY WITH NORMAL MU 0.5 SD LOW
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
            period = pm.Deterministic("period", tt.exp(logperiod))

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
#            map_soln = xo.optimize(start=model.test_point)
            map_soln = xo.optimize(start=model.test_point, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[logamp])
            map_soln = xo.optimize(start=map_soln, vars=[logperiod])
            map_soln = xo.optimize(start=map_soln, vars=[logQ0])
            map_soln = xo.optimize(start=map_soln, vars=[logdeltaQ])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[mix])
 #           map_soln = xo.optimize(start=map_soln, vars=[mean])
 #           map_soln = xo.optimize(start=map_soln, vars=[logamp, logperiod])
 #           map_soln = xo.optimize(start=map_soln, vars=[mix])

            map_soln = xo.optimize(start=map_soln)

        with model:
            mu = xo.eval_in_model(gp.predict(time, return_var=False), map_soln)

        if iterative is False:
            self.gp_soln  = map_soln
            self.gp_model = mu
            self.gp_flux  = self.norm_flux - (mu+1)

        else:
            self.gp_it_soln  = map_soln
            self.gp_it_model = mu
            self.gp_it_glux  = self.norm_flux - (mu+1)



    def equivalent_duration(self, time, flux, error):
        """Calculates the equivalent width and error for a given flare.
        """
        x = time * 60.0 * 60.0 * 24.0
        flux = flux/np.nanmedian(flux) - 1.0
        ed = np.nansum(np.diff(x) * flux[:-1])
        err = np.nansum( (flux / error)**2.0 / np.size(error))
        return ed, err


    def identify_flares(self, detrended_flux=None, detrended_flux_err=None,
                        method="savitsky-golay", N1=3, N2=1, N3=2, sigma=2.5, minsep=3,
                        cut_ends=5, fake=False):
        
        id = IdentifyFlares(self)
        if fake == False:
            self.brks, self.flares = id.identify_flares(detrended_flux=None, detrended_flux_err=None,
                                                        method=method, N1=3, N2=1, N3=2, sigma=2.5, minsep=3,
                                                        cut_ends=5, fake=False)
        else:
            brks, flares = id.identify_flares(detrended_flux=None, detrended_flux_err=None,
                                                        method=method, N1=3, N2=1, N3=2, sigma=2.5, minsep=3,
                                                        cut_ends=5, fake=True)
            return flares
        

    def flare_recovery(self, nflares=100, mode='uniform', ed=[0.5, 130.0], ampl=[1e-3, 0.1],
                       recovery_resolution=5.0):
        """Determines the flare recovery probability.
        """
        ed   = [np.nanmin(self.flares.ed_rec_s.values), np.nanmax(self.flares.ed_rec_s.values)]
        ampl = [np.nanmin(self.flares.ampl_rec.values)-1, np.nanmax(self.flares.ampl_rec.values)-1]

        ir   = InjectionRecovery(self, nflares=nflares, mode=mode, ed=ed, ampl=ampl, breaks=self.brks)
        self.ir = ir

        rec_table = self.ir.rec_table[self.ir.rec_table.ed_rec_s > 0.]

        bins = np.round(len(rec_table)/recovery_resolution)
        if bins <= 0:
            bins = 1

        prob, xedges, yedges = self.ir.recovery_probability(rec_table, bins)
        plt.imshow(prob, origin='lower')
        plt.colorbar()
        plt.show()
        # Finds recovery probability for originally detected flares
        ed = np.log10(self.flares.ed_rec_s)
        am = self.flares.ampl_rec

        rec_prob = pd.DataFrame(columns=['rec_prob'])
        rec = []
        print(xedges, yedges)
        for i in range(len(ed)):
            xbin = np.where( (ed[i] >= xedges[:-1]) & (ed[i] <= xedges[1:]) )[0]
            ybin = np.where( (am[i] >= yedges[:-1]) & (am[i] <= yedges[1:]) )[0]
            print(ed[i], am[i], xbin, ybin)
            if (len(xbin) > 0) & (len(ybin) > 0):
                rec.append( prob[xbin, ybin][0])
            else:
                rec.append(0)
        rec_prob['rec_prob'] = rec
        self.flares = self.flares.join(rec_prob)
        self.recovery_tests = ir



    def display_flares(self, time=None, flux=None, flare_table=None, mask=None):
        """Plots stars where the flares are on the light curve.
        """
        if time is None:
            time = self.time
        if flux is None:
            if self.gp_flux is None:
                flux = self.sg_flux
            elif self.sg_flux is None:
                flux = self.gp_flux
            else:
                flux = self.norm_flux

        if flare_table is None:
            flare_table = self.flares
        if mask is None:
            mask = np.zeros(len(time), dtype=int)

        plot_flares(time, flux, flare_table, mask)

    def periodogram(self, save=False):
        """Plots periodogram from Lomb-Scargle.
        """
        plot_periodogram(self.LS_period, self.LS_power,
                         save, self.tic)

    def residuals(self, x=None, y=None, model=None):
        """Plots residuals from a given detrending model.
        """
        if x is None:
            x = self.time
        if y is None:
            y = self.norm_flux
        if model is None:
            if self.sg_flux is None:
                model = self.gp_flux
            elif self.gp_flux is None:
                model = self.sg_flux

        plot_residuals(x, y, model)
