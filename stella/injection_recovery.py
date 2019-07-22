import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as c

from .identify import *

from scipy.stats import binned_statistic

__all__ = ['InjectionRecovery']

class InjectionRecovery(object):
    
    def __init__(self, yso, nflares=100, mode='uniform', ed=[0.5, 130.0],
                 ampl=[1e-3, 0.1], breaks=None):
        self.yso = yso
        
        self.time     = yso.time
        self.flux     = yso.norm_flux
        self.flares   = yso.flares
        self.nflares  = nflares    # number of fake flare injections

        self.generate_fake_flares(ed, ampl, mode=mode)
        self.inject_flares(breaks)

        columns = ['istart', 'istop', 'tstart', 'tstop',
                   'ed_rec_s', 'ed_rec_err', 'ampl_rec', 'energy_ergs', 'rec']
        self.rec_table = pd.DataFrame(columns=columns)

        self.refind_flares()


    def flare_model(self, time, t0, amp, ed, gauss_rise, exp_decay, 
                    uptime=10):
        """Generates a flare model given parameters.
        """
        amp -= 1

        dt = np.nanmedian(np.diff(time))
        timeup = np.linspace(np.nanmin(time)-dt, np.nanmax(time)+dt, time.size*uptime)
        
        up_t0 = timeup[np.where( timeup >= time[t0] )[0][0]]

        rise = np.where(timeup <= up_t0)[0]
        fall = np.where(timeup >  up_t0)[0]

        rise_model = amp * np.exp( -(timeup[rise] - up_t0)**2.0 / (2.0*gauss_rise**2.0) )
        fall_model = amp * np.exp( -(timeup[fall] - up_t0) / exp_decay )
        
        model = np.append(rise_model, fall_model)

        flare = binned_statistic(timeup, model, statistic='mean', bins=len(time))[0]
        return flare


    def generate_fake_flares(self, ed, ampl, mode='uniform'):
        """Generates a distribution of fake flares to try and recover.
        """
        if ed is None:
            ed   = [np.nanmin(self.flares.ed_rec_s)-2, np.nanmax(self.flares.ed_rec_s)+2]
        if ampl is None:
            ampl = [np.nanmin(self.flares.ampl_rec)-0.033, np.nanmax(self.flares.ampl_rec)+0.033]

        if mode.lower() == 'uniform':
            rand_ampl = np.random.uniform(ampl[0], ampl[1], size=self.nflares)
            rand_ed   = np.random.uniform(ed[0]  , ed[1]  , size=self.nflares)
        
        if mode.lower() == 'loglog':
            ampls = np.logspace(ampl[0], ampl[1], num=5*self.nflares)
            rand_ampl = np.log10(ampls[np.random.randint(0, len(ampls), size=self.nflares)])

            eds   = np.log10(np.logspace(ed[0]  , ed[1]  , num=5*self.nflares))
            rand_ed = eds[np.random.randint(0, len(eds), size=self.nflares)]

        self.fake_ampls = rand_ampl + 1
        self.fake_edurs = rand_ed
    

    def inject_flares(self, brks):
        """Injects random flares into a light curve.
        Returns: new light curves with new flares
        """
        rand_t0 = []

        ends = np.array([], dtype=int)
        for b in brks:
            ends = np.append(ends, np.arange(b[0], b[0]+11, 1))
            ends = np.append(ends, np.arange(b[-1]-10, b[-1]+1, 1))
        ends = np.unique(ends)

        for i in range(5*self.nflares):
            random =  np.random.randint(len(self.time), size=1)[0]
            if random not in ends:
                rand_t0.append(random)
            if len(rand_t0) == 1.5*self.nflares:
                break

        rand_t0 = np.array(rand_t0)
        corrected_t0, corrected_ed = [], []
        models  = []

        for i in range(self.nflares):
            t0 = rand_t0[i]
            a0 = self.fake_ampls[i] - 0.05
            ed = self.fake_edurs[i]
            gauss_rise = np.random.uniform(low=0.01, high=0.06, size=1)[0]
            exp_decay  = np.random.uniform(low=0.01, high=0.05, size=1)[0]

            model = self.flare_model(self.time, t0, a0, ed, gauss_rise, exp_decay) 
            new_flux = np.copy(self.flux) + np.abs(model)

            # Corrected fake flare parameters due to binning
            corrected_t0.append(self.time[np.argmax(new_flux-self.flux)])
            x = self.time * 60.0 * 60.0 * 24.0
            corrected_ed.append(np.nansum(np.diff(x) * np.abs(model[:-1])))

            models.append(new_flux)

        self.models  = np.array(models)
        self.fake_t0 = np.array(corrected_t0)
        self.fake_edurs = np.array(corrected_ed)


    def recovery_probability(self, results, bins):
        """Returns the probability a flare of given amp & ed would be detected.
        """
        ed = np.log10(results.ed_rec_s.values)
        am = results.ampl_rec.values
        prob, xedges, yedges = np.histogram2d(ed, am, bins=bins)
        prob = (prob - np.nanmin(prob)) / (np.nanmax(prob) - np.nanmin(prob))
        prob /= np.sum(prob)
        self.probability = prob
        return prob, xedges, yedges


    def refind_flares(self):
        """Identifies flares in model light curves.
        """
        potential = []

        known_tstart = self.yso.flares.tstart.values
        known_tstop  = self.yso.flares.tstop.values   

        id = IdentifyFlares(self.yso)

        for i, model in enumerate(self.models):
            inj_ed = self.fake_edurs[i]
            inj_am = self.fake_ampls[i]
            inj_t0 = self.fake_t0[i]

            de_flux, de_flux_err = self.yso.savitsky_golay(fake=True,
                                                           flux=model)

            brks, f = id.identify_flares(detrended_flux=de_flux,
                                         detrended_flux_err=de_flux_err,
                                         N1=3, N2=1, N3=1, sigma=2.5, fake=True)

            rec = f[ (f.tstart.values <= inj_t0) & (f.tstop.values >= inj_t0) ]
            rec = rec.copy()

            if len(rec.tstart.values) != 0:
                rec['rec'] = i

            rec['inj_amp'] = inj_am
            rec['inj_ed']  = inj_ed
            rec['inj_t0']  = inj_t0

            self.rec_table = self.rec_table.append(rec, sort=True)

        return
