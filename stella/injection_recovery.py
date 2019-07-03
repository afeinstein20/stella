import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as c

__all__ = ['InjectionRecovery']

class InjectionRecovery(object):
    
    def __init__(self, yso, nflares=100):
        self.yso = yso
        
        self.time    = yso.time
        self.flux    = yso.norm_flux
        self.flares  = yso.flares
        self.nflares = nflares    # number of fake flare injections


    def flare_model(self, time, t0, amp, ed, gauss_rise, exp_decay):
        """Generates a flare model given parameters.
        """
        t0 = time[t0]

        rise = np.where(time <= t0)[0]
        fall = np.where(time >  t0)[0]

        rise_model = amp * np.exp( -(time[rise]-t0)**2.0 / (2.0*gauss_rise**2.0) )
        fall_model = amp * np.exp( -(time[fall]-t0) / exp_decay )
        
        return np.append(rise_model, fall_model)


    def generate_fake_flares(self, ed, ampl, mode='uniform'):
        """Generates a distribution of fake flares to try and recover.
        """
        if ed is None:
            ed   = [np.min(self.flares.ed_rec_s)-2, np.max(self.flares.ed_rec_s)+2]
        if ampl is None:
            ampl = [np.min(self.flares.ampl_rec)-0.033, np.max(self.flares.ampl_rec)+0.033]

        if mode.lower() == 'uniform':
            rand_ampl = np.random.uniform(ampl[0], ampl[1], size=self.nflares)
            rand_ed   = np.random.uniform(ed[0]  , ed[1]  , size=self.nflares)
        
        if mode.lower() == 'loglog':
            ampls = np.logspace(ampl[0], ampl[1], num=5*self.nflares)
            rand_ampl = np.log10(ampls[np.random.randint(0, len(ampls), size=self.nflares)])

            eds   = np.log10(np.logspace(ed[0]  , ed[1]  , num=5*self.nflares))
            rand_ed = eds[np.random.randint(0, len(eds), size=self.nflares)]

        self.fake_ampls = rand_ampl
        self.fake_edurs = rand_ed
    

    def inject_flares(self):
        """Injects random flares into a light curve.
        Returns: new light curves with new flares
        """
        rand_t0 = np.random.randint(len(self.time), size=self.nflares)
        
        models = []
        for i in range(self.nflares):
            t0 = rand_t0[i]
            a0 = self.fake_ampls[i] - 0.05
            ed = self.fake_edurs[i]
            gauss_rise = np.random.uniform(low=0.01, high=0.06, size=1)[0]
            exp_decay  = np.random.uniform(low=0.01, high=0.05, size=1)[0]

            model = self.flare_model(self.time, t0, a0, ed, gauss_rise, exp_decay) 
            
            new_flux = np.copy(self.flux)
            new_flux = new_flux + np.abs(model)
            models.append(new_flux)

        self.models  = np.array(models)
        self.fake_t0 = self.time[rand_t0]
