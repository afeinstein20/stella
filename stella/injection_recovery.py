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

#        self.generate_fake_flares()


    def flare_model(self, time, t0, amp, ed, gauss_rise, exp_decay):
        """Generates a flare model given parameters.
        """
        rise = np.where(time <= t0)[0]
        fall = np.where(time >  t0)[0]

        rise_model = amp * np.exp( -(time[rise]-t0)**2.0 / (2.0*gauss_rise**2.0) )
        fall_model = amp * np.exp( -(time[fall]-t0) / exp_decay )
        return np.append(rise_model, fall_model)


    def generate_fake_flares(self, mode='uniform', ed=None, ampl=None, rate=None):
        """Generates a distribution of fake flares to try and recover.
        """
        if ed is None:
            ed   = [np.min(self.flares.ed_rec_s)-2, np.max(self.flares.ed_rec_s)+2]
        if ampl is None:
            ampl = [np.min(self.flares.ampl_rec)-0.3, np.max(self.flares.ampl_rec)+0.3]
        if rate is None:
            rate = [1e-3, 1e4]

        if mode.lower() == 'uniform':
            rand_ampl = np.random.uniform(ampl[0], ampl[1], size=self.nflares)
            rand_ed   = np.random.uniform(ed[0]  , ed[1]  , size=self.nflares)
        
        if mode.lower() == 'loglog':
            ampls = np.logspace(ampl[0], ampl[1], num=5*self.nflares)
            rand_ampl = ampls[np.randint(0, len(ampls), size=self.nflares)]

            eds   = np.logspace(ed[0]  , ed[1]  , num=5*self.nflares)
            rand_ed = eds[np.randint(0, len(eds), size=self.nflares)]

        return rand_ampl, rand_ed
    
