import os
import batman
import numpy as np
from tqdm import tqdm
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.table import Table
from lightkurve.lightcurve import LightCurve as LC

__all__ = ['XOSims']


class XOSims(object):
    """
    This class sets up all the tools to build the simulated
    training, validation, and test sets.
    """
    
    def __init__(self, output_dir, cadences=300):
        
        self.cadences = cadences
        self.output_dir = output_dir
        

    def batman_models(self, t, inputs):
        """
        Uses batman to create transit models. 

        Parameters
        ----------
        t : np.ndarray
           Array of time values.
        inputs : np.ndarray
           Array of parameters to feed into the model. The 
           parameters are ordered as follows:
           [Rstar/Rsun, Rp/Rearth, period, a/rstar, inclination,
            eccentricity, periastron, limbdarkening1, limbdarkening2].
        
        Returns
        -------
        f : np.ndarray
           Flux array with a transit.
        inputs : np.ndarray
           Array of parameters that were fed into the batman model.
        """
        rstar = inputs[0] * u.Rsun
        rprstar = (inputs[2]*u.Rearth) / rstar
        per = inputs[1]*u.day
        a = (per.to(u.year).value**2)**(1./3.) * u.AU
        arstar = a / rstar
        
        inputs[2] = rprstar.to(u.m/u.m).value
        inputs[3] = arstar.to(u.m/u.m).value
        
        params = batman.TransitParams()
        params.t0 = t[int(len(t)/2)]    #time of inferior conjunction
        params.per = per.value           #orbital period
        params.rp = rprstar.to(u.m/u.m).value  #planet radius (in units of stellar radii)
        params.a  = arstar.to(u.m/u.m).value   #semi-major axis (in units of stellar radii)
        params.inc = inputs[4]            #orbital inclination (in degrees)
        params.ecc = inputs[5]             #eccentricity
        params.w = inputs[6]              #longitude of periastron (in degrees)
        params.u = [inputs[7], 
                    inputs[8]]       #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"  #limb darkening model
        m = batman.TransitModel(params, t)    #initializes model
        
        return m.light_curve(params), inputs       #calculates light curve


    def create_transit_models(self, nmodels, limits=None, seed=231):
        """
        Creates the transit models and saves to the output directory
        initialized with this class. Creates a subdirectory in this
        output path called `batman_models/`.
        
        Parameters
        ----------
        nmodels : int
           The total number of models to create.
        limits : dictionary, optional
           A dictionary containing the lower and upper bounds for 
           each parameter fed into the batman model. Default is 
           set by the uniform distributions provided in ...
        seed : int, optional
           A number to set np.random.seed so the output models are
           reproducible. Default is 231.

        Attributes
        ----------
        transit_table : astropy.table.Table
           Table of output transit model parameters.
        """
