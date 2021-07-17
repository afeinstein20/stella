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
        transit_models : np.ndarray
           Array of the transit models. Will be roughly shape
           (nmodels, cadences).
        transit_labels : np.ndarray
           An array of ones used to identify which examples include
           transits.
        transit_table : astropy.table.Table
           Table of output transit model parameters.
        """
        transit_table = Table(names=['id','rstar', 'per', 'rprstar', 
                                     'arstar', 'inc', 'ecc', 
                                     'w', 'u1', 'u2'])
        transit_models = np.zeros((nmodels, self.cadences))
        transit_labels = np.zeros(nmodels, dtype=int)

        time = np.linspace(1320, 1321.5, self.cadences)
        outfn = 'batman_models/batman_{0:06d}.npy'
        
        if limits == None:
            limits = [ [0.1, 1.5], #rstar (rsun)
                       [1.0, 10.], #period (days)
                       [4.0, 15.], #rplanet (rearth)
                       [0, 0],     #placeholder for a/rstar
                       [86., 90.], #inclination
                       [0, 0],     #eccentricity
                       [-180, 180],#periastron
                       [0.0, 1.0], #limb darkening (u1)
                       [0.0, 1.0]  #limb darkening (u2)
                     ]

        np.random.seed(seed)

        for i in tqdm(range(nmodels)):
            f, inp = self.batman_models(time,
                                        [np.random.uniform(limits[0][0], limits[0][1], 1),
                                         np.random.uniform(limits[1][0], limits[1][1], 1),
                                         np.random.uniform(limits[2][0], limits[2][1], 1),
                                         np.random.uniform(limits[3][0], limits[3][1], 1),
                                         np.random.uniform(limits[4][0], limits[4][1], 1),
                                         np.random.uniform(limits[5][0], limits[5][1], 1),
                                         np.random.uniform(limits[6][0], limits[6][1], 1),
                                         np.random.uniform(limits[7][0], limits[7][1], 1),
                                         np.random.uniform(limits[8][0], limits[8][1], 1) ]
                                        )
            # checks to make sure there's a realistic transit in the model
            if len(np.where(f==1)[0]) != CADENCES:
                np.save(os.path.join(self.output_dir, outfn.format(i)),
                        [f-1.0, np.append(i, inp)])
                transit_table.add_row(np.append(i, inp))
                transit_models[i] = f - 1.0
                transit_labels[i] = 1

        remove = np.where(transit_labels==0)[0]
        transit_models = np.delete(transit_models, remove, axis=0)
        transit_labels = np.delete(transit_labels, remove)

        self.transit_models = transit_models
        self.transit_labels = transit_labels
        self.transit_table  = transit_table
