import numpy as np
from tqdm import tqdm
import more_itertools as mit
from astropy import units as u
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.optimize import minimize

from .utils import *

__all__ = ['FlareParameters']


class FlareParameters(object):
    """
    Uses the predictions from the neural network
    and identifies flaring events based on consecutive
    points. Users define a given probability threshold
    for accpeting a flare event as real.
    """

    def __init__(self, cnn):
        """
        Uses the times, fluxes, and predictions defined
        in stella.ConvNN to identify and fit flares, as
        well as do injection-recovery for completeness.
        
        Parameters
        ----------
        cnn : stella.ConvNN object
        
        Attributes
        ----------
        ids : np.array
        model : tensorflow.python.keras.engine.sequential.Sequential
        times : np.ndarray
        fluxes : np.ndarray
        predictions : np.ndarray
        """
        self.ids         = cnn.predict_ids
        self.model       = cnn
        self.times       = cnn.predict_times
        self.fluxes      = cnn.predict_fluxes
        self.flux_errs   = cnn.predict_errs
        self.predictions = cnn.predictions


    def group_inds(self, values):
        """
        Groups regions marked as flares (> prob_threshold) for
        flare fitting. Indices within 4 of each other are grouped
        as one flare.

        Returns
        -------
        results: np.ndarray
             An array of arrays, which are groups of indices
             supposedly attributed with a single flare.
        """
        results = []

        for i, v in enumerate(values):
            if i == 0:
                mini = maxi = v
                temp = [v]
            else:
                # SETS 4 CADENCE LIMIT
                if (np.abs(v-maxi) <= 4):
                    temp.append(v)
                    if v > maxi:
                        maxi = v
                    if v < mini:
                        mini = v
                else:
                    results.append(temp)
                    mini = maxi = v
                    temp = [v]
                
                # GETS THE LAST GROUP
                if i == len(values)-1:
                    results.append(temp)

        return np.array(results)



    def identify_flare_peaks(self, threshold=0.75, cut_ends=30, injected=False,
                             ids=None, times=None, fluxes=None, flux_errs=None,
                             predictions=None):
        """
        Finds where the predicted value is above the threshold
        as a flare candidate. Groups consecutive indices as one
        flaring event.
        
        Parameters
        ----------
        threshold : float, optional
             The probability threshold for believing an event
             is a flare. Default is 0.75.
        cut_ends : int, optional
             Allows for ignoring the ends of a given light curve.
             Default is 30 cadences are cut.
        injected : bool, optional
             Returns table of recovered flares instead of setting attribute.
             Used for injection-recovery. Default is False.

        Attributes
        ----------
        flare_table : astropy.table.Table
             A table of flare times, amplitudes, and equivalent
             durations. Equivalent duration given in units of days.
        """

        def chiSquare(var, x, y, yerr, model):
            """
            Used in scipy.optimize.minimize to compute chi-square
            on a flare model.
            """
            amp, rise, decay = var            
            m = flare_lightcurve(x, int(len(x)/2), amp, rise,
                                 decay, y=model)[0]

            return np.sum( (y-m)**2 / yerr**2)        

        if injected is False:
            times          = self.times
            ids            = self.ids
            fluxes         = self.fluxes
            flux_errs      = self.flux_errs
            predictions    = self.predictions
            self.threshold = threshold
            self.cut_ends  = cut_ends
            
        # INITIALIZES ASTROPY TABLE
        tab = Table(names=['ID', 'tpeak', 'amp', 'amp_err',
                           'ed', 'ed_err', 'prob'])

        fit_to_region = 50

        # REGION AROUND FLARE TO FIT TO
        for i in range(len(times)):

            time = times[i][cut_ends:len(times[i])-cut_ends] + 0.0
            flux = fluxes[i][cut_ends:len(fluxes[i])-cut_ends] + 0.0
            flux /= np.nanmedian(flux)

            err  = flux_errs[i][cut_ends:len(flux_errs[i])-cut_ends] + 0.0
            prob = predictions[i][cut_ends:len(predictions[i])-cut_ends] + 0.0

            where_prob_higher = np.where(prob >= threshold)[0]

            groupings = self.group_inds(where_prob_higher)

            if len(groupings) > 0:
                for g in groupings:
                    argmax = g[np.argmax(flux[g])]
                    amp = flux[argmax]
                    tpeak = time[argmax]
                    prob  = prob[argmax]

                    fit_min = argmax - fit_to_region
                    fit_max = argmax + fit_to_region

                    # FITS UNDERLYING POLYNOMIAL (MASKS FLARE)
                    mask_inds = np.append(np.arange(argmax-fit_to_region, argmax-4, 1, dtype=int),
                                          np.arange(argmax+4, argmax+fit_to_region, 1, dtype=int))
                    underfit = np.polyfit(time[mask_inds], flux[mask_inds], deg=6)
                    model = np.poly1d(underfit)
                    model = model(flux[fit_min:fit_max])
                                  
                    x = minimize(chiSquare, x0=[amp, 0.0001, 0.0005],
                                     args=(time[fit_min:fit_max], 
                                           flux[fit_min:fit_max], 
                                           err[fit_min:fit_max], model), method='L-BFGS-B')
                        
#                        norm_flux = (ff - model) / model
                        
#                        ed = np.abs(np.sum(norm_flux[:-1] * np.diff(ft) )) * u.day
#                        ed_err = np.sum( (fe/model)**2 )
                        
#                        row = [id, tpeak, amp, err[peak], ed, ed_err, prob[peak]]
#                        tab.add_row(row)



    def injection_recovery(self, amps=[0.001, 0.1], flares_per_inj=20, iters=5):
        """
        Completes injection recovery based on a set of flare parameters. A tqdm
        loading bar will appear that will track each light curve. 

        Parameters
        ----------
        amps : list, optional
             Minimum and maximum flare amplitude to recover. Default is [0.001, 0.1].
        flares_per_inj : int, optional
             Number of flares injected per recovery. Default is 20.
        iters : int, optional
             Number of iterations per each light curve. Default is 5. 

        Attributes
        ----------
        inj_table : astropy.table.Table
             Table of injected flare parameters.

        """
        inj_tab = Table(names=['ID', 'tpeak', 'rec_amp', 'inj_amp', 'prob', 'recovered'])
        
        predictions = []

        for i in tqdm(range(len(self.times))):

            ids = np.full(iters, self.ids[i])

            inj_time = np.full( (iters, len(self.times[i])), self.times[i] )
            inj_errs = np.full( (iters, len(self.flux_errs[i])), self.flux_errs[i] )
            inj_model = np.full( (iters, len(self.fluxes[i])), self.fluxes[i] )
            inj_preds = np.zeros( (iters, len(self.fluxes[i])) )

            t0s, inj_amps, rises, decays = flare_parameters(size=int(iters*flares_per_inj),
                                                            time=inj_time[0],
                                                            amps=amps,
                                                            cut_ends=self.cut_ends)

            x = 0

            # LOOPS THROUGH HOWEVER MANY ITERATIONS SPECIFIED
            for n in range(iters):            

                x_start = x + 0
                model = np.zeros(len(self.fluxes[i]))
                for f in range(flares_per_inj):
                    m, p = flare_lightcurve(inj_time[n], t0s[x], inj_amps[x],
                                            rises[x], decays[x])
                    model = model + m + 0.0
                    x += 1

                inj_model[n] = inj_model[n] + model
                preds = self.model.predict(ids, [inj_time[n]], [inj_model[n]],
                                           [inj_errs[n]], injected=True)

                inj_preds[n] = preds + 0.0

                tab = self.identify_flare_peaks(injected=True, ids=ids,
                                                times       = [inj_time[n]],
                                                fluxes      = [inj_model[n]],
                                                flux_errs   = [inj_errs[n]],
                                                predictions = preds)

                for f in np.arange(x_start, x, 1, dtype=int):
                    subtab = tab[ (tab['tpeak'] >= inj_time[n][t0s[f]]) & 
                                  (tab['tpeak'] <= inj_time[n][t0s[f]]) ]
                    if len(subtab) > 0:
                        row = [subtab['ID'][0], subtab['tpeak'][0], subtab['amp'][0],
                                inj_amps[f], subtab['prob'][0], 1]
                    else:
                        row = [ids[0], inj_time[n][t0s[f]], 0, inj_amps[f],
                               preds[0][t0s[f]], 0]

                    inj_tab.add_row(row)

        self.inj_tab = inj_tab
        return inj_model, inj_preds
