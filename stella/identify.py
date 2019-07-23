import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as c
from astropy.stats import sigma_clip

__all__ = ['IdentifyFlares']

class IdentifyFlares(object):
    
    def __init__(self, yso):
        self.yso = yso
        
    def pick_flux(self, detrended_flux, method):
        if detrended_flux is None:
            if (self.yso.gp_flux is not None) and (method.lower() == "gp"):
                detrended_flux = self.yso.gp_flux
            elif (self.yso.sg_flux is not None) and (method.lower() == "savitsky-golay"):
                detrended_flux = self.yso.sg_flux
            elif (detrended_flux is None) and (self.yso.sg_flux is None) and (self.yso.gp_flux is None):
                print("No detrended flux was recognized. Using normalized flux.")
                detrended_flux = self.yso.norm_flux

        return detrended_flux


    def tag_flares(self, flux, sig):
        mask = sigma_clip(flux, sigma=sig).mask
        median = np.nanmedian(flux[mask])
        isflare = np.where( (mask==True) & ( (flux-median) > 0.))[0]
        return isflare


    def identify_flares(self, detrended_flux=None, detrended_flux_err=None,
                        method="savitsky-golay", N1=3, N2=1, N3=2, sigma=2.5, minsep=3,
                        cut_ends=5, fake=False):
        """Identifies flare candidates in a given light curve.                                                                                                                         
        """
        
        detrended_flux = self.pick_flux(detrended_flux, method)

        if detrended_flux_err is None:
            detrended_flux_err = self.yso.flux_err

        columns = ['istart', 'istop', 'tstart', 'tstop',
                   'ed_rec_s', 'ed_rec_err', 'ampl_rec', 'energy_ergs']
        flares = pd.DataFrame(columns=columns)
        brks = self.yso.find_breaks()

        istart, istop = np.array([], dtype=int), np.array([], dtype=int)

        # If there are no breaks in the light curve
        if len(brks) == 0:
            brks = np.array( [np.arange(0, len(self.yso.time), 1, dtype=int)] )

        for b in brks:
            time  = self.yso.time[b]
            flux  = detrended_flux[b]
            error = detrended_flux_err[b]

            isflare = self.tag_flares(flux, sigma)
            candidates = isflare[isflare > 0]

            if len(candidates) < 1:
                if fake is False:
                    print("No flares found in ", np.nanmin(time), " - ", np.nanmax(time))
            else:
                # Find start & stop indices and combine neighboring candidates
                sep_cand = np.where(np.diff(candidates) > minsep)[0]
                istart_gap = candidates[ np.append([0], sep_cand + 1) ]
                istop_gap = candidates[ np.append(sep_cand,
                                                  [len(candidates) - 1]) ]

            # CUTS 5 DATA POINTS FROM EACH BREAK
            ends = ((istart_gap > cut_ends) & ( (istart_gap+np.nanmin(b)) < (np.nanmax(b)-cut_ends)) )
            # Ensures the start of the flare is above the median of the normalized corrected flux
            good_inds = flux[istart_gap[ends]] > np.nanmedian(flux)

            istart = np.append(istart, istart_gap[ends][good_inds] + np.nanmin(b))
            istop  = np.append(istop , istop_gap[ends][good_inds]  + np.nanmin(b) + 1)

            ed_rec, ed_rec_err = np.array([]), np.array([])
            ampl_rec = np.array([])
            for i in range(len(istart)):
                time = self.yso.time[istart[i]:istop[i]+1]
                flux = detrended_flux[istart[i]:istop[i]+1]
                err  = detrended_flux_err[istart[i]:istop[i]+1]
                ed, ed_err = self.yso.equivalent_duration(time=time, flux=flux, error=err)
                ed_rec     = np.append(ed_rec, ed)
                ed_rec_err = np.append(ed_rec_err, ed_err)
                ampl_rec   = np.append(ampl_rec, np.nanmax(flux))

        flares['istart']     = istart
        flares['istop']      = istop
        flares['ed_rec_s']   = ed_rec
        flares['ed_rec_err'] = ed_rec_err
        flares['ampl_rec']   = ampl_rec
        flares['tstart']     = self.yso.time[istart]
        flares['tstop']      = self.yso.time[istop]

        if (self.yso.lum is not None) and (type(self.yso.lum) != np.ma.core.MaskedConstant):
            energy = (flares.ed_rec_s.values * u.s) * (self.yso.lum * c.L_sun)                                                                                                       
            energy = energy.to(u.erg)
            flares['energy_ergs'] = energy.value

        return brks, flares

