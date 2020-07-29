import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astroquery.vizier import Vizier
from lightkurve.search import search_lightcurvefile

__all__ = ['DownloadSets']

class DownloadSets(object):
    """
    Downloads the flare catalog and light curves needed
    for the CNN training, test, and validation sets.
    This class also reformats the light curves into .npy
    files and removes the associated FITS files to save
    space.
    """

    def __init__(self, fn_dir, flare_catalog_name=None):

        """
        Parameters
        ----------
        fn_dir : str
             The path to where the catalog and light
             curve files are stored.
        flare_catalog_name : str, optional
             What the flare catalog will be saved as. Default 
             is 'Guenther_2020_flare_catalog.txt'.

        Attributes
        ----------
        fn_dir : str
            Path for storing data.
        """
        self.fn_dir = fn_dir
        self.flare_table = None

        if flare_catalog_name is None:
            self.flare_catalog_name = 'Guenther_2020_flare_catalog.txt'
        else:
            self.flare_catalog_name = flare_catalog_name


    def download_catalog(self):
        """
        Downloads the flare catalog using Vizier.
        The flare catalog is named 'Guenther_2020_flare_catalog.txt'.
        The star catalog is named 'Guenther_2020_star_catalog.txt'.

        Attributes
        ----------
        flare_table : astropy.table.Table
             Flare catalog that was downloaded.
        """

        Vizier.ROW_LIMIT = -1

        catalog_list = Vizier.find_catalogs('TESS flares sectors')
        catalogs = Vizier.get_catalogs(catalog_list.keys())

        self.flare_table = catalogs[1]
        self.flare_table.rename_column('_tab2_5', 'tpeak')
        self.flare_table.write(os.path.join(self.fn_dir, self.flare_catalog_name),
                          format='csv')
        return


    def download_lightcurves(self, remove_fits=True):
        """
        Downloads light curves for the training, validation, and
        test sets. 

        Parameters
        ----------
        remove_fits : bool, optional
             Allows the user to remove the TESS light curveFITS 
             files when done. This will save space. Default is True.
        """
        if self.flare_table is None:
            self.flare_table = Table.read(os.path.join(self.fn_dir,
                                                       self.flare_catalog_name), 
                                          format='ascii')
            

        tics = np.unique(self.flare_table['TIC'])
        npy_name = '{0:09d}_sector{1:02d}.npy'

        for i in tqdm(range(len(tics))):
            slc = search_lightcurvefile('TIC'+str(tics[i]),
                                        mission='TESS',
                                        cadence='short',
                                        sector=[1,2])

            if len(slc) > 0:
                lcs = slc.download_all(download_dir=self.fn_dir)

                for j in range(len(lcs)):
                    lc = lcs[j].PDCSAP_FLUX.normalize()
                    
                    np.save(os.path.join(self.fn_dir, npy_name.format(tics[i], lc.sector)),
                            np.array([lc.time, lc.flux, lc.flux_err]))
                    
                    # Removes FITS files when done
                    if remove_fits == True:
                        for dp, dn, fn in os.walk(os.path.join(self.fn_dir, 'mastDownload')):
                            for file in [f for f in fn if f.endswith('.fits')]:
                                os.remove(os.path.join(dp, file))
                                os.rmdir(dp)
                
                
        if remove_fits == True:
            os.rmdir(os.path.join(self.fn_dir, 'mastDownload/TESS'))
            os.rmdir(os.path.join(self.fn_dir, 'mastDownload'))
