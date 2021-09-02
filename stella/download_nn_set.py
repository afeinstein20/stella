import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astroquery.vizier import Vizier
from lightkurve.search import search_lightcurve

__all__ = ['DownloadSets']

class DownloadSets(object):
    """
    Downloads the flare catalog and light curves needed
    for the CNN training, test, and validation sets.
    This class also reformats the light curves into .npy
    files and removes the associated FITS files to save
    space.
    """

    def __init__(self, fn_dir=None, flare_catalog_name=None):

        """
        Parameters
        ----------
        fn_dir : str, optional
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
        if fn_dir != None:
            self.fn_dir = fn_dir
        else:
            self.fn_dir = os.path.join(os.path.expanduser('~'), '.stella')
        
        if os.path.isdir(self.fn_dir) == False:
            os.mkdir(self.fn_dir)

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
            slc = search_lightcurve('TIC'+str(tics[i]),
                                    mission='TESS',
                                    exptime=120,
                                    sector=[1,2],
                                    author='SPOC')


            if len(slc) > 0:
                lcs = slc.download_all(download_dir=self.fn_dir)

                for j in range(len(lcs)):
                    # Default lightkurve flux = pdcsap_flux
                    lc = lcs[j].normalize()
                    
                    np.save(os.path.join(self.fn_dir, npy_name.format(tics[i], lc.sector)),
                            np.array([lc.time.value, 
                                      lc.flux.value, 
                                      lc.flux_err.value]))
                    
                    # Removes FITS files when done
                    if remove_fits == True:
                        for dp, dn, fn in os.walk(os.path.join(self.fn_dir, 'mastDownload')):
                            for file in [f for f in fn if f.endswith('.fits')]:
                                os.remove(os.path.join(dp, file))
                                os.rmdir(dp)
                
                
        if remove_fits == True:
            os.rmdir(os.path.join(self.fn_dir, 'mastDownload/TESS'))
            os.rmdir(os.path.join(self.fn_dir, 'mastDownload'))

    def download_models(self, all_models=False):
        """
        Downloads the stella pre-trained convolutional neural network
        models from MAST.

        Parameters
        ----------
        all_model : bool, optional
             Determines whether or not to return all 100 trained models
             or the 10 models used in Feinstein et al. (2020) analysis
             in the attribute `models`. Default is False.

        Attributes
        ----------
        model_dir : str
             Path to where the CNN models have been downloaded.
        models : np.array
             Array of model filenames.
        """
        hlsp_path = 'http://archive.stsci.edu/hlsps/stella/hlsp_stella_tess_ensemblemodel_all_tess_v0.1.0_bundle.tar.gz'

        new_path = os.path.join(self.fn_dir, 'models')
        
        if os.path.isdir(new_path) == False:
            os.mkdir(new_path)

        if len(os.listdir(new_path)) == 100:
            print('Models have already been downloaded to ~/.stella/models')
        
        else:
            os.system('cd {0} && curl -O -L {1}'.format(self.fn_dir, hlsp_path))
            tarball = [os.path.join(self.fn_dir, i) for i in os.listdir(self.fn_dir) if i.endswith('tar.gz')][0]
            os.system('cd {0} && tar -xzvf {1}'.format(self.fn_dir, tarball))
            
            os.system('cd {0} && mv *.h5 {1}'.format(self.fn_dir, new_path))
        

        self.model_dir = new_path
        models = np.sort([os.path.join(new_path, i) for i in os.listdir(new_path)])

        if all_models == True:
            self.models = models

        else:
            model_seeds = [4, 5, 18, 28, 29, 38, 50, 77, 78, 80]
            self.models = models[model_seeds]
