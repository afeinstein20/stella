import os
import numpy as np
from tqdm import tqdm
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.table import Table
from lightkurve.lightcurve import LightCurve as LC

from stella.utils import break_rest, do_the_shuffle, split_data

__all__ = ['XOSet']

class XOSet(object):

    def __init__(self, cadences=300,
                 starry_path='/home/afeinstein/stella_exoplanets/starry_models', 
                 batman_path='/home/afeinstein/stella_exoplanets/batman_models', 
                 bkg_path='/home/afeinstein/stella_exoplanets/bkglc', 
                 frac_balance=0.0, training=0.8, validation=0.9):
        """
        Parameters
        ----------
        cadences : int, optional
           The number of cadences to go into each example.
        starry_path : str
           The path to where the starry spot models are stored.
        batman_path : str
           The path to where the batman transit models are stored.
        bkg_path : str
           The path to where the background light curves are stored.
        frac_balance : float, optional
        traininig : float, optional
        validation : float, optional
        """

        self.cadences    = cadences
        self.starry_path = starry_path
        self.batman_path = batman_path
        self.bkg_path    = bkg_path

        self.import_starry()
        self.import_transits()
        self.import_backgrounds()
        self.combine()
        self.split(frac_balance=frac_balance, training=training, validation=validation)

    def import_starry(self):
        """
        Imports the starry models used to inject into the training, validation,
        and test sets.

        Attributes
        ----------
        model_cutouts : np.ndarray
        spot_ids : np.ndarray
        spot_table : astropy.table.Table
        """
        step = int((22*u.day).to(u.min).value/10)
        time = np.linspace(0,48,step) * u.day
        spot_time = time.value
        
        model_fns = os.listdir(self.starry_path)
        
        spot_table = Table(names=['model', 'inc', 'Prot', 'nspots', 'lat', 
                                  'lon', 'intensity', 'spotsize'],
                           dtype=[np.int64, np.float64, np.float64, np.int64, 
                                  np.ndarray, np.ndarray, np.ndarray, np.ndarray])
        
        model_cutouts = np.zeros( (415000, self.cadences) )
        spot_ids = np.zeros( len(model_cutouts), dtype='U20')
        
        x = 0
        
        for i in tqdm(range(len(model_fns))):
            
            data = np.load(os.path.join(self.starry_path, model_fns[i]), allow_pickle=True)
            dict_values = [i]
            for key in list(data[0].keys()):
                dict_values.append(data[0][key])
            spot_table.add_row(dict_values)

            y = 0
            per_model = int(len(data[1])/self.cadences)
            
            for j in range(per_model):
                model_cutouts[x] = data[1][y:y+self.cadences]-np.nanmedian(data[1][y:y+self.cadences])
                spot_ids[x] = 'spot{0:04d}'.format(i)
                x += 1
                model_cutouts[x] = np.flip(model_cutouts[x-1])
                spot_ids[x] = 'spot{0:04d}'.format(i)
                x += 1
                y += self.cadences
                
        model_cutouts = np.delete(model_cutouts, np.arange(x, len(model_cutouts),1,dtype=int), axis=0)
        spot_ids = np.delete(spot_ids, np.arange(x,len(spot_ids),1,dtype=int))
        
        self.model_cutouts = model_cutouts
        self.spot_ids = spot_ids
        self.spot_table = spot_table
        

    def import_backgrounds(self):
        """
        Imports background light curves and scales to general noise of TESS.

        Attributes
        ----------
        backgrounds : np.ndarray
        bkg_ids : np.ndarray
        """
        bkg_fns = os.listdir(self.bkg_path)
        
        backgrounds = np.zeros((len(bkg_fns)*30, self.cadences))
        bkg_ids = np.zeros(len(bkg_fns)*30, dtype='U10')

        x = 0

        for i in tqdm(range(len(bkg_fns))):
            dat = np.load(os.path.join(self.bkg_path, bkg_fns[i]))
            
            set1 = np.arange(0,len(dat)/2-self.cadences/2,1,dtype=int)
            set2 = np.arange(len(dat)/2+self.cadences/2, len(dat), 1, dtype=int)
            
            for s in [set1, set2]:
                per_model = int(len(s)/self.cadences)
                y = 0
                for j in range(per_model):
                    bkg = dat[s][y:y+self.cadences]-np.nanmedian(dat[s][y:y+self.cadences]) + 0.0
                    
                if len(np.where(np.isnan(bkg)==True)[0]) < 1:
                    backgrounds[x] = bkg
                    bkg_ids[x] = 'bkg{0:05d}'.format(i)
                    x += 1
                    y += self.cadences

        remove = np.where(bkg_ids=='')[0]
        backgrounds = np.delete(backgrounds, remove[:-1], axis=0)
        bkg_ids = np.delete(bkg_ids, remove[:-1])

        self.backgrounds = backgrounds
        self.bkg_ids = bkg_ids


    def import_transits(self):
        """
        Imports the transit models to inject.

        Attributes
        ---------
        transit_models : np.ndarray
        transit_ids : np.ndarray
        transit_table : astropy.table.Table
        """
        model_fns = np.sort([os.path.join(self.batman_path, i) for i in os.listdir(self.batman_path)]) 
        transit_models = np.zeros((len(model_fns), self.cadences))
        transit_ids = np.zeros(len(model_fns), dtype=int)
        
        transit_table = Table(names=['id','rstar', 'per', 'rprstar', 
                                     'arstar', 'inc', 'ecc', 
                                     'w', 'u1', 'u2'])
        
        for i in tqdm(range(len(model_fns))):
            data = np.load(model_fns[i], allow_pickle=True)
            transit_models[i] = data[0]
            transit_ids[i] = model_fns[i].split('_')[-1][:-4]
            transit_table.add_row(data[1])
            
        self.transit_models = transit_models
        self.transit_ids = transit_ids
        self.transit_table = transit_table

    def combine(self):
        """
        Combines the spot model, transit model, and background noise for the training set.

        Attributes
        ---------
        dataset : np.ndarray
           All examples.
        ids : np.ndarray
           Array of identifiers for what spot model, transit model, and 
           background went into that example.
        labels : np.ndarray
           Binary labels for which examples have transits (1) or not (0).
        """
        diff = np.diff(self.backgrounds/500, axis=1)
        maxd = np.nanmax(diff, axis=1)
        backgrounds = self.backgrounds[maxd<0.1] + 0.0

        bkg_std = np.nanstd(backgrounds/500, axis=1)
        trn_std = np.nanstd(self.transit_models,  axis=1)

        pt = 8
        pb = 4

        totalset = int(len(trn_std)*pt*pb)
        DATASET = np.zeros((totalset, self.cadences))
        LABELS = np.zeros(totalset, dtype=int)
        
        np.random.seed(456)
        b_tracker = np.zeros(totalset, dtype='U30')
        m_tracker = np.zeros(totalset, dtype='U30')
        t_tracker = np.full(totalset, 'None', dtype='U30')
        
        x = 0

        for i in tqdm(range(len(self.transit_models))):
            bind = np.where(bkg_std < trn_std[i])[0]
            
            rand = np.random.randint(0,len(bind),pt*pb)
            b = self.backgrounds[bind[rand]]
            b_tracker[x:x+pb*pt] = self.bkg_ids[bind[rand]]
            
            mrand = np.random.randint(0,len(self.model_cutouts), pt*pb)
            m = self.model_cutouts[mrand]
            m_tracker[x:x+pb*pt] = self.spot_ids[mrand]
            
            which_t = np.random.choice(np.arange(0,pt*pb,1), pt, replace=False)
            tinds = np.arange(x, x+pb*pt,1)[which_t]
            t_tracker[tinds] = self.transit_ids[i]
            
            DATASET[x:x+pb*pt] = self.backgrounds[bind[rand]]/500+self.model_cutouts[mrand]
            DATASET[tinds] += self.transit_models[i]
            LABELS[tinds] = 1
    
            x += (pb*pt)

        id_tab = Table(names=['bkg', 'spot', 'transit'],
                       dtype=['U40', 'U40', 'U40'],
                       data=[b_tracker, m_tracker, 
                             t_tracker])

        IDS = np.zeros(len(DATASET), dtype='U50')
        for i in range(len(DATASET)):
            IDS[i] = '_'.join(str(e) for e in id_tab[i])

        self.dataset = DATASET
        self.labels = LABELS
        self.ids = IDS
        self.id_table = id_tab
        self.m_tracker = m_tracker

    def split(self, frac_balance=0.73, training=0.80, validation=0.90):
        """
        Splits the data into the training, validation, and test sets.

        Parameters
        ----------
        frac_balance : float, optional
           The amount of negative classes to remove. Default = 0.73.
        training : float, optional
           Assigns the percentage of the training set data for the 
           model. Default is 80%.
        validation : float, optional
           Assigns the percentage of the validation and testing set
           data for the model Default is 90% (i.e. 10% in the validation
           set and 10% in the test set).

        Attributes
        ----------
        train_data : np.ndarray
           The training data.
        train_labels : np.ndarray
           The binary labels for the training data.
        val_data : np.ndarray
           The validation data.
        val_labels : np.ndarray
           The binary labels for the validation data.
        val_ids : np.ndarray
           The ID labels for the validation data.
        test_data : np.ndarray
           The test data.
        test_labels : np.ndarray
           The labels for the test data.
        test_ids : np.ndarray
           The ID labels for the test data.
        """

        SHUFFLE_IDS, SHUFFLE_MATRIX, SHUFFLE_LABELS, SHUFFLE_MODELS = do_the_shuffle(self.dataset,
                                                                                     self.labels, 
                                                                                     self.m_tracker,
                                                                                     self.ids,
                                                                                     frac_balance)
        
        misc = split_data(SHUFFLE_LABELS,
                          SHUFFLE_MATRIX,
                          SHUFFLE_IDS,
                          SHUFFLE_MODELS,
                          training=training,
                          validation=validation)

        self.train_data = misc[0]
        self.train_labels = misc[1]
        self.val_data = misc[2]
        self.val_labels = misc[3]
        self.val_ids = misc[4]
        self.test_data = misc[5]
        self.test_labels = misc[6]
        self.test_ids = misc[7]
        
