import os, sys
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

__all__ = ['ModelMetrics']


class ModelMetrics(object):
    """
    Calculates model metrics based on either ensemble models
    or cross validation.
    """

    def __init__(self, fn_dir, data_set='validation', mode='ensemble'):
        """
        Initializes class. Requires a directory where all
        of the files from the same model runs are saved

        Parameters
        ----------
        fn_dir : str
             Path to where all of the tables are saved.
        data_set : str, optional
             Sets which data set to look at. Default is 'validation'. 
             Other option is 'test'. DO NOT LOOK AT THE TEST SET UNTIL
             YOU ARE COMPLETELY HAPPY WITH YOUR MODEL.
        mode : str, optional
             Sets which models to calculate metrics for. Default is 
             'ensemble'. Other option is 'cross_val' for cross validation
             models.
        """
        self.dir      = fn_dir
        self.data_set = data_set
        self.mode     = mode

        
    def parse_parameters(self):
        """
        Pareses file names in fn_dir for the seed, number of epochs, 
        fractional balance, and fold (for cross validation).

        Attributes
        ----------
        seed : int, np.array
        folds : np.array
        epochs : int
        frac_balance : float
        models : np.array
        """
        if self.mode is 'cross_val':
            files = [i for i in os.listdir(self.dir) if 'crossval' in i]

        elif self.mode is 'ensemble':
            files = [i for i in os.listdir(self.dir) if 'ensemble' in i]

        # SEPARATES MODELS FROM TABLES
        models, tables = [], []
        for fn in files:
            if fn.endswith('.h5'):
                models.append(fn)
            else:
                tables.append(fn)
            

        self.seed  = seed
        self.folds = folds
        self.epochs = epochs
        self.frac_balance = frac_balance
