import os, sys
import numpy as np
from pylab import *
from astropy.table import Table, Column
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

    def __init__(self, fn_dir, mode='ensemble'):
        """
        Initializes class. Requires a directory where all
        of the files from the same model runs are saved

        Parameters
        ----------
        fn_dir : str
             Path to where all of the tables are saved.
        mode : str, optional
             Sets which models to calculate metrics for. Default is 
             'ensemble'. Other option is 'cross_val' for cross validation
             models.
        """
        self.dir       = fn_dir
        self.mode      = mode

        self.load_data()

        if mode is 'ensemble':
            self.ensemble_average()

        
    def load_data(self):
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
        df = os.listdir(self.dir)

        if self.mode is 'cross_val':
            files = np.sort([i for i in df if 'crossval' in i])
            folds = []

        elif self.mode is 'ensemble':
            files = np.sort([i for i in df if 'ensemble' in i])
            seeds = []

        self.models  = [i for i in files if i.endswith('.h5')]
        predval = [i for i in files if 'predval' in i][0]
        history = [i for i in files if 'histories' in i][0]
        try:
            predtest = [i for i in files if 'predtest' in i][0]
            self.predtest_table = Table.read(os.path.join(self.dir, predtest),
                                             format='ascii')
        except:
            self.predtest_table = None
            print("No predictions on test set available.")

        parsing     = self.models[0].split('_')
        self.seeds  = int(parsing[1].split('s')[1])
        self.epochs = int(parsing[2].split('i')[1])
        self.frac_balance = float(parsing[3].split('b')[1][0:4])
        
        for m in self.models:
            if self.mode is 'cross_val':
                f = int(m.split('_')[4].split('.')[0][1:])
                folds.append(f)
                self.folds = folds
            if self.mode is 'ensemble':
                s = int(m.split('_')[1].split('s')[1])
                seeds.append(s)
                self.seeds = seeds

        self.predval_table = Table.read(os.path.join(self.dir, predval), format='ascii')
        self.history_table = Table.read(os.path.join(self.dir, history), format='ascii')


    def ensemble_average(self):
        """
        Creates an average prediction column in the predval and predtest 
        tables if mode = 'ensemble' and there is more than 1 model to evaluate.
        Else, the average prediction column is the same as the prediction
        column in the table.
        """
        mean_arr = []
        
        colnames = [i for i in self.predval_table.colnames if 'pred' in i]
        for cn in colnames:
            mean_arr.append(np.round(self.predval_table[cn].data, 3))
        self.predval_table.add_column(Column(np.nanmean(mean_arr, axis=0),
                                             name='mean_pred'))

        if self.predtest_table is not None:
            mean_arr = []
            colnames = [i for i in self.predtest_table.colnames if 'pred' in i]
            for cn in colnames:
                mean_arr.append(np.round(self.predtest_table[cn].data, 3))
            self.predtest_table.add_column(Column(np.nanmean(mean_arr, axis=0),
                                                  name='mean_pred'))


    def pred_round(self, table, threshold):
        """ Rounds the average prediction based on a threshold. """
        pr = np.zeros(len(table))
        pr[table['mean_pred'].data >= threshold] = 1
        pr[table['mean_pred'].data <  threshold] = 0
        table.add_column(Column(pr, name='pred_round'))
        return table


    def calculate_metrics(self, threshold=0.5, data_set='validation'):
        """
        Calculates average precision, accuracy, recall, and precision-recall
        curve for flares above a given threshold value.
        
        Parameters
        ----------
        threshold : float, optional
             The value above which something is considered a flare.
             Default is 0.5.
        data_set : str, optional
             Sets which data set to look at. Default is 'validation'.
             Other option is 'test'. DO NOT LOOK AT THE TEST SET UNTIL
             YOU ARE COMPLETELY HAPPY WITH YOUR MODEL.

        Attributes
        ----------
        average_precision : float
        accuracy : float
        recall_score : float
        precision_score : float
        prec_recall_curve : np.array
             2D array of precision and recall for plotting purposes.
        """
        if data_set is 'validation':
            table = self.predval_table
        elif data_set is 'test':
            if self.predtest_table is not None:
                table = self.predtest_table
            else:
                raise ValueError("No test set predictions found.")

        # SETS ARRS FOR AVG PRECISION, ACC., RECALL SCORE, PREC. SCORE
        ap, ac, rs, ps = [], [], [], []
        p_cur, r_cur = [], []

        # SETS KEYS TO LOOK FOR IN TABLE FOR EITHER METHOD
        if self.mode is 'ensemble':
            gt  = table['gt'].data
            key = 'pred'
            table = self.pred_round(table, threshold)
            self.predval_table = table

        elif self.mode is 'cross_val':
            gt  = None
            key = 'pred_f'

        for i, val in enumerate([i for i in table.colnames if key in i]):
            if self.mode is 'cross_val':
                gt_key = 'gt_' + val.split('_')[1]
                gt = table[gt_key].data

            # CALCULATES AVERAGE PRECISION SCORE
            ap.append( np.round( average_precision_score(gt,
                                                         table[val].data,
                                                         average=None), 4))
            # ROUNDED BASED ON THRESHOLD
            arr = np.copy(table[val].data)
            arr[arr >= threshold] = 1.0
            arr[arr <  threshold] = 0.0

            # CALCULATES ACCURACY
            ac.append( np.round( np.sum(arr == gt) / len(table), 4))

            if self.mode is 'cross_val':
                # CALCULATES RECALL SCORE
                rs.append( np.round( recall_score(gt, arr), 4))
            
                # CALCULATES PRECISION SCORE
                ps.append( np.round( precision_score(gt, arr), 4))

                # CREATES PRECISION RECALL CURVE
                prec_curve, rec_curve, _ = precision_recall_curve(gt, table[val].data)
                p_cur.append(prec_curve)
                r_cur.append(rec_curve)

        if self.mode is 'ensemble':
            rs = np.round( recall_score( gt, table['pred_round']), 4)
            ps = np.round( precision_score( gt, table['pred_round']), 4)
            p_cur, r_cur, _ = precision_recall_curve(gt, table['mean_pred'].data)

            self.average_precision = ap[-1]
            self.accuracy = ac[-1]

        else:
            self.average_precision = ap
            self.accuracy = ac

        self.recall_score = rs
        self.precision_score = ps
        self.prec_recall_curve = np.array([r_cur, p_cur])

    
    def confusion_matrix(self, ds, threshold=0.5, colormap='inferno', 
                         data_set='validation'):
        """
        Plots the confusion matrix of true positives,
        true negatives, false positives, and false
        negatives.

        Parameters
        ----------
        ds : stella.DataSet
             Object needed to look at light curves from the validation
             or the test set.
        threshold : float, optional
             Defines the threshold for positive vs. negative cases.
             Default is 0.5 (50%).
        colormap : str, optional
             Colormap to draw colors from to plot the light curves
             on the confusion matrix. Default is 'inferno'.
         data_set : str, optional
             Sets which data set to look at. Default is 'validation'.
             Other option is 'test'. DO NOT LOOK AT THE TEST SET UNTIL
             YOU ARE COMPLETELY HAPPY WITH YOUR MODEL. 
        """
        # GETS THE COLORS FOR PLOTTING
        cmap = cm.get_cmap(colormap, 15)
        colors = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            colors.append(matplotlib.colors.rgb2hex(rgb))
        colors = np.array(colors)

        # PLOTTING NORMALIZED LIGHT CURVE TO GIVEN SUBPLOT
        def plot_lc(data, ind, ax, color, offset):
            """ Plots the light curve on a given axis. """
            ax.set_xlim(0,200)
            ax.set_ylim(-3,3.5)
            ax.axvline(100, linestyle='dotted', color='gray',
                       linewidth=0.5)
            ax.set_yticks([])
            ax.set_xticks([])

            # NORMALIZING FLUX TO PEAK
            lc = data[ind] - np.nanmedian(data[ind])
            lc /= np.abs(np.nanmax(lc, axis=0))
            lc += offset

            ax.plot(lc, color=color, linewidth=2.5)
            return ax

        # GETS THE TABLE & VALIDATION DATA FOR THE MATRIX
        if data_set == 'validation':
            df = self.predval_table
            x_val = ds.val_data
        elif data_set == 'test':
            df = self.predtest_table
            x_val = ds.test_data

        # INDICES FOR THE CONFUSION MATRIX
        ind_tn = np.where( (df['pred_round'] == 0) & (df['gt'] == 0) )[0]
        ind_fn = np.where( (df['pred_round'] == 0) & (df['gt'] == 1) )[0]
        ind_tp = np.where( (df['pred_round'] == 1) & (df['gt'] == 1) )[0]
        ind_fp = np.where( (df['pred_round'] == 1) & (df['gt'] == 0) )[0]

        order = [ind_tn, ind_fp, ind_fn, ind_tp]
        titles = ['True Negatives', 'False Positives',
                  'False Negatives', 'True Positives']
        shifts = [-2, 0, 2]

        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,8))

        i = 0

        for ax in axes.reshape(-1):
            inds = order[i]
            which = np.random.randint(0,len(inds),3)

            for j in range(3):
                ax = plot_lc(x_val, inds[which[j]], ax, colors[j*2+1],
                             shifts[j])

            ax.set_title(titles[i], fontsize=20)

            if titles[i] == 'False Positives' or titles[i] == 'False Negatives':
                ax.set_facecolor('lightgray')

            i += 1

        return fig
