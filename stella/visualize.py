import os, glob
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

__all__ = ['Visualize']

class Visualize(object):
    """
    Creates diagnostic plots for the neural network.
    """

    def __init__(self, cnn, set='validation'):
        """
        Initialized visualization class.

        Parameters
        ----------
        cnn : stella.ConvNN
        set : str, optional
             An option to view the results of the
             validation set or the testing set. The
             testing set should only be looked at at
             the very end of creating, training, and 
             testing the network using the validation set.
             Default is 'validation'. The alternative 
             option is 'test'.
        """
        self.cnn = cnn
        self.set = set

        if set.lower() == 'validation':
            self.data_set = cnn.val_data
        if set.lower() == 'test':
            self.data_set = cnn.test_data

        if cnn.history is not None:
            self.history = cnn.history.history
        if cnn.history_table is not None:
            self.history_table = cnn.history_table

        self.epochs  = cnn.epochs

        if cnn.prec_recall_curve is not None:
            self.prec_recall = cnn.prec_recall_curve
        else:
            self.prec_recall = None


    def loss_acc(self, train_color='k', val_color='darkorange'):
        """
        Plots the loss & accuracy curves for the training
        and validation sets.

        Parameters
        ----------
        train_color : str, optional
             Color to plot the training set in. Default is black.
        val_color : str, optional
             Color to plot the validation set in. Default is
             dark orange.
        """
        epochs = np.arange(0, self.epochs, 1)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,4))

        ax1.plot(epochs, self.history['loss'], c=train_color,
                 linewidth=2, label='Training')
        ax1.plot(epochs, self.history['val_loss'], c=val_color,
                  linewidth=2, label='Validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(epochs, self.history['accuracy'], c=train_color,
                 linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], c=val_color,
                 linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        
        plt.subplots_adjust()

        return fig

    
    def precision_recall(self, **kwargs):
        """
        Plots the ensemble-averaged precision recall metric.

        Parameters
        ----------
        **kwargs : dictionary, optional
             Dictionary of parameters to pass into matplotlib.
        """
        fig = plt.figure(figsize=(8,5))

        plt.plot(self.prec_recall[0], self.prec_recall[1], **kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        return fig


    def confusion_matrix(self, threshold=0.5, colormap='inferno'):
        """
        Plots the confusion matrix of true positives,
        true negatives, false positives, and false 
        negatives.

        Parameters
        ----------
        threshold : float, optional
             Defines the threshold for positive vs. negative cases.
             Default is 0.5 (50%).
        colormap : str, optional
             Colormap to draw colors from to plot the light curves
             on the confusion matrix. Default is 'inferno'.
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
        df = self.cnn.create_df(threshold, mode="confusion", data_set=self.set)
        x_val = self.data_set + 0.0

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
