import os, glob
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Visualize']

class Visualize(object):
    """
    Creates diagnostic plots for the neural network.
    """

    def __init__(self, cnn):
        """
        Initialized visualization class.

        Parameters
        ----------
        cnn : stella.ConvNN
        """
        self.history = cnn.history.history
        self.epochs  = cnn.epochs

        if cnn.prec_recall_curve:
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

        ax2.plot(epochs, self.history['acc'], c=train_color,
                 linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], c=val_color,
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


    def confusion_matrix(self, **kwargs):
        """
        Plots the confusion matrix of true positives,
        true negatives, false positives, and false 
        negatives.
        """
        return
