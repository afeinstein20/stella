import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_residuals', 'plot_periodogram', 'plot_flares']

def plot_residuals(time, flux, model):
    if model is not None:
        resid = flux - model
    else:
        raise ValueError("You have no detrended flux to compare to. Try again.")
        
    plt.figure(figsize=(14,8))
    gs  = gridspec.GridSpec(3,3)
    
    ax1 = plt.subplot(gs[0:2,0:])
    ax1.set_xticks([])
    ax1.plot(x, y, 'k', linewidth=3, label='Raw', alpha=0.8)
    ax1.plot(x, model, c='orange', label='Model')
    plt.legend()
    ax2 = plt.subplot(gs[2, 0:])
    ax2.plot(x, resid, c='turquoise', linewidth=2)
    
    plt.show()


def plot_periodogram(LS_period, LS_power, save=False, tic=0, directory='.'):
    plt.plot(LS_period, LS_power, 'k', alpha=0.8)
    plt.xlabel('Period [days]')
    plt.ylabel('Lomb-Scargle Power')
    if save is False:
        plt.show()
    else:
        fn = '{}_periodogram.png'.format(tic)
        path = os.path.join(directory, fn)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=200)


def plot_flares(time, flux, flare_table, mask):
    if flare_table is None:
        return("Please call YoungStars.identify_flares() before calling this function.")

    plt.figure(figsize=(12,6))
    plt.plot(time[mask], flux[mask], c='k', alpha=0.8)

    for i,p in flare_table.iterrows():
        istart, istop = int(p.istart), int(p.istop)
        plt.plot(time[istart:istop+1], flux[istart:istop+1], '*',
                 ms=10, c='turquoise')

    plt.ylim(np.nanmin(flux[mask])-0.01, np.nanmax(flux[mask])+0.01)
    plt.xlim(np.nanmin(time[mask]), np.nanmax(time[mask]))
    plt.ylabel('Noralized Flux')
    plt.xlabel('Time (BJD - 2457000)')
    plt.tight_layout()
    plt.show()
