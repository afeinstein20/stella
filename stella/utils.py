import numpy as np
from astropy import units as uni
from scipy.interpolate import interp1d

def flare_lightcurve(time, t0, amp, rise, fall, y=None):
    """
    Generates a simple flare model with a Gaussian rise and an 
    exponential decay.

    Parameters
    ----------
    time : np.ndarray
         A time array.
    amp : float
         The amplitude of the flare.
    t0 : int
         The index in the time array where the flare will occur.
    rise : float
         The Gaussian rise of the flare.
    fall : float
         The exponential decay of the flare.
    y : np.ndarray, optional
         Underlying stellar activity. Default if None.

    Returns
    ----------
    flare_model : np.ndarray
         A light curve of zeros with an injected flare of given parameters
    row : np.ndarray
         The parameters of the injected flare. Returns - 
         [t0, amplitude, duration, gauss_rise, exp_decay].
    """
    def gauss_rise(time, flux, amp, t0, rise):
        return amp * np.exp( -(time - t0)**2.0 / (2.0*rise**2.0) ) + flux
    
    def exp_decay(time, flux, amp, t0, fall):
        return amp * np.exp( -(time - t0) / fall ) + flux

    growth = np.where(time <= time[t0])[0]
    decay  = np.where(time >  time[t0])[0]

    if y is None:
        y = np.ones(len(time))

    growth_model = gauss_rise(time[growth], y[growth], amp, time[t0], rise)
    decay_model  = exp_decay(time[decay]  , y[decay] , amp, time[t0], fall)

    model = np.append(growth_model, decay_model)
    
    return model, np.array([time[t0], amp, 0, rise, fall])


def flare_parameters(size, time, amps, cut_ends=30):
    """
    Generates an array of random amplitudes at different times with
    different rise and decay properties.

    Parameters
    ----------
    size : int
         The number of flares to generate.
    times : np.array
         Array of times where a random subset will be chosen for flare
         injection. 
    amps : list
         List of minimum and maximum of flare amplitudes to draw from a 
         normal distribution. 
    cut_ends : int, optional
         Number of cadences to cut from the ends of the light curve.
         Default is 30.
    
    Returns
    ----------
    flare_t0s : np.ndarray
         The distribution of flare start time indices.
    flare_amps : np.ndarray
         The distribution of flare amplitudes.
    flare_rises : np.ndarray
         The distribution of flare rise rates.
    flare_decays : np.ndarray
         The distribution of flare decays rates.
    """
    # CHOOSES UNIQUE TIMES FOR INJ-REC PURPOSES
    randtimes   = np.random.randint(cut_ends, len(time)-cut_ends, size*2)
    randtimes   = np.unique(randtimes)
    randind     = np.random.randint(0, len(randtimes), size)
    randtimes   = randtimes[randind]

    flare_amps  = np.random.uniform(amps[0], amps[1], size)
    flare_rises = np.random.uniform(0.00005,  0.0002,  size)

    # Relation between amplitude and decay time
    flare_decays = np.random.uniform(0.0003, 0.004, size)

    return randtimes, flare_amps, flare_rises, flare_decays


def break_rest(time, flux, flux_err, cadences):
    """
    Breaks up the non-signal cases into bite-sized cadence-length chunks.

    Parameters
    ----------
    time : np.ndarray
         Array of time.
    flux : np.ndarray
         Array of fluxes.
    flux_err : np.ndarray
         Array of flux errors.
    cadences : int
         Number of cadences for the training-validation-test set.

    Returns
    -------
    time : np.ndarray
         Array of times without the signal.
    flux : np.ndarray
         Array of fluxes without the signal.
    err : np.ndarray
         Array of flux errors without the signal.
    """
    # BREAKING UP REST OF LIGHT CURVE INTO CADENCE SIZED BITES                                                        
    diff = np.diff(time)
    breaking_points = np.where(diff > (np.median(diff) + 1.5*np.std(diff)))[0]
    
    tot = 100
    ss  = 1000
    nonflare_time = np.zeros((ss, cadences))
    nonflare_flux = np.zeros((ss, cadences))
    nonflare_err = np.zeros((ss, cadences))
    
    x = 0
    for j in range(len(breaking_points)+1):
        if j == 0:
            start = 0
            end = breaking_points[j]
        elif j < len(breaking_points):
            start = breaking_points[j-1]
            end = breaking_points[j]
        else:
            start = breaking_points[-1]
            end = len(time)

        if np.abs(end-start) > (2*cadences):
            broken_time = time[start:end]
            broken_flux = flux[start:end]
            broken_err  = flux_err[start:end]

            # DIVIDE LIGHTCURVE INTO EVEN BINS                                                                        
            c = 0
            while (len(broken_time) - c) % cadences != 0:
                c += 1

            # REMOVING CADENCES TO BIN EVENLY INTO CADENCES                                                           
            temp_time = np.delete(broken_time, np.arange(len(broken_time)-c,
                                                         len(broken_time), 1, dtype=int) )
            temp_flux = np.delete(broken_flux, np.arange(len(broken_flux)-c,
                                                         len(broken_flux), 1, dtype=int) )
            temp_err = np.delete(broken_err, np.arange(len(broken_err)-c,
                                                       len(broken_err), 1, dtype=int) )

            # RESHAPE ARRAY FOR INPUT INTO MATRIX                                                                     
            temp_time = np.reshape(temp_time,
                                   (int(len(temp_time) / cadences), cadences) )
            temp_flux = np.reshape(temp_flux,
                                   (int(len(temp_flux) / cadences), cadences) )
            temp_err  = np.reshape(temp_err,
                                   (int(len(temp_err) / cadences), cadences) )
            
            # APPENDS TO BIGGER MATRIX                                                                                
            for f in range(len(temp_flux)):
                if x >= ss:
                    break
                else:
                    nonflare_time[x] = temp_time[f]
                    nonflare_flux[x] = temp_flux[f]
                    nonflare_err[x] = temp_err[f]
                    x += 1

    nonflare_time = np.delete(nonflare_time, np.arange(x, ss, 1, dtype=int), axis=0)
    nonflare_flux = np.delete(nonflare_flux, np.arange(x, ss, 1, dtype=int), axis=0)
    nonflare_err  = np.delete(nonflare_err,  np.arange(x, ss, 1, dtype=int), axis=0)
    
    return nonflare_time, nonflare_flux, nonflare_err
    
                            
def do_the_shuffle(training_matrix, labels, training_other, training_ids, frac_balance):
    """
    Shuffles the data in a random order and fixes data inbalance based on
    frac_balance.
    
    Parameters
    ----------
    training_matrix : np.ndarray
         Array of data sets for training-validation-test sets.
    labels : np.array
         Array of 0 or 1 labels for the training_matrix.
    training_other : np.array
    training_ids : np.array
    frac_balance : float
    
    Returns
    -------
    """
    np.random.seed(321)
    ind_shuffle = np.random.permutation(training_matrix.shape[0])
    
    labels2 = np.copy(labels[ind_shuffle])
    matrix2 = np.copy(training_matrix[ind_shuffle])
    other2  = np.copy(training_other[ind_shuffle])
    ids2    = np.copy(training_ids[ind_shuffle])
    
    # INDEX OF NEGATIVE CLASS                                                                                         
    ind_nc = np.where(labels2 == 0)

    # RANDOMIZE INDEXES                                                                                               
    np.random.seed(123)
    ind_nc_rand = np.random.permutation(ind_nc[0])
    
    # REMOVE FRAC_BALANCE% OF NEGATIVE CLASS                                                                          
    length = int(frac_balance * len(ind_nc_rand))

    newlabels = np.delete(labels2, ind_nc_rand[0:length])
    newtraining_other  = np.delete(other2 , ind_nc_rand[0:length])
    newtraining_ids    = np.delete(ids2   , ind_nc_rand[0:length])
    newtraining_matrix = np.delete(matrix2, ind_nc_rand[0:length], axis=0)

    ind_pc = np.where(newlabels==1)
    ind_nc = np.where(newlabels==0)
    print("{} positive classes (flare)".format(len(ind_pc[0])))
    print("{} negative classes (no flare)".format(len(ind_nc[0])))
    print("{}% class imbalance\n".format(np.round(100 * len(ind_pc[0]) / len(ind_nc[0]))))
    
    return newtraining_ids, newtraining_matrix, newlabels, newtraining_other


def split_data(labels, training_matrix, ids, other, training, validation):
    """
    Splits the data matrix into a training, validation, and testing set.

    Parameters
    ----------
    labels : np.array
         Array of labels for each data row.
    training_matrix : np.ndarray
         Array of training-validation-test data.
    ids : np.array
         Array of identifiers for the light curves.
    other : np.array
         Array of signals (for flares -- tpeak; for transits -- SNR).
    training : float
         How much of the data should be in the training set.
    validatoin : float
        How much of the data should be in the validation & test set.

    Returns 
    -------
    x_train : np.ndarray
    y_train : np.narray
    x_val : np.ndarray
    y_val : np.narray
    val_ids : np.array
    val_other : np.array
    x_test : np.ndarray
    y_test : np.array
    test_ids : np.array
    test_other : np.array
    """
    train_cutoff = int(training * len(labels))
    val_cutoff   = int(validation * len(labels))
    
    x_train = training_matrix[0:train_cutoff]
    y_train = labels[0:train_cutoff]
    
    x_val = training_matrix[train_cutoff:val_cutoff]
    y_val = labels[train_cutoff:val_cutoff]
    
    x_test = training_matrix[val_cutoff:]
    y_test = labels[val_cutoff:]
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val   = x_val.reshape(x_val.shape[0], x_train.shape[1], 1)
    x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    test_ids   = ids[val_cutoff:]
    test_other = other[val_cutoff:]
    
    val_ids   = ids[train_cutoff:val_cutoff]
    val_other = other[train_cutoff:val_cutoff]
    
    return x_train, y_train, x_val, y_val, val_ids, val_other, x_test, y_test, test_ids, test_other
