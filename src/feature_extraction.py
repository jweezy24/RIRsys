from posixpath import split
import numpy as np
import scipy.signal as sig
from utils import *
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift
from numpy.linalg import inv

#This function will split the data into Equal length segments in Time domain
# We separate the work in this way for cleanly code.
# X = input signal
# win_size = size of split 
def split_data_time(X, win_size):
    chop = len(X) -  len(X)%win_size

    return np.array(np.split(X[:chop], win_size))


#This function will split the data into Equal length segments in frequency domain
# We separate the work in this way for cleanly code.
# X = input signal
# win_size = size of split 
def split_data_freq(X, win_size):
    chop = len(X) -  len(X)%win_size
    
    X_tmp = np.array(np.split(X[:chop], win_size))

    for i in range(X_tmp.shape[0]):
        X_tmp[i] = fft(X_tmp[i])

    return X_tmp

# Takes average of vector v
def extract_average(v):
    return np.average(abs(v))

# Returns median of vector v
def extract_median(v):
    return np.median(v)

# Returns max of vector v
def extract_max(v):
    return np.max(abs(v))

# Returns power of input vector v
def extract_power(v):
    return (np.abs(fft(v))**2).mean()

#This function will extract features from a sound file.
#The file should be either an impulse response estimation.
#X = the signal input
#win_size = size of window
def extract_features(X, win_size):

    #The first thing we want to do is divide up the data into time and frequency chunks
    X_t = split_data_time(X,win_size)
    X_f = split_data_freq(X,win_size)

    feature_vector = [ [] for i in range(0,X_t.shape[0])]

    for i in range(0,X_t.shape[0]):
        row_t = X_t[i].T.flatten()
        row_f = X_f[i].T.flatten()
        
        f_t = extract_average(row_t)
        f_f = extract_average(row_f)

        feature_vector[i].append(f_t)
        feature_vector[i].append(f_f)
        
        f_t = extract_median(row_t)
        f_f = extract_median(row_f)

        feature_vector[i].append(f_t)
        feature_vector[i].append(f_f)

        f_t = extract_power(row_t)
        f_f = extract_power(row_f)

        feature_vector[i].append(f_t)
        feature_vector[i].append(f_f)

        f_t = extract_max(row_t)
        f_f = extract_max(row_f)

        feature_vector[i].append(f_t)
        feature_vector[i].append(f_f)

    return np.matrix(feature_vector)