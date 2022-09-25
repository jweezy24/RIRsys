from posixpath import split
import numpy as np
import scipy.signal as sig
import pandas as pd
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
def extract_features(X, win_size,sr):


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


#A helper function for the thread pool.
def threaded_code(f, y, ind, all_data,lock, sr=None):
    
    
    if sr != None:
        x = pd.DataFrame(f(y,sr))
    else:
        x = pd.DataFrame(f(y))
    
    with lock:
        all_data[ind].append( x )

    return 1

def librosa_spectrual_features(X, win_size,fs,feats=6, threaded=True):
    import librosa
    from multiprocessing import Pool, Manager

    #The first thing we want to do is divide up the data into time chunks
    X_t = split_data_time(X,win_size)

    #next we treat each row of the X_t matrix as its own time capture

    if threaded:
        manager = Manager()

        all_data = [ manager.list() for i in range(0,feats) ]

        lock = manager.Lock()
        l = manager.list(all_data)
        
        p = Pool(32)
        threads = []
        count = 0

        for row in X_t:


            a = p.apply_async(threaded_code, [ librosa.feature.chroma_stft , row,0,l,lock,fs ] )
            threads.append(a)
            

            a = p.apply_async(threaded_code, [ librosa.feature.chroma_cqt , row,1,l,lock,fs ] )
            threads.append(a)

            a = p.apply_async(threaded_code, [ librosa.feature.chroma_cens , row,2,l,lock,fs ] )
            threads.append(a)

            a = p.apply_async(threaded_code, [ librosa.feature.melspectrogram , row,3,l,lock,fs ] )
            threads.append(a)

            a = p.apply_async(threaded_code, [ librosa.feature.mfcc , row,4,l,lock,fs ] )
            threads.append(a)

            a = p.apply_async(threaded_code, [ librosa.feature.rms , row,5,l,lock,fs ] )
            threads.append(a)

            count +=6
        
        
        if count > 0:
            while count > 0:
                for t in threads:
                    if t.ready():
                        threads.remove(t)
                        count-=1 
                    
        p.close()
        p.terminate()
    
    else:
        
        l = [ [] for i in range(0,feats) ]

        for row in X_t:


            x = pd.DataFrame(librosa.feature.chroma_stft(y=row,sr=fs) )
            l[0].append(x)

            x = pd.DataFrame(librosa.feature.chroma_cqt(y=row,sr=fs) )
            l[1].append(x)

            x = pd.DataFrame(librosa.feature.chroma_cens(y=row,sr=fs) )
            l[2].append(x)

            x = pd.DataFrame(librosa.feature.melspectrogram(y=row,sr=fs) )
            l[3].append(x)

            x = pd.DataFrame(librosa.feature.mfcc(y=row,sr=fs) )
            l[4].append(x)

            x = pd.DataFrame(librosa.feature.rms(y=row) )
            l[5].append(x)

    return np.matrix(l,dtype=object)
    