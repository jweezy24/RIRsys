import numpy as np
import scipy.signal as sig
from utils import *
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift
from numpy.linalg import inv

# ALL FUNCTIONS RETURN DATA WITHIN THE TIME DOMAIN.

# x = input
# y = output
# fs = sampling rate of the signal
# Resource: http://www.ece.tufts.edu/~maivu/ES150/8-lti_psd.pdf
# Returns a single dimension impulse response estimation H. 
def estimate_IR_power_spectrum(x,y, fs=48000):
    #Compute the cross spectrual density of x and y
    S_yx = sig.csd(x,y,fs=fs)[1]


    #compute the power spectrual density of x and y
    S_X = sig.periodogram(x,fs=fs,scaling='density')[1]
    S_Y = sig.periodogram(y,fs=fs,scaling='density')[1]

    #Need to append zeros to the CSD

    #These are the differences in length
    zeros1 = abs(len(S_yx) - len(S_X))
    zeros2 = abs(len(S_yx) - len(S_Y))

    #Append zeros to the csd array
    S_yx1 = np.concatenate( (S_yx, np.array([0 for i in range(zeros1)]))) 
    S_yx2 = np.concatenate( (S_yx, np.array([0.000001 for i in range(zeros2)]))) 

    #Estimate transfer function conjugate
    H_conj = S_yx1/S_X
    #Estimate transfer function
    H = S_Y/S_yx2

    #Estimate Impulse response
    h = abs(irfft(H))

    return h


# d = input
# x = output
# L = segement length
# Paper for reference: https://ieeexplore.ieee.org/document/8369106
# Returns a minimal-error IR matrix where each row is a separate impulse. 
def estimate_IR_kronecker_product(d,x,L):

    d = d.copy()
    x = x.copy()

    #first thing we want to do is break up d and x into L samples
    L = nextpow2(L)
    L2 =  int(len(d) / L)
    chop = abs(L2*L- len(d))
    chop = len(d) - chop
    
    prod = L2*L

    assert(len(d[:chop]) == prod) 
    assert(len(x[:chop]) == prod) 
    
    d = np.array(np.split(d[:chop],L))
    x = np.array(np.split(x[:chop],L))

    #init all variables

    #Matrix that will contain the weiner filters that are generated each impulse response.
    H_weiner = []
    #Matrix of impulse response estimations.
    H = np.array([])
    #Matrix of minimized estimations of each impulse responses.
    H_min = []
    #Matrix of error values for each impulse response row.
    error_matrix = np.array([])


    for i in range(d.shape[0]):
        
        # Cast row to matrix object so that the dimensions are (n,1) rather than (n,) this will casue bugs when transposing 
        row_d = np.matrix(d[i,:])
        row_x = np.matrix(x[i,:])

        #This is so we skip the zeros from the first two seconds of readings.
        n_zeros = np.count_nonzero(row_x==0)

        #If there are a lot of zeros nans show up in later calculations so we continue if there are too many zeros.
        if n_zeros >= (len(row_d)/2):
            continue

        #Calculate ffts for each row
        X = rfft(row_x)
        D = rfft(row_d)

        #Estimate SNR for each row
        lambd_est_d = signaltonoise(d[i,:]) 
        lambd_est_x = signaltonoise(x[i,:])

        #Deconvolve output with input
        h_i =  np.matrix(irfft( (1/len(row_x)) * ((D * X.conj())/(X* X.conj() + lambd_est_x**2))).real).T

        #Check for nans, if there are nan values in the resulting matrix calculations will fail.
        if(np.isnan(h_i).any()):
            continue

        #This conditional is here as an initator for the numpy matrix H.
        #This could maybe be removed.
        if len(H) == 0:
            H = h_i
        else:
            H = np.hstack((H,h_i))
        
        #Calculate estimated input with our estimated h 
        y_i = np.dot(h_i.T,row_x.T)

        #calculate error 
        e_t = row_d - y_i
        
        #This conditional is here as an initator for the numpy matrix error_matrix.
        #This could maybe be removed.
        if len(error_matrix) == 0:
            error_matrix = np.matrix(e_t)
        else:
            error_matrix = np.vstack((error_matrix, e_t))

        #Calculate cross correlation of the rows
        p = np.matrix(compute_correlation_function(d[i,:],x[i,:])[0]).T

        #calculate covariance matrix of the square matrix of x
        R = np.matrix(np.cov(np.dot(row_x.T,row_x)))

        #Calculate variance of d
        sigma = np.var(row_d)

        b = np.dot(h_i.T,p)
        a = np.dot(h_i.T, R)
        a = np.dot(a,h_i)

        minimized_h = sigma**2 - 2*b + a
        
        # Grab weiner iterative weiner filters
        # Removed due to some matrixes having a det of 0

        # weiner_filter_i = np.dot(np.linalg.inv(R), p)
        # H_weiner.append(weiner_filter_i)
    
    H = np.matrix(H)

    U1,S,U2 = np.linalg.svd(H)


    #Knocker product 
    for i in range(S.shape[0]):
        #Estimation comes from paper.
        h1 = np.sqrt(S[i]) * U1[:,i]
        h2 = np.sqrt(S[i]) * U2[:,i]
        
        #This is so that the array is in the format described in the paper.
        # Removed so that the final impulse response is 1 dimensional 
        h2 = h2.T

        #This conditional is here as an initator for the numpy matrix H_min.
        #This could maybe be removed.
        if len(H_min) == 0:
            H_min = np.matrix(np.kron(h1,h2))
        else:
            H_min += np.kron(h1,h2)

    H_min = normalize(H_min)
    
    return H_min  


# x = input
# y = output
# Returns a IR estimation using weiner deconvolution.
def estimate_IR_weiner_deconvolution(x_c,y_c):

    x = x_c.copy()
    y = y_c.copy()

    # Estimate the SNRs of both signals
    noise_est = signaltonoise(y) # + signaltonoise(x)

    # Determine the length of the ffts as a power of 2
    L = len(y) + len(x) -1  # linear convolution length
    N = nextpow2(L)

    #Take ffts of the signals
    X = rfft(x,N)
    Y = rfft(y,N)

    # Transfer function estimation
    H = ((Y*Y.conj())/(X*Y.conj() + noise_est**2))

    #Impulse response
    h = irfft(H).real

    return h[:len(x)]


# x = input
# y = output
# Returns a IR estimation using weiner deconvolution.
def estimate_IR_deconvolution(x,y):

    # Determine the length of the ffts as a power of 2
    L = len(y) + len(x) -1  # linear convolution length
    N = nextpow2(L)

    #Take ffts of the signals
    X = rfft(x,N)
    Y = rfft(y,N)

    # Transfer function estimation
    H = Y/(X+0.000001)
    # H = ((Y*Y.conj())/(X*Y.conj()+0.00001))

    #Impulse response
    h = irfft(H).real

    return h[:len(x)]

# x = input
# y = output
# Returns a IR estimation using an iterative filtering idea that I came up with.
def estimate_IR_iterative_filtering(x,y,window_len=10):
    
    #first step is to apply hanning windows of different lengths of the signal and take the ffts of the products.
    #For this we want to use the input and output.
    #The input will tell us what the results should look like. The aim here is to build a central fft that emphasizes room components and minimizes noise. Windows lessen the impact of aliasing caused from noise.

    
    x_split = np.array(np.split(x,window_len))
    y_split = np.array(np.split(y,window_len))
    window = scipy.signal.hann( (len(x)//window_len) )
    
    x_fft_matrix = []
    for row in x_split:
        row = row*window
        row_fft = normalize(fft(row))
        if len(x_fft_matrix) == 0:
            x_fft_matrix = row_fft
        else:
            x_fft_matrix = np.vstack( (x_fft_matrix,row_fft) )
    
    y_fft_matrix = []
    for row in y_split:
        row = row*window
        row_fft = normalize(fft(row))
        if len(y_fft_matrix) == 0:
            y_fft_matrix = row_fft
        else:
            y_fft_matrix = np.vstack( (y_fft_matrix,row_fft) )
    
    formulated_fft_x = np.array([complex(0) for i in range(len(row_fft))])
    formulated_fft_y = np.array([complex(0) for i in range(len(row_fft))])
    for j in range(x_fft_matrix.shape[0]):
        row_x = x_fft_matrix[j]
        row_y = y_fft_matrix[j]

        ave_tmp_x = np.average(abs(row_x))
        ave_tmp_y = np.average(abs(row_y))
        for k in range(row_x.shape[0]):
            val_x = row_x[k]
            val_y = row_y[k]

            min_x = val_x/100
            min_y = val_y/100
            if abs(val_x) < ave_tmp_x/2:
                row_x[k] = min_x
            
            if abs(val_y) < ave_tmp_y/2:
                row_y[k] = min_y

            if row_x[k] == min_x and row_y[k] != min_y:
                row_y[k] = complex(ave_tmp_y,0)
                row_x[k] = min_x

        formulated_fft_x += row_x
        formulated_fft_y += row_y

    formulated_fft_x /= x_fft_matrix.shape[0]
    formulated_fft_y /= y_fft_matrix.shape[0]

    formulated_fft = (formulated_fft_y*formulated_fft_y.conj())/(formulated_fft_x*formulated_fft_y.conj() + signaltonoise(y)) 
    # formulated_fft = formulated_fft_y

    return ifft(formulated_fft,len(x)).real


