import os
import glob
import scipy
from scipy.io.wavfile import write as wavwrite
import numpy as np
import sounddevice as sd
from multiprocessing import Process
from multiprocessing import Manager
import time
import soundfile
import matplotlib.pyplot as plt
import pickle
import sys
import math
import pywt


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def signaltonoise(a, axis=0, ddof=1):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


# Will return the first channel of a multi channel audio source
def get_raw_audio_stream(path,scale=False):
    if scale:
        data,freq = soundfile.read(path)
        data = normalize(data).transpose()
        if len(data.shape) > 1:
            for i in range(1,data.shape[1]):
                data[:,0] += data[:,i]
            data = data[:,0]/data.shape[1]
    else:
        data,freq = soundfile.read(path)
        if len(data.shape) > 1:
            for i in range(1,data.shape[1]):
                data[:,0] += data[:,i]
            data = data[:,0]/data.shape[1]
    
    return (freq,np.array(data))


def find_newest_file(folder):
    
    list_of_files_tmp = glob.glob(f'{folder}/*/*.wav') # * means all if need specific format then *.csv
    
    list_of_files = []
    
    for fi in list_of_files_tmp:
        if "RIR1.wav" not in fi:
            continue
        else:
            list_of_files.append(fi)

    latest_file = max(list_of_files, key=os.path.getctime)
    
    print(latest_file)
    return latest_file

def set_stimulus():
    testStimulus = stim.stimulus('sinesweep', 48000)
    testStimulus.generate(48000, 10, 0.2, 1 ,2, 2, [0, 0])

    return testStimulus


def measure_impulse(testStimulus,play_sound):

    # Record
    # recorded = record(testStimulus.signal,args.fs,args.inputChannelMap,args.outputChannelMap)
    

    recorded = record_or_play(testStimulus.signal,48000,play_sound=play_sound)
    
    if type(recorded) == type(None):
        return None
    # recorded = 0

    #multi-threaded Microphones
    # streams = record_multi_mic(testStimulus.signal,args.fs,args.inputChannelMap,args.outputChannelMap)
    # print(streams)
    streams = 0
    
    # Deconvolve
    if type(recorded) != type(0):

        recorded = shift_samples(testStimulus.signal.T[0],recorded.T[0],recorded.T[0],use_pearson=False, start=468000, stop=624000)

       
        recorded = recorded.reshape(-1,1)


        RIR = testStimulus.deconvolve(recorded)
        # Truncate
        lenRIR = 1.2;
        startId = testStimulus.signal.shape[0] -int(2*48000) -1
        endId = startId + int(lenRIR*48000)
        # save some more samples before linear part to check for nonlinearities
        startIdToSave = startId - int(48000/2)
        RIRtoSave = RIR[startIdToSave:endId,:]
        RIR = RIR[startId:endId,:]

        # Save recordings and RIRs
        saverecording(RIR, RIRtoSave, testStimulus.signal, recorded, 48000)
    else:

        recorded1 = streams[10]
        recorded2 = streams[11]

        RIR1 = testStimulus.deconvolve(recorded1)
        RIR2 = testStimulus.deconvolve(recorded2)

        # Truncate
        lenRIR = 1.2;
        startId = testStimulus.signal.shape[0] - args.endsilence*args.fs -1
        endId = startId + int(lenRIR*args.fs)
        # save some more samples before linear part to check for nonlinearities
        startIdToSave = startId - int(args.fs/2)
        
        RIRtoSave1 = RIR1[startIdToSave:endId,:]
        RIR1 = RIR1[startId:endId,:]

        RIRtoSave2 = RIR2[startIdToSave:endId,:]
        RIR2 = RIR2[startId:endId,:]

        # Save recordings and RIRs
        saverecording(RIR1, RIRtoSave1, testStimulus.signal, recorded1, args.fs,multi=True,ind=1,dirname='recorded/multirir1')
        saverecording(RIR2, RIRtoSave2, testStimulus.signal, recorded2, args.fs,multi=True,ind=2,dirname='recorded/multirir1')



def record_or_play(testsignal,fs,play_sound=False):
    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    
    print(f"STARTING AT {time.time()}")

        
    if play_sound:
        sd.play(testsignal, samplerate=fs)
        sd.wait()
        recorded = None
        
    else:
        recorded = sd.rec(frames=len(testsignal),channels=1)
        sd.wait()

    return recorded

#--------------------------
def record(testsignal,fs,inputChannels,outputChannels,return_dict=0,play_sound=False,sd2=sd):
    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    print("Input channels:",  inputChannels)
    print("Output channels:", outputChannels)

    print(f"{inputChannels} HERE")

    # Start the recording
    if type(return_dict) == type(0):
        recorded = sd.playrec(testsignal, samplerate=fs, input_mapping = inputChannels,output_mapping = outputChannels)
        sd.wait()
        print(f"{inputChannels} HERE")
    else:
        print(f"{inputChannels} HERE")
        if play_sound:
            print(f"{inputChannels} PLAY SOUND TOP")
            sd.play(testsignal, samplerate=fs)
            sd.wait()
            recorded = None
            print(f"{inputChannels} PLAY SOUND BOTTOM")
            
        else:
            sd.default.device = inputChannels[0]
            print(f"{inputChannels} RECORD TOP")
            added_time = fs*4
            print(len(testsignal))
            recorded = sd.rec(frames=len(testsignal)+added_time, channels=1)
            sd.wait()
            print(f"{inputChannels} RECORD BOTTOM")

            return_dict[inputChannels[0]] = recorded
            print(f"{inputChannels} BYTES SAVED")
        

    return recorded


#--------------------------
def record_multi_mic(testsignal,fs,inputChannels,outputChannels):

    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    print("Input channels:",  inputChannels)
    print("Output channels:", outputChannels)

    manager = Manager()
    streams = manager.dict()

    import sounddevice as sd2

    # Start the recording
    p1 = Process(target=record, args=(testsignal,fs,[10],[2],streams,),daemon=True,name="T1")
    p2 = Process(target=record, args=(testsignal,fs,[11],[2],streams,),daemon=True,name="T2")
    
    p1.start()
    p2.start()
    
    recorded = sd.play(testsignal, samplerate=fs)
    sd.wait()

    # p1.join()
    # p2.join()

    time.sleep(1)


    procs = [p1,p2]
    running = True

    while running:
        counter =0
        for p in procs:
            # print(f"{p.name} = {p.is_alive()} ")
            if not p.is_alive():
                counter+=1

        if counter == 2:
            running = False
    
    print(streams)

    return streams


#--------------------------
def saverecording(RIR, RIRtoSave, testsignal, recorded, fs,multi=False,ind=1,dirname=""):
    if not multi:
        dirflag = False
        counter = 1
        dirname = 'recorded/newrir1'
        while dirflag == False:
            if os.path.exists(dirname):
                counter = counter + 1
                dirname = 'recorded/newrir' + str(counter)
            else:
                os.mkdir(dirname)
                dirflag = True

        # Saving the RIRs and the captured signals
        np.save(dirname+ '/RIR.npy',RIR)
        np.save(dirname+ '/RIRac.npy',RIRtoSave)
        wavwrite(dirname+ '/sigtest.wav',fs,testsignal)

        for idx in range(recorded.shape[1]):
            wavwrite(dirname+ '/sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ '/RIR' + str(idx+1) + '.wav',fs,RIR[:,idx])

        # Save in the recorded/lastRecording for a quick check
        np.save('recorded/lastRecording/RIR.npy',RIR)
        np.save( 'recorded/lastRecording/RIRac.npy',RIRtoSave)
        wavwrite( 'recorded/lastRecording/sigtest.wav',fs,testsignal)
        for idx in range(recorded.shape[1]):
            wavwrite('sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ '/RIR' + str(idx+1) + '.wav',fs,RIR[:,idx])


        print('Success! Recording saved in directory ' + dirname)
    else:
        dirflag = False
        counter = 1
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        # Saving the RIRs and the captured signals
        np.save(dirname+ f'/RIR{ind}.npy',RIR)
        np.save(dirname+ f'/RIRac{ind}.npy',RIRtoSave)
        wavwrite(dirname+ '/sigtest.wav',fs,testsignal)

        for idx in range(recorded.shape[1]):
            wavwrite(dirname+ '/sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ f'/RIR{ind}_' + str(idx+1) + '.wav',fs,RIR[:,idx])

        # Save in the recorded/lastRecording for a quick check
        np.save('recorded/lastRecording/RIR.npy',RIR)
        np.save( 'recorded/lastRecording/RIRac.npy',RIRtoSave)
        wavwrite( 'recorded/lastRecording/sigtest.wav',fs,testsignal)
        for idx in range(recorded.shape[1]):
            wavwrite('sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ '/RIR' + str(idx+1) + '.wav',fs,RIR[:,idx])

        print('Success! Recording saved in directory ' + dirname)

def shift_samples(orig, recorded,full_recording,use_pearson=False, start=50000,stop=100000,est_delay=7000):
    correlations = []
    index = 0
    step = -1
    tmp_roll = recorded
    s = np.array([0 for i in range(1000)])
    tmp_roll2 = np.concatenate((s, tmp_roll[start-est_delay:stop-est_delay]))
    tmp_original = np.concatenate((s, orig[start:stop]))
    for i in range(0,13000):
        
        tmp_roll2 = np.roll(tmp_roll2,step)
        
        if use_pearson:
            corr = scipy.stats.pearsonr(tmp_roll2,tmp_original)[0]
            correlations.append(corr)
        else:
            corr = np.inner(tmp_roll2,tmp_original)
            correlations.append(corr)
        
    
    correlations = np.array(correlations)

    if index == 0:
        index = np.argmax(abs(correlations))
    
    print(f"Correlation at index {index} and has value {correlations[index]}")
    
    # rest_of_clip = scipy.ndimage.shift(full_recording,-index,cval=np.NaN)
    
    final = np.roll(full_recording,-index)
    #final = np.array([x for x in rest_of_clip if np.isnan(x) == False])
    
    return final


def nextpow2( L ):
    N = 2
    while N < L: N = N * 2
    return N

def generate_wavs():
    public_audio_path = "public_audio_files/song.wav"

    audio = get_raw_audio_stream(public_audio_path)[1]

    s = np.array([0 for i in range(1000)])

    audio = np.concatenate((s, audio))

    recording = "before_shifting_recording.wav"

    recording = get_raw_audio_stream(recording)[1]

    final = shift_samples(audio,recording,record0ing)

    wavwrite(f'original.wav', 48000, audio )
    
    wavwrite(f'after_shifting_recording.wav', 48000, final )

def compute_correlation_function(x, y):
    '''Compute correlation function/kappa.'''
    N, M = len(x), len(y)
    ccf = 1/N * np.correlate(x, y, mode='same')
    kappa = np.arange(-M+1, N)

    return ccf, kappa

def build_Henk_matrix(data):
    from scipy.linalg import hankel
    return hankel(data)

def impulse_response_estimation(y,x, T=3, Tr=2, fs=48000,n=100):
    import scipy.signal as sig
    from scipy.signal import lfilter
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift

    # fs = 48000  # sampling rate
    # T =  3  # length of the measurement signal in sec
    # Tr = 1  # length of the expected system response in sec
    # the larger n is, the smoother curve will be
    b = [0.5 / n] * n
    a = 0.5

    y = lfilter(b,a,y)

    # x = x[abs(x)>0.0000001]
    # y = y[abs(y)>0.0000001]

    h = 1/len(y) * sig.fftconvolve(y, x, mode='full')

    # ccfxy,kappx = compute_correlation_function(x,y)
    # h = ccfxy


    h = h[int(fs*(T+Tr)):int(fs*(2*T+Tr))]

    # plt.plot(np.arange(len(h)), h)
    # plt.show()

    

    t = 1/fs * np.arange(len(h))
    return(t,h)

def impulse_response_estimation_ls(y,x,T=3, Tr=2, fs=48000, n=50, wind=48000, smooth=100,plot_IR=False,save_IR=False):
    import scipy.signal as sig
    from scipy.signal import lfilter
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift
    from numpy.linalg import inv
    import noisereduce as nr
    import cmath

    b = [1.0 / smooth] * smooth
    a = 1

    # y = lfilter(b,a,y)
    # x = lfilter(b,a,x)

    y = normalize(y)
    x = normalize(x)

    y = nr.reduce_noise(y=y, sr=fs)

    iterator = len(y)/wind
    hann = sig.hann(int(len(y)/wind))
    for i in range(0,len(y),int(iterator)):
        if len(y[i:i+int(iterator)]) < len(hann) or len(x[i:i+int(iterator)]) < len(hann):
            break
        y[i:i+int(iterator)] = y[i:i+int(iterator)]*hann
        x[i:i+int(iterator)] = x[i:i+int(iterator)]*hann

    # lambd_est = signaltonoise(x) + signaltonoise(y)

    L = len(y) + len(x) -1  # linear convolution length

    N = nextpow2(L)

    
    X = fft( x, N )
    Y = fft( y, N )


    A =  1/len(X) * (X*Y.conj())/(Y*Y.conj())

    f_real = A.real
    f_imag = A.imag



    H = A
    HA = np.argsort(abs(H))


    H = H[HA]

    Hank_zs = build_Henk_matrix(H[:20])
    Hank_ps = build_Henk_matrix(H[len(H)-19:])

    U_zs,E_zs,V_zs = scipy.linalg.svd(Hank_zs)
    U_ps,E_ps,V_ps = scipy.linalg.svd(Hank_ps)

    zeros = U_zs[:,0].flatten()

    poles = U_ps[:,0].flatten()


    # zeros = np.linalg.eig(np.dot(U_zs,V_zs))[1].flatten()
    # poles = np.linalg.eig(np.dot(U_ps,V_ps))[1].flatten()



    def f(x, zeros):
        y = np.zeros(x.shape, dtype=complex)
        for i, v in enumerate(zeros):
            y *= x
            y += v
        return y



    
    
    H_vals = []
    points = np.linspace(-np.pi/2, np.pi/2, len(X))    
    H_vals =   (f(points,zeros)/f(points,poles))


    h = ifft(np.array(H_vals)).real

    h = h/( max(h)*1.01 )


    

    # print(h)
    if plot_IR:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
        ax1.plot(np.arange(len(A)), abs(A))
        ax2.plot(np.arange(len(E_zs)), E_zs)
        ax3.plot(np.arange(len(H_vals)), H_vals)
        ax4.plot(np.arange(len(h)), h)
        plt.show()
    
    if save_IR:
        wavwrite('test_wav_impulse.wav',fs,h)
    
    t = [0]
    return(t,h)

def impulse_response_estimation_power_spectral(y,x,T=3, Tr=2, fs=48000,n=50, wind=1000, smooth=100):
    import scipy.signal as sig
    from scipy.signal import lfilter
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift
    from numpy.linalg import inv
    import noisereduce as nr
    import cmath

    S_yx = sig.csd(x,y)[1]
    S_xx = sig.periodogram(x,fs=fs)[1]
    S_yy = sig.periodogram(y,fs=fs)[1]

    z1 = len(S_xx) - len(S_yx)
    z2 = len(S_yy) - len(S_yx)

    print(S_xx)
    print(S_yx)
    zeros1 = np.zeros(z1)
    zeros2 = np.zeros(z2)

    S_yx1 = np.concatenate( (S_yx, zeros1) )
    S_yx2 = np.concatenate( (S_yx, zeros2) )

    H_div = (fft(x)*fft(y).conj())/(fft(y)*fft(y).conj())

    H_1 = S_yx1/S_xx
    H_2 = S_yy/S_yx1
    H = (S_yy/S_xx) 

    # H = ifft(H).real


    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    # ax1.scatter(np.arange(len(A)), abs(A))
    ax2.plot(np.arange(len(H_div)), H_div)
    ax3.plot(np.arange(len(H)), H)
    ax3.set_yscale('log')

    # H_vals= H_vals/( max(H_vals)*1.01 )

    # wavwrite('test_wav_impulse.wav',fs,H)

    plt.show()



    # h = ifft(np.array(H_vals)).real#abs(ifft(np.array(H_vals)))#ifft(H_vals).real

    return([0],H)




def testing_synced_responses():
    public_audio_path = "public_audio_files/test_voice.wav"

    silence = [0 for i in range(48000*2)]

    audio = np.concatenate((get_raw_audio_stream(public_audio_path)[1],silence))

    for root, dirs, files in os.walk("/home/jweezy/Documents/ambient_audio_experiments/pi_system_software/public_audio_files", topdown=False):

        for name in files:
            if "2." in name or "test_voice" in name or "song" in name or "white_noise" in name or ".wav" not in name:
                continue
            else:
                name = name.replace(".wav","")
    
                public_audio_path = f"public_audio_files/{name}.wav"

                audio2 = get_raw_audio_stream(public_audio_path,True)[1][48000*2:]

                public_audio_path = f"public_audio_files/{name}2.wav"

                audio3 = get_raw_audio_stream(public_audio_path,True)[1][48000*2:]

                # x = np.random.uniform(-0.0001, 0.0001, size=int(len(audio2)/48000*48000))

                t,h = impulse_response_estimation(audio2,audio)
                t2,h2 = impulse_response_estimation(audio3,audio)
                wavwrite(f"../audio_dataset/Isaac_data/test_impulse_{name}_1.wav", 48000, h)
                wavwrite(f"../audio_dataset/Isaac_data/test_impulse_{name}_2.wav", 48000, h2)



def convolution( xt, ht, decon=True, plot_ffts=False):
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift

    x = xt.copy()
    h = ht.copy()

    L = len(h) + len(x) -1  # linear convolution length

    lambd_est = signaltonoise(x) + signaltonoise(h)

    N = nextpow2(L)
    
    X = fft( x, N )
    H = fft( h, N )

    # spectral math
    if decon:
        print(f"LAMBDA ESTIMATE = {lambd_est}")
        Y = (np.conj(H)*X)/(H*np.conj(H)+lambd_est**2)
        
    else:   
        Y = X * H


    # FFT plots
    if plot_ffts:
        # A = rfft( A, N )
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))


        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("FFT of Impulse Response")
        ax1.plot([i for i in range(len(H))], abs(H))
        ax1.set_yscale('log')
        ax2.plot([i for i in range(len(X))], abs(X))
        ax2.set_yscale('log')
        ax3.plot([i for i in range(len(Y))], abs(Y))
        ax3.set_yscale('log')
        
        plt.show()
    
    
    y = irfft( Y )
    y = np.array(abs(y)).astype("float64")
    clip_factor = 1.01
    y = y/( max(y)*clip_factor )
   #y = normalize(y)
    return y[:len(x)]


def sum_ffts(X,Y,Z,iterator,add=True):

    count = 0
    print(len(Y))
    for i in range(0,len(Y),int(iterator)):
        if add:
            Y[count] = Y[i:i+int(iterator)-1].sum()
            X[count] = X[i:i+int(iterator)-1].sum()
            Z[count] = Z[i:i+int(iterator)-1].sum()
            count+=1
        else:
            Y[count] = Y[i:i+int(iterator)-1].sum()/iterator
            X[count] = X[i:i+int(iterator)-1].sum()/iterator    
            Z[count] = Z[i:i+int(iterator)-1].sum()/iterator
            count+=1

    Y = Y[:count]
    X = X[:count]
    Z = Z[:count]

    print(count)

    return X,Y,Z


def windowed_cosine_distance_calculation(x,y,window_len):
    chop = len(x) -  len(x)%window_len
    

    xsplit = np.split( x[:chop],window_len )
    ysplit = np.split( y[:chop],window_len )

    cosine_distances = []
    for i in range(len(xsplit)):
        d = scipy.spatial.distance.cosine(xsplit[i],ysplit[i])
        cosine_distances.append(d)

    return cosine_distances


def dwt_analysis(data,rooms):
    from math import log2,ceil

    local_data = {}

    for room in rooms:
        tmp = data[room]
        for key in tmp.keys():
            fs,IR = tmp[key]

            

            lvls = 5

            # cA,cD = pywt.dwt(IR, 'dmey','zero')

            #scales = np.arange(1,128)
            #res = pywt.cwt(IR, scales, 'gaus1')
            #local_data[f"{room}_{key}_cwt"] = res

            local_data[f"{room}_{key}_cA"] = normalize(IR)
            local_data[f"{room}_{key}_cD"] = normalize(IR)
            
            
            
    
    return local_data

#Measures distance metrics between two signals or data.
def distance_compare(x,y,use_fft=True):
    from scipy.spatial.distance import correlation,cosine,euclidean
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft


    if not use_fft:
        
        d1 = correlation(x,y)
        d2 = cosine(x,y)
        d3 = euclidean(x,y)
        return d1,d2,d3,1
    else:
        x = abs(fft(x))
        y = abs(fft(y))
        d1 = correlation(x,y)
        d2 = cosine(x,y)
        d3 = euclidean(x,y)
        return d1,d2,d3,1


def distance_compare_windows(x,y,use_fft=False,window_len=100):
    from scipy.spatial.distance import correlation,cosine,euclidean
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft

    x_split = np.array_split(x,window_len)[:-1]
    y_split = np.array_split(y,window_len)[:-1]
    if not use_fft:
        d1 = 0
        d2 = 0
        d3 = 0
        dh = 0
        for i in range(0,window_len-1):
            x = x_split[i]
            y = y_split[i]

            from teaspoon.SP.tsa_tools import takens

            embedded_ts_x = takens(x, n = 2, tau = 15)
            embedded_ts_y = takens(y, n = 2, tau = 15)

            from ripser import ripser
            #calculate the rips filtration persistent homology
            result_x = ripser(embedded_ts_x, maxdim=1)
            result_y = ripser(embedded_ts_y, maxdim=1)

            diagram_x = result_x['dgms']
            diagram_y = result_y['dgms']


            B, D = diagram_x[1].T[0], diagram_x[1].T[1]
            L_x = D - B 

            B, D = diagram_y[1].T[0], diagram_y[1].T[1]
            L_y = D - B

            from teaspoon.SP.information.entropy import PersistentEntropy
            h_x = PersistentEntropy(lifetimes = L_x)
            h_y = PersistentEntropy(lifetimes = L_y)   

            d1 += correlation(x,y)
            d2 += cosine(x,y)
            d3 += euclidean(x,y)
            dh += (h_x-h_y)
        return d1/(window_len-1),d2/(window_len-1),d3/(window_len-1),dh/(window_len-1)
    else:
        x = abs(fft(x))
        y = abs(fft(y))
        d1 = correlation(x,y)
        d2 = cosine(x,y)
        d3 = euclidean(x,y)
        return d1,d2,d3,1