import numpy as np
import scipy
import scipy.signal
import pywt

def bit_agreement_windowed_means(a,b,window_size=10):
    m1 = a.mean()
    m2 = b.mean()
    bs1 = ""
    bs2 = ""
    for i in range(len(a)):
        if i*window_size < len(a):
            window1 = a[i*window_size: i*window_size+window_size]
            window2 = b[i*window_size: i*window_size+window_size]
            if window1.mean() > m1:
                bs1+="1"
            else:
                bs1+="0"
            if window2.mean() > m2:
                bs2+="1"
            else:
                bs2+="0"
        else:
            break

    matching = 0
    N = 0
    if len(bs1) != len(bs2):
        N = len(bs1) if len(bs1) > len(bs2) else len(bs2)
    else:
        N = len(bs1)

    count = 0
    for i in range(N):
        if i < len(bs1):
            ch1 = bs1[i]
        else:
            N = i
            break
        if i < len(bs2):
            ch2 = bs2[i]
        else:
            N = i
            break

        if ch1 == ch2:
            matching+=1
        count+=1
    agreement = matching/count

    return agreement


def bit_agreement_windowed_rms(a,b,window_size=10):
    m1 = np.sqrt(np.mean(a**2))
    m2 = np.sqrt(np.mean(b**2))
    bs1 = ""
    bs2 = ""
    for i in range(len(a)):
        if i*window_size < len(a):
            window1 = a[i*window_size: i*window_size+window_size]
            window2 = b[i*window_size: i*window_size+window_size]
            rms1 = np.sqrt(np.mean(window1**2))
            rms2 = np.sqrt(np.mean(window2**2))
            if rms1 > m1:
                bs1+="1"
            else:
                bs1+="0"
            if rms2 > m2:
                bs2+="1"
            else:
                bs2+="0"
        else:
            break

    matching = 0
    print(bs1)
    print(bs2)
    for i in range(len(bs1)):
        ch1 = bs1[i]
        ch2 = bs2[i]
        if ch1 == ch2:
            matching+=1

    agreement = matching/len(bs1)

    return agreement


def bit_agreement_galois(a,b,window_size=1000):
    from scipy.fft import fft, fftfreq, ifft

    from Crypto.Util import number
    p = number.getPrime(16)


    L = len(a) + len(b)  - 1
    N = nextpow2( L )

    A = abs(fft( a, N ))
    B = abs(fft( b, N ))

    tmp_set_a = []
    tmp_set_b = []


    freq = 10
    for i in A:
        val = round(i*freq,5)
        tmp_set_a.append(val)
        freq+=1


    freq = 10
    for i in B:
        val = round(i*freq,5)
        tmp_set_b.append(val)
        freq+=1

    both = zip(tmp_set_a,tmp_set_b)
    binary_str1 = ""
    binary_str2 = ""
    for x,y in both:
        x_i = int((p*x) )% p
        y_i = int((p*y)) % p

        binary_str1 += bin(x_i)[2:].zfill(16)
        binary_str2 += bin(y_i)[2:].zfill(16)

    total = 0
    matching = 0
    for i in range(0,len(binary_str1)):
        tmp_a = binary_str1[i]
        tmp_b = binary_str2[i]
        total+=1
        if tmp_a == tmp_b:
            matching+=1
    
    agreement = matching/total

    return agreement

def integral_bit_agreement(x,y,start=100,end=200):
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    X = np.array([i for i in range(len(x))])
    f1 = interp1d(X,x)
    Y = np.array([i for i in range(len(y))])
    f2 = interp1d(Y,y)
    
    bs1 = ""
    bs2 = ""

    diff = abs(start-end)
    a,a2 = quad(f1,start,end)
    a = a/diff
    b,b2 = quad(f2,start,end)
    b = b/diff
    step = 1
    for i in range(0,diff,int(diff/256)):
        tmp1,tmp_a2 = quad(f1,start+i,start+i+step)
        tmp2,tmp_a2 = quad(f2,start+i,start+i+step)
        
        if tmp1 > a:
            bs1+="1"
        else:
            bs1+="0"

        if tmp2 > b:
            bs2+="1"
        else:
            bs2+="0"

    print(bs1)
    print(bs2)
    matching = 0
    for i in range(len(bs1)):
        if bs1[i] == bs2[i]:
            matching+=1
    agreement = matching/len(bs1)
    return agreement


def base_sine_wave(x):
    return np.sin(x) + np.random.normal(scale=0.1, size=len(x))

def bit_agreement_cosine_distance(x,y,base_wave,step=100):
    from scipy.spatial.distance import cosine
    from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

    bs1 = ""
    bs2 = ""
   
    a = cosine(x,base_wave)
    b = cosine(y,base_wave)

    for i in range(0,int(len(x)),step):
        tmp_x = x[i:(i)+step]
        tmp_y = y[i:i+step]
        tmp_base = base_wave[i:i+step]
        a_tmp = cosine(tmp_x,tmp_base)
        b_tmp = cosine(tmp_y,tmp_base)

        if a_tmp > a:
            bs1 += "1"
        else:
            bs1+="0"
        if b_tmp > b:
            bs2 += "1"
        else:
            bs2+="0"
    
    matching = 0
    for i in range(len(bs1)):
        if bs1[i] == bs2[i]:
            matching+=1
    
    if bs1 == '' or bs2 == '':
        return 0
    else:
        print(bin(int(bs1,2) ^ int(bs2,2)))

    agreement = matching/len(bs1)
    return agreement


def create_bit_streams_audio(x1,window_len=10000, bands=1000):
    FFTs = []
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft

    if window_len == 0:
        window_len = len(x)

    x = np.array(x1.copy())
    wind = scipy.signal.windows.hann(window_len)
    for i in range(0,len(x),window_len):
        
        if len(x[i:i+window_len-1]) < window_len:
            wind = scipy.signal.windows.hann(len(x[i:i+window_len-1]))
            x[i:i+window_len-1] = x[i:i+window_len-1] * wind
        else:
            x[i:i+window_len-1] = x[i:i+window_len-1] * wind

        FFTs.append(abs(rfft(x[i:i+window_len-1])))
 
  
    E = {}
    bands_lst = []
    for i in range(0,len(FFTs)):
        frame = FFTs[i]
        bands_lst.append([ frame[k:k+bands-1] for k in range(0,len(frame),bands)])
        for j in range(0,len(bands_lst[i])):
            E[(i,j)] = np.sum(bands_lst[i][j])

    bs = ""
    for i in range(1,len(FFTs)):
        for j in range(0,len(bands_lst[i])-1):
            
            if E[(i,j)] -E[(i,j+1)] - (E[(i-1,j)] - E[(i-1,j+1)]) > 0:
                bs+= "1"
            else:
                bs+="0"
    
    return bs

def bit_agreement_ambient_audio_scheme(x,y,window_len=100, bands=25):
    FFTs = []
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft

    bs1 = create_bit_streams_audio(x,window_len,bands)
    bs2 = create_bit_streams_audio(y,window_len,bands)

    matching = 0
    for i in range(0,len(bs1)):
        if bs1[i] == bs2[i]:
            matching+=1
    
    if matching == 0:
        agreement = 0
    else:
        agreement = matching/len(bs1)

    if bs1 == '' or bs2 == '':
        return 0
    else:
        print(bin(int(bs1,2) ^ int(bs2,2)))
    
    return agreement

def create_bit_streams_wavelet_transform(x,window_len):
    bs1 = ""
    wavelets = pywt.wavelist(kind='discrete')
    wavelet_index = 1
    wave = wavelets[wavelet_index]
    all_values = []
    all_polys_A = []
    all_polys_D = []
    if window_len == 0:
        cA,cD = pywt.dwt(x, wave)
        ind1 = int(np.argmax(cD))%256
        ind2 = int(np.argmax(cA))%256
        bs1+= '{0:08b}'.format(ind1)
        bs1+= '{0:08b}'.format(ind2)
    else:
        for i in range(0,len(x), window_len):
            window = x[i:i+window_len]
            cA,cD = pywt.dwt(window, wave)
            
            A = np.poly1d(cA[:3])
            D = np.poly1d(cD[:3])
            
            all_polys_A.append(A)
            all_polys_D.append(D)

            val = np.max(cA)
            ind1 = abs(int(1/val))%256
            all_values.append(ind1)
            bs1+= '{0:08b}'.format(ind1)
    
    print(bs1)

    all_values = np.array(all_values)
    return (bs1,all_values,all_polys_A,all_polys_D)

def bit_agreement_wavelet_transform_algorithm(x,y,window_len=256, spec_plot=False):
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft

    bs1,av1,apa1,apd1 = create_bit_streams_wavelet_transform(x,window_len)
    bs2,av2,apa2,apd2 = create_bit_streams_wavelet_transform(y,window_len)
    
    cD,cA = pywt.dwt(x, 'db1')
    cD2,cA2 = pywt.dwt(y, 'db1')


    if spec_plot:
        x_axis = [i for i in range(100)]
        for i in range(len(apa1)):
            fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(18, 5))
            ax1.plot(x_axis, [apa1[i](j) for j in range(len(x_axis))] )
            ax1.plot(x_axis, [apa2[i](j) for j in range(len(x_axis))] )
            # ax1.plot([i for i in range(len(cA2))], cA2)
            plt.show()
            
    
    matching = 0
    for i in range(len(bs1)):
        if i < len(bs2) and bs1[i] == bs2[i]:
            matching+=1

    if matching == 0:
        return 0
    else:
        return matching/len(bs1)