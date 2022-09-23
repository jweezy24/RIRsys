import unittest
import scipy
import sys
import random
import argparse

# Adding the src path to our environment so that the functions can be imported to the tests folder
sys.path.insert(1, 'src/')
# sys.path.insert(1, '../src/')

from IR_estimation_algorithms import *
from bit_stream_algorithms import *

import numpy as np

data = {}

#Argparse init

parser = argparse.ArgumentParser()
parser.add_argument('--room1', help='File path for first room impulse response.')
parser.add_argument('--room2', help='File path for second room impulse response. Should be a legitimate device that should authenticate with room1.')
parser.add_argument('--room3', help='File path for third room impulse response. Should be an adversarial case, in other words, a different location than room1.')
parser.add_argument('--original_audio', help='Path to a untainted wav file that the room files listen to within the room.')
parser.add_argument('--align_buffers', action='store_true', help="Argument that specifies if the data is aligned or not. Default value is false and the code will align the buffer." )
args = vars(parser.parse_args())

def data_init(r1,r2,r3,orig,chop_silence=True):

    r1 = get_raw_audio_stream(r1)[1]
    r2 = get_raw_audio_stream(r2)[1]
    r3 = get_raw_audio_stream(r3)[1]
    orig = get_raw_audio_stream(orig)[1]
    
    start = 219840
    stop = 267840
    aligned_buffer_r1 = shift_samples(orig,r1,r1,start=start,stop=stop)
    aligned_buffer_r2 = shift_samples(orig,r2,r2,start=start,stop=stop)
    aligned_buffer_r3 = shift_samples(orig,r3,r3,start=start,stop=stop)

    if chop_silence:
        aligned_buffer_r1 = aligned_buffer_r1[48000*2:len(aligned_buffer_r1)-48000*2]
        aligned_buffer_r2 = aligned_buffer_r2[48000*2:len(aligned_buffer_r2)-48000*2]
        aligned_buffer_r3 = aligned_buffer_r3[48000*2:len(aligned_buffer_r3)-48000*2]
        orig = orig[48000*2:len(orig)-48000*2]


    return aligned_buffer_r1,aligned_buffer_r2,aligned_buffer_r3,orig

# The test based on unittest module
class Test_IR_Algorithms(unittest.TestCase):

    def test_eval_decon(self):
        global data

        r1 = data["r1"]
        r2 = data["r2"]
        r3 = data["r3"]
        orig = data["orig"]
        
        # Estimate our Impulse response
        IR_est_1 = estimate_IR_deconvolution(r1,orig)
        IR_est_2 = estimate_IR_deconvolution(r2,orig)
        IR_est_3 = estimate_IR_deconvolution(r3,orig)


        data["IR_est_decon_1"] = IR_est_1
        data["IR_est_decon_2"] = IR_est_2
        data["IR_est_decon_3"] = IR_est_3


    def test_eval_weiner_decon(self):
        global data

        r1 = data["r1"]
        r2 = data["r2"]
        r3 = data["r3"]
        orig = data["orig"]
        
        # Estimate our Impulse response
        IR_est_1 = estimate_IR_weiner_deconvolution(r1,orig)
        IR_est_2 = estimate_IR_weiner_deconvolution(r2,orig)
        IR_est_3 = estimate_IR_weiner_deconvolution(r3,orig)



        data["IR_est_decon_w_1"] = IR_est_1
        data["IR_est_decon_w_2"] = IR_est_2
        data["IR_est_decon_w_3"] = IR_est_3

    def test_eval_power_spectrum(self):
        global data

        r1 = data["r1"]
        r2 = data["r2"]
        r3 = data["r3"]
        orig = data["orig"]
        
        # Estimate our Impulse response
        IR_est_1 = estimate_IR_power_spectrum(r1,orig)
        IR_est_2 = estimate_IR_power_spectrum(r2,orig)
        IR_est_3 = estimate_IR_power_spectrum(r3,orig)


        data["IR_est_power_1"] = IR_est_1
        data["IR_est_power_2"] = IR_est_2
        data["IR_est_power_3"] = IR_est_3


    def test_eval_kronecker_product(self):
        global data

        system = data["system"]
        input_signal = data["input"]
        output = data["output"]
        
        # Estimate our Impulse response
        IR_est = estimate_IR_kronecker_product(input_signal,output,1)

        # Generate real IR
        IR = system.impulse(N=len(IR_est))[1]
        

        data["IR_kronecker"] = IR
        data["IR_est_kronecker"] = IR_est

    def test_eval_iterative_windowing(self):
        global data

        r1 = data["r1"]
        r2 = data["r2"]
        r3 = data["r3"]
        orig = data["orig"]
        
        # Estimate our Impulse response
        IR_est_1 = estimate_IR_iterative_filtering(orig,r1)
        IR_est_2 = estimate_IR_iterative_filtering(orig,r2)
        IR_est_3 = estimate_IR_iterative_filtering(orig,r3)

        data["IR_est_wind_1"] = IR_est_1
        data["IR_est_wind_2"] = IR_est_2
        data["IR_est_wind_3"] = IR_est_3


def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_IR_Algorithms('test_eval_decon'))
    suite.addTest(Test_IR_Algorithms('test_eval_weiner_decon'))
    suite.addTest(Test_IR_Algorithms('test_eval_power_spectrum'))
    suite.addTest(Test_IR_Algorithms('test_eval_iterative_windowing'))
    # suite.addTest(Test_IR_Algorithms('test_eval_kronecker_product'))
    return suite

def plot_all_test(tests,plot_ffts=False,windowed_cosine_distance=False):
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=len(tests), ncols=5, figsize=(18, 5))

    count = 0
    for ax1,ax2,ax3,ax4,ax5 in axs:
        i = tests[count]
        if plot_ffts:
            IR_est_1 =  abs(rfft(data[f"IR_est_{i}_1"]))
            IR_est_2 =  abs(rfft(data[f"IR_est_{i}_2"]))
            IR_est_3 =  abs(rfft(data[f"IR_est_{i}_3"]))
            orig =  abs(rfft(data[f"orig"]))

            ax1.plot(np.arange(0,len(IR_est_1)), IR_est_1)
            ax1.set_yscale("log")
            ax1.set_title("Room1 IR Estimate fft")
            ax2.plot(np.arange(0,len(IR_est_2)), IR_est_2)
            ax2.set_yscale("log")
            ax2.set_title("Room2 IR Estimate fft")
            ax3.plot(np.arange(0,len(IR_est_3)), IR_est_3)
            ax3.set_yscale("log")
            ax3.set_title("Room3 IR Estimate fft")
            ax4.plot(np.arange(0,len(orig)), orig )
            ax4.set_yscale("log")
            ax4.set_title("Original fft of audio")
            count+=1

            if windowed_cosine_distance:
                win = 8
                cd1 = windowed_cosine_distance_calculation(IR_est_1, IR_est_2, win)

                cd2 = windowed_cosine_distance_calculation(IR_est_2, IR_est_3, win)

                cd3 = windowed_cosine_distance_calculation(IR_est_3, IR_est_1, win)

                ax5.plot(np.arange(len(cd1)), cd1)
                ax5.plot(np.arange(len(cd1)), cd2)
                ax5.plot(np.arange(len(cd1)), cd3)

            else:
                d1 = scipy.spatial.distance.cosine(IR_est_1,IR_est_2)
                d2 = scipy.spatial.distance.cosine(IR_est_2,IR_est_3)
                d3 = scipy.spatial.distance.cosine(IR_est_3,IR_est_1)

                x=[1,2,3]
                y=[d1,d2,d3]
                ax5.bar(x,y,0.2)
            


        else:
            IR_est_1 =  data[f"IR_est_{i}_1"]
            IR_est_2 =  data[f"IR_est_{i}_2"]
            IR_est_3 =  data[f"IR_est_{i}_3"]
            orig =  data[f"orig"]

            ax1.plot(np.arange(0,len(IR_est_1)), IR_est_1)
            ax1.set_title("Room1 IR Estimate")
            ax2.plot(np.arange(0,len(IR_est_2)), IR_est_2)
            ax2.set_title("Room2 IR Estimate")
            ax3.plot(np.arange(0,len(IR_est_3)), IR_est_3)
            ax3.set_title("Room3 IR Estimate")
            ax4.plot(np.arange(0,len(orig)), orig )
            ax4.set_title("Original audio")
            count+=1

            if windowed_cosine_distance:
                win = 8
                cd1 = windowed_cosine_distance_calculation(IR_est_1, IR_est_2, win)

                cd2 = windowed_cosine_distance_calculation(IR_est_2, IR_est_3, win)

                cd3 = windowed_cosine_distance_calculation(IR_est_3, IR_est_1, win)

                ax5.plot(np.arange(len(cd1)), cd3)
                ax5.plot(np.arange(len(cd1)), cd2)
                ax5.plot(np.arange(len(cd1)), cd1)

            else:
                d1 = scipy.spatial.distance.cosine(IR_est_1,IR_est_2)
                d2 = scipy.spatial.distance.cosine(IR_est_2,IR_est_3)
                d3 = scipy.spatial.distance.cosine(IR_est_3,IR_est_1)

                x=[1,2,3]
                y=[d1,d2,d3]
                ax5.bar(x,y,0.2)

    plt.show()

def compare_bit_streams(tests):
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=len(tests), ncols=3, figsize=(18, 5))
    count = 0
    public_audio = get_raw_audio_stream("/home/jweezy/Documents/ambient_audio_experiments/audio_dataset/Isaac_data/white_noise_10.wav")[1]

    for ind in range(len(tests)):
        i = tests[ind]
        # IR_est_1 =  convolution( public_audio, data[f"IR_est_{i}_1"])
        # IR_est_2 =  convolution( public_audio, data[f"IR_est_{i}_2"])
        # IR_est_3 =  convolution( public_audio, data[f"IR_est_{i}_3"])

        IR_est_1 =  data[f"IR_est_{i}_1"]
        IR_est_2 =  data[f"IR_est_{i}_2"]
        IR_est_3 =  data[f"IR_est_{i}_3"]

        # IR_est_1 =  rfft(data[f"IR_est_{i}_1"]).real
        # IR_est_2 =  rfft(data[f"IR_est_{i}_2"]).real
        # IR_est_3 =  rfft(data[f"IR_est_{i}_3"]).real

        # bw = base_sine_wave(np.arange(0,len(IR_est_1)))
        bw = public_audio[:len(IR_est_1)]

        agreement_11 = 0#bit_agreement_cosine_distance(IR_est_1,IR_est_2,bw)
        agreement_21 = 0#bit_agreement_cosine_distance(IR_est_1,IR_est_3,bw)
        agreement_31 = 0#bit_agreement_cosine_distance(IR_est_3,IR_est_2,bw)

        agreement_12 = bit_agreement_ambient_audio_scheme(IR_est_1,IR_est_2)
        agreement_22 = bit_agreement_ambient_audio_scheme(IR_est_1,IR_est_3)
        agreement_32 = bit_agreement_ambient_audio_scheme(IR_est_3,IR_est_2)

        agreement_13 = quam_bit_agreement(IR_est_1,IR_est_2)
        agreement_23 = quam_bit_agreement(IR_est_1,IR_est_3)
        agreement_33 = quam_bit_agreement(IR_est_3,IR_est_2)

        
        y = [agreement_11,agreement_21,agreement_31]
        x = [1,2,3]
        axs[ind][0].bar(x,y)
        y = [agreement_12,agreement_22,agreement_32]
        x = [1,2,3]
        axs[ind][1].bar(x,y)
        y = [agreement_13,agreement_23,agreement_33]
        x = [1,2,3]
        axs[ind][2].bar(x,y)
        count+=1


    
    plt.show()




if __name__ == '__main__':

    r1,r2,r3,orig = data_init(args["room1"],args["room2"],args["room3"],args["original_audio"])

    data["r1"] = r1
    data["r2"] = r2
    data["r3"] = r3
    data["orig"] = orig

    runner = unittest.TextTestRunner()
    runner.run(suite())

    tests = ["decon", "decon_w","power","wind"]
    plot_all_test(tests)
    tests = ["decon", "decon_w","power","wind"]
    compare_bit_streams(tests)



