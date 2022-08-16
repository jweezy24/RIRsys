import unittest
import scipy
import sys
import random
# Adding the src path to our environment so that the functions can be imported to the tests folder
# sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')
from IR_estimation_algorithms import *

import numpy as np

data = {}

def simulation_init(signal_length):
    system = scipy.signal.lti([1, 2], [3, 4])


    # Setting signal length
    signal_length = 2**8

    # Setting length for starting sine signal
    length = np.pi * 2

    range_inputs = np.arange(-length, length, length/signal_length  )

    # Create random amplitudes list to check after running tests to evaluate correctness 
    random_amps = []
    input_signal = np.sin(range_inputs)
    for i in range(0,10):
        
        # How many sine cycles
        cycles = random.randint(1,10)

        #Amplitude of the pure sine wave
        amplitude = random.randint(1,signal_length)
        

        length = np.pi * 2 * cycles

        range_inputs = np.arange(0, length, length/signal_length  )

        #Build tmp signal to convolve with the input signal to build a random signal
        tmp_signal = amplitude * np.sin(range_inputs)


        # Convolve tmp signal with input signal
        input_signal = scipy.signal.fftconvolve(input_signal,tmp_signal,'full')
    
    input_signal = normalize(input_signal)
    
    I = np.ndarray((len(range_inputs),1), buffer=input_signal)

    tout, output, xout = scipy.signal.lsim(system, U=I, T=range_inputs)

    return system,input_signal,output

# The test based on unittest module
class Test_IR_Algorithms(unittest.TestCase):

    def test_eval_decon(self):
        global data

        system = data["system"]
        input_signal = data["input"]
        output = data["output"]
        
        # Estimate our Impulse response
        IR_est = estimate_IR_weiner_deconvolution(input_signal,output)

        # Generate real IR
        IR = system.impulse()[1]


        data["IR_decon"] = IR
        data["IR_est_decon"] = IR_est


    def test_eval_weiner_decon(self):
        global data

        system = data["system"]
        input_signal = data["input"]
        output = data["output"]
        
        # Estimate our Impulse response
        IR_est = estimate_IR_weiner_deconvolution(input_signal,output)

        # Generate real IR
        IR = system.impulse()[1]
        

        data["IR_decon_w"] = IR
        data["IR_est_decon_w"] = IR_est

    
    def test_eval_power_spectrum(self):
        global data

        system = data["system"]
        input_signal = data["input"]
        output = data["output"]
        
        # Estimate our Impulse response
        IR_est = estimate_IR_power_spectrum(input_signal,output)

        # Generate real IR
        IR = system.impulse()[1]
        

        data["IR_power"] = IR
        data["IR_est_power"] = IR_est



def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_IR_Algorithms('test_eval_decon'))
    suite.addTest(Test_IR_Algorithms('test_eval_weiner_decon'))
    suite.addTest(Test_IR_Algorithms('test_eval_power_spectrum'))
    return suite

def plot_all_test(tests):
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=len(tests), ncols=4, figsize=(18, 5))

    count = 0
    for ax1,ax2,ax3,ax4 in axs:
        i = tests[count]
        IR_est =  abs(rfft(data[f"IR_est_{i}"]))
        IR =  abs(rfft(data[f"IR_{i}"]))
        A = abs(rfft(data["input"]))
        B = abs(rfft(data["output"]))

        ax1.plot(np.arange(0,len(IR_est)), IR_est)
        ax1.set_title("IR Estimate fft")
        ax2.plot(np.arange(0,len(IR)), IR)
        ax2.set_title("Actual IR fft")
        ax3.plot(np.arange(0,len(A)), A)
        ax3.set_title("fft of Input Values")
        ax4.plot(np.arange(0,len(B)), B )
        ax4.set_title("fft of Output Values")
        count+=1

    plt.show()

if __name__ == '__main__':
    signal_length = 2**10

    system,input_signal,output = simulation_init(signal_length)

    data["system"] = system
    data["input"] = input_signal
    data["output"] = output
    runner = unittest.TextTestRunner()
    runner.run(suite())

    

    tests = ["decon", "decon_w","power"]
    plot_all_test(tests)

