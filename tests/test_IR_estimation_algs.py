import unittest
import scipy
import sys
import random
# Adding the src path to our environment so that the functions can be imported to the tests folder
sys.path.insert(1, '../src')
from IR_estimation_algorithms import *

import numpy as np

data = {}

# The test based on unittest module
class Test_IR_Algorithms(unittest.TestCase):


    def test_eval_weiner_decon(self):
        global data
        # Simulated system to evaluate over
        # Transfer function, in this case, is H= ((s-1)(s-2))/((s-3)(s-4))
        system = scipy.signal.lti([1, 2], [3, 4])


        # Setting signal length
        signal_length = 2**8

        # Setting length for starting sine signal
        length = np.pi * 2 * 2

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


        # Sending the signal through our simulated system
        tout, output, xout = scipy.signal.lsim(system, U=I, T=range_inputs)
        
        # Estimate our Impulse response
        IR_est = estimate_IR_weiner_deconvolution(input_signal,output)

        # Generate real IR
        IR = system.impulse()[1]

        data["input"] = input_signal
        data["output"] = output
        data["IR"] = IR
        data["IR_est"] = IR_est


def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_IR_Algorithms('test_eval_weiner_decon'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

    IR_est =  abs(rfft(data["IR_est"]))
    IR =  abs(rfft(data["IR"]))
    A = abs(rfft(data["input"]))
    B = abs(rfft(data["output"]))
    import matplotlib.pyplot as plt
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
    ax1.plot(np.arange(0,len(IR_est)), IR_est)
    ax2.plot(np.arange(0,len(IR)), IR)
    ax3.plot(np.arange(0,len(A)), A)
    ax4.plot(np.arange(0,len(B)), B )
    plt.show()