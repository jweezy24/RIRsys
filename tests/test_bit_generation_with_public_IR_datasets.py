from concurrent.futures import thread
import unittest
import scipy
import sys
import os
import random
import argparse

# Adding the src path to our environment so that the functions can be imported to the tests folder
# sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')

from IR_estimation_algorithms import *
from bit_stream_algorithms import *

import numpy as np

data = {}
rooms = []
results = {}

def parse_BUT_datset(path,data,rooms):

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if ".wav" in name and "RIR" in root:
                file_path = os.path.join(root, name)
                lst = file_path.split("/")

                #Parsing file path based on readme
                room = lst[4]
                mic_config = lst[5]
                spk_config = lst[6]
                mic_id = lst[7]
                
                #Adding room to rooms array
                if room not in rooms:
                    print(room)
                    rooms.append(room)


                if room in data.keys():
                    tmp_dict = data[f"{room}"]
                    tmp_dict[f"{mic_config}_{spk_config}_{mic_id}"] = get_raw_audio_stream(file_path)
                    data[f"{room}"] = tmp_dict
                
                else:
                    data[f"{room}"] = {f"{mic_config}_{spk_config}_{mic_id}": get_raw_audio_stream(file_path)}
    return (data,rooms)

def parse_open_air_dataset(path,data,rooms):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if ".wav" in name:
                lst = root.split("/")

                file_path = os.path.join(root, name)

                room = lst[-2]
                fs,IR = get_raw_audio_stream(file_path)

                        
                secs = len(IR)/fs
                samps = int(24000*(secs+1))
                IR = scipy.signal.resample(IR,samps)
                fs = 24000

                #Adding room to rooms array
                if room not in rooms:
                    print(room)
                    rooms.append(room)


                if room in data.keys():
                    tmp_dict = data[f"{room}"]
                    tmp_dict[f"{name}"] = (fs,IR[:24000])
                    data[f"{room}"] = tmp_dict
                
                else:
                    data[f"{room}"] = {f"{name}": (fs,IR[:24000])}
    return data,rooms

def threaded_code_siggs(a,b,c1,c2):
    agreement = bit_agreement_ambient_audio_scheme(a,b)
    return c1,c2,agreement

def threaded_code_windowed_means(a,b,c1,c2):
    agreement = bit_agreement_windowed_means(a,b)
    return c1,c2,agreement

def threaded_code_cosine_distance(a,b,c1,c2,base_wave):
    agreement = bit_agreement_cosine_distance(a,b,base_wave)
    return c1,c2,agreement

def threaded_code_quam(a,b,c1,c2):
    agreement = quam_bit_agreement(a,b)
    return c1,c2,agreement


# The test based on unittest module
class Test_Bit_Generation_Algorithms(unittest.TestCase):

    def test_eval_windowed_means_same_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool

        max_procs = 20
        P = Pool(max_procs)

        for room in rooms:
            curr = data[room]
            
            #Each room will generate a bit agreement matrix
            means_matrix = np.ndarray( (len(data[room].keys()),len(data[room].keys())))
            c1 = 0
            c2 = 0

            futures = []

            for key in curr.keys():
                c2 = 0
                for key2 in curr.keys():

                    a = curr[key][1]
                    b = curr[key2][1]

                    val = P.apply_async(threaded_code_windowed_means, (a,b,c1,c2))

                    futures.append(val)
                    
                    c2+=1
                c1+=1

            for val in futures:
                c1,c2,agreement = val.get()
                means_matrix[c1,c2] = agreement

            print(f"Unpacked {room}")
            print(means_matrix)
            results[f"{room}"] = means_matrix
        
        P.close()

        np.save("data_cache/test_eval_windowed_means_same_room.npy",results)    
        np.save("data_cache/test_eval_windowed_means_same_room2.npy",rooms)
        results = {}


    def test_eval_windowed_means_diff_room(self):
        global data
        global rooms
        global results


        from multiprocessing import Pool
        
        max_procs = 30
        P = Pool(max_procs)
        comps = []

        for i in range(0,len(rooms)):
            for j in range(0,len(rooms)):
                room = rooms[i]
                room2 = rooms[j]
                if room == room2:
                    continue

                curr = data[room]
                curr2 = data[room2]
                #Each room will generate a bit agreement matrix
                means_matrix = np.ndarray( (len(data[room].keys()),len(data[room2].keys())))
                
                c1 = 0
                c2 = 0

                futures = []

                for key in curr.keys():
                    c2 = 0
                    for key2 in curr2.keys():

                        a = curr[key][1]
                        b = curr2[key2][1]

                        val = P.apply_async(threaded_code_windowed_means, (a,b,c1,c2))

                        futures.append(val)
                        
                        c2+=1
                    c1+=1

                for val in futures:
                    c1,c2,agreement = val.get()
                    means_matrix[c1,c2] = agreement

                print(f"Unpacked {room}_{room2}")
                print(means_matrix)
                results[f"{room}_{room2}"] = means_matrix
                comps.append(f"{room}_{room2}")
        
        P.close()

        np.save("data_cache/test_eval_windowed_means_diff_room.npy",results)    
        np.save("data_cache/test_eval_windowed_means_diff_room2.npy",comps)
        results = {}


    def test_eval_cosine_distance_same_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool

        base_wave = None

        
        max_procs = 20
        P = Pool(max_procs)

        for room in rooms:
            curr = data[room]
            
            #Each room will generate a bit agreement matrix
            means_matrix = np.ndarray( (len(data[room].keys()),len(data[room].keys())))
            c1 = 0
            c2 = 0

            futures = []

            for key in curr.keys():
                c2 = 0
                for key2 in curr.keys():

                    a = curr[key][1]
                    b = curr[key2][1]

                    if type(base_wave) == type(None):
                        base_wave = base_sine_wave(np.arange(len(a)))

                    val = P.apply_async(threaded_code_cosine_distance, (a,b,c1,c2,base_wave))

                    futures.append(val)
                    
                    c2+=1
                c1+=1

            for val in futures:
                c1,c2,agreement = val.get()
                means_matrix[c1,c2] = agreement

            print(f"Unpacked {room}")
            print(means_matrix)
            results[f"{room}"] = means_matrix
        
        P.close()

        np.save("data_cache/test_eval_cosine_distance_same_room.npy",results)    
        np.save("data_cache/test_eval_cosine_distance_same_room2.npy",rooms)
        results = {}


    def test_eval_cosine_distance_diff_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool

        base_wave = None

        def threaded_code(a,b,c1,c2,base_wave):
            agreement = bit_agreement_cosine_distance(a,b,base_wave)
            return c1,c2,agreement
        
        max_procs = 30
        P = Pool(max_procs)
        comps = []

        for i in range(0,len(rooms)):
            for j in range(0,len(rooms)):
                room = rooms[i]
                room2 = rooms[j]
                if room == room2:
                    continue

                curr = data[room]
                curr2 = data[room2]
                #Each room will generate a bit agreement matrix
                means_matrix = np.ndarray( (len(data[room].keys()),len(data[room2].keys())))
                
                c1 = 0
                c2 = 0

                futures = []

                for key in curr.keys():
                    c2 = 0
                    for key2 in curr2.keys():

                        a = curr[key][1]
                        b = curr2[key2][1]
                        
                        if type(base_wave) == type(None):
                            base_wave = base_sine_wave(np.arange(len(a)))

                        val = P.apply_async(threaded_code_cosine_distance, (a,b,c1,c2,base_wave))

                        futures.append(val)
                        
                        c2+=1
                    c1+=1

                for val in futures:
                    c1,c2,agreement = val.get()
                    means_matrix[c1,c2] = agreement

                print(f"Unpacked {room}_{room2}")
                print(means_matrix)
                results[f"{room}_{room2}"] = means_matrix
                comps.append(f"{room}_{room2}")
        
        P.close()

        np.save("data_cache/test_eval_cosine_distance_diff_room.npy",results)    
        np.save("data_cache/test_eval_cosine_distance_diff_room2.npy",comps)
        results = {}


    def test_eval_siggs_same_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool
        
        max_procs = 20
        P = Pool(max_procs)

        for room in rooms:
            curr = data[room]
            
            #Each room will generate a bit agreement matrix
            means_matrix = np.ndarray( (len(data[room].keys()),len(data[room].keys())))
            c1 = 0
            c2 = 0

            futures = []

            for key in curr.keys():
                c2 = 0
                for key2 in curr.keys():

                    fs1,a = curr[key]
                    fs2,b = curr[key2]


                    val = P.apply_async(threaded_code_siggs, (a,b,c1,c2))

                    futures.append(val)
                    
                    c2+=1
                c1+=1

            for val in futures:
                c1,c2,agreement = val.get()
                means_matrix[c1,c2] = agreement

            print(f"Unpacked {room}")
            print(means_matrix)
            results[f"{room}"] = means_matrix
        
        P.close()

        np.save("data_cache/test_eval_siggs_same_room.npy",results)    
        np.save("data_cache/test_eval_siggs_same_room2.npy",rooms)
        results = {}


    def test_eval_siggs_diff_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool
        
        max_procs = 30
        P = Pool(max_procs)
        comps = []

        for i in range(0,len(rooms)):
            for j in range(0,len(rooms)):
                room = rooms[i]
                room2 = rooms[j]
                if room == room2:
                    continue

                curr = data[room]
                curr2 = data[room2]
                #Each room will generate a bit agreement matrix
                means_matrix = np.ndarray( (len(data[room].keys()),len(data[room2].keys())))
                
                c1 = 0
                c2 = 0

                futures = []

                for key in curr.keys():
                    c2 = 0
                    for key2 in curr2.keys():

                        a = curr[key][1]
                        b = curr2[key2][1]

                        val = P.apply_async(threaded_code_siggs, (a,b,c1,c2))

                        futures.append(val)
                        
                        c2+=1
                    c1+=1

                
                for val in futures:
                    c1,c2,agreement = val.get()
                    means_matrix[c1,c2] = agreement

                print(f"Unpacked {room}_{room2}")
                print(means_matrix)
                results[f"{room}_{room2}"] = means_matrix
                comps.append(f"{room}_{room2}")
        
        P.close()

        np.save("data_cache/test_eval_siggs_diff_room.npy",results)    
        np.save("data_cache/test_eval_siggs_diff_room2.npy",comps)
        results = {}


    def test_eval_quam_same_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool
        
        max_procs = 20
        P = Pool(max_procs)

        for room in rooms:
            curr = data[room]
            
            #Each room will generate a bit agreement matrix
            means_matrix = np.ndarray( (len(data[room].keys()),len(data[room].keys())))
            c1 = 0
            c2 = 0

            futures = []

            for key in curr.keys():
                c2 = 0
                for key2 in curr.keys():

                    a = curr[key][1]
                    b = curr[key2][1]

                    val = P.apply_async(threaded_code_quam, (a,b,c1,c2))

                    futures.append(val)
                    
                    c2+=1
                c1+=1

            for val in futures:
                c1,c2,agreement = val.get()
                means_matrix[c1,c2] = agreement

            print(f"Unpacked {room}")
            print(means_matrix)
            results[f"{room}"] = means_matrix
        
        P.close()

        np.save("data_cache/test_eval_quam_same_room.npy",results)    
        np.save("data_cache/test_eval_quam_same_room2.npy",rooms)
        results = {}
        time.sleep(2)


    def test_eval_quam_diff_room(self):
        global data
        global rooms
        global results

        from multiprocessing import Pool

        max_procs = 30
        P = Pool(max_procs)
        comps = []

        for i in range(0,len(rooms)):
            for j in range(0,len(rooms)):
                room = rooms[i]
                room2 = rooms[j]
                if room == room2:
                    continue

                curr = data[room]
                curr2 = data[room2]
                #Each room will generate a bit agreement matrix
                means_matrix = np.ndarray( (len(data[room].keys()),len(data[room2].keys())))
                
                c1 = 0
                c2 = 0

                futures = []

                for key in curr.keys():
                    c2 = 0
                    for key2 in curr2.keys():

                        a = curr[key][1]
                        b = curr2[key2][1]

                        val = P.apply_async(threaded_code_quam, (a,b,c1,c2))

                        futures.append(val)
                        
                        c2+=1
                    c1+=1

                for val in futures:
                    c1,c2,agreement = val.get()
                    means_matrix[c1,c2] = agreement

                print(f"Unpacked {room}_{room2}")
                print(means_matrix)
                results[f"{room}_{room2}"] = means_matrix
                comps.append(f"{room}_{room2}")
        
        P.close()

        np.save("data_cache/test_eval_quam_diff_room.npy",results)    
        np.save("data_cache/test_eval_quam_diff_room2.npy",comps)
        results = {}
    

def suite():
    global results

    suite = unittest.TestSuite()

    if not os.path.exists("data_cache/test_eval_windowed_means_same_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_windowed_means_same_room'))
    if not os.path.exists("data_cache/test_eval_windowed_means_diff_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_windowed_means_diff_room'))

    if not os.path.exists("data_cache/test_eval_cosine_distance_same_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_cosine_distance_same_room'))

    if not os.path.exists("data_cache/test_eval_cosine_distance_diff_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_cosine_distance_diff_room'))

    if not os.path.exists("data_cache/test_eval_siggs_same_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_siggs_same_room'))

    if not os.path.exists("data_cache/test_eval_siggs_diff_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_siggs_diff_room'))

    if not os.path.exists("data_cache/test_eval_quam_same_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_quam_same_room'))

    if not os.path.exists("data_cache/test_eval_quam_diff_room.npy"):
        suite.addTest(Test_Bit_Generation_Algorithms('test_eval_quam_diff_room'))
    
    return suite


def visualize_results(results,rooms):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=1, ncols=len(rooms), figsize=(18, 5))
    c = 0
    for room in rooms:
        print(room)
        ax = axs[c]
        d = results[()][room]
        print(d)
        sns.heatmap(d,vmin=0,vmax=1,ax=ax)
        c+=1
    
    plt.show()

def visualize_all(cache_loc):
    print("HERE")
    for root, dirs, files in os.walk(cache_loc, topdown=False):
        for name in files:
            if "test_eval_" in name and "2" not in name and "BUT" not in root:
                res_tmp = np.load(os.path.join(root, f"{name}"),allow_pickle=True)

                n = name.split(".")[0]
                rooms_tmp = np.load(os.path.join(root, f"{n}2.npy"),allow_pickle=True)
                print(rooms_tmp)
                visualize_results(res_tmp,rooms_tmp)
            

if __name__ == "__main__":
    
    # data,rooms = parse_BUT_datset("../audio_datasets/BUT/BUT_ReverbDB",data,rooms)
    
    data,rooms = parse_open_air_dataset("../audio_datasets/OPEN",data,rooms)
    
    runner = unittest.TextTestRunner()
    runner.run(suite())

    visualize_all("data_cache")



