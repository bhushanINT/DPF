
# Python program to read json file

import numpy as np
import json
from matplotlib import pyplot as plt 
from scipy import signal
import os

def extract_PID_data(data, PROTOCOL,LABEL,PLOT_FLAG):
    
    if (PROTOCOL == 'SAE'):

        if LABEL == 'IMAP':
            PID_TAG = '106'
        elif LABEL == 'ENGINE LOAD':
            PID_TAG = '92'
        elif LABEL == 'ENGINE RPM':
            PID_TAG = '190'
        elif LABEL == 'FUEL RATE':
            PID_TAG = '183'
        elif LABEL == 'MAF':
            PID_TAG = '132'
        elif LABEL == 'BOOST':
            PID_TAG = '102'
        elif LABEL == 'BAROMETER':
            PID_TAG = '108'
        elif LABEL == 'THROTTLE':
            PID_TAG = '91'
        elif LABEL == 'ATGMF': # Eaxuast Gas Flow Rate
            PID_TAG = '3236'
        elif LABEL == 'DPFDP': # DPF Diffrential Pressure
            PID_TAG = '3251'
        elif LABEL == 'SCRT':   # SCR Catalyst Temperature Before Catalyst
            PID_TAG = '4360'

    elif(PROTOCOL == 'ISO'):

        if LABEL == 'IMAP':
            PID_TAG = '0B, 87BC'#MODIFY the "if PID_TAG in State1:" loop (append for loop on top)  
        elif LABEL == 'ENGINE LOAD':
            PID_TAG = '04'
        elif LABEL == 'ENGINE RPM':
            PID_TAG = '0C'
        elif LABEL == 'FUEL RATE':
            PID_TAG = '5E'
        elif LABEL == 'MAF':
            PID_TAG = '10'
        elif LABEL == 'BOOST':
            PID_TAG = '102'
        elif LABEL == 'BAROMETER':
            PID_TAG = '33'
        elif LABEL == 'THROTTLE':
            PID_TAG = '11, 47, 48, 49, 4A, 4B'#MODIFY the "if PID_TAG in State1:" loop (append for loop on top)  

    print("PID TAG is-->" + LABEL + " ," +PROTOCOL,"Mapping is --->", PID_TAG)

    Time_vec = []
    Val_vec = []
    #print(len(data[0]))
    #data = data[0]
    print(len(data))
    for data_cnt in range(0,len(data[0])):
        if "pids" in data[0][data_cnt]:
            if len(data[0][data_cnt]['pids'])>0:
                for sub_pid_cnt in range(0,len(data[0][data_cnt]['pids'])):  #this loop
                    State = data[0][data_cnt]['pids'][sub_pid_cnt]
                    #print(State)
                    #print(data_cnt)
                    #for state_cnt in range(0,len(State)):
                    if PID_TAG in State:
                        #print("--------------------------------------------IN----------------------------------")
                        Time_vec.append(np.array(State[PID_TAG]['timestamp'], dtype=np.int64))
                        Val_vec.append(np.array(State[PID_TAG]['value'], dtype=float))

    print("Number of time stamp avilables are--->",len(Val_vec))

    if (PLOT_FLAG == 1):            
        plt.title("Time Series") 
        plt.xlabel("Time Stamp") 
        plt.ylabel(LABEL) 
        plt.scatter(Time_vec,Val_vec) 
        plt.show()
                
    return Time_vec,Val_vec



STATE_TYPE = "A" # A-Active N-Normal

SAMPLE_NUM = "0"  # Total 46 Samples avilable

PATH = "D:/Work/Timeseries_models/DATA/DPF_data/" + STATE_TYPE + "_State/State" + SAMPLE_NUM + "/"   

dir_list = os.listdir(PATH)
print(dir_list)

VARIABLES = ['ENGINE RPM','DPFDP', 'ATGMF', 'SCRT', 'FUEL RATE', 'IMAP','ENGINE LOAD','THROTTLE']

fig, axs = plt.subplots(len(VARIABLES), len(dir_list))

for data_packet_cnt in range(0,len(dir_list)):
    OBD_data_path = PATH + dir_list[data_packet_cnt] 
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]

    for var_cnt in range(0,len(VARIABLES)):
        VAR = VARIABLES[var_cnt]
        Time_Vec, Value_Vec = extract_PID_data(OBD_data,'SAE',VAR, 0)
        print(len(Time_Vec))
        print(len(Value_Vec))
        #exit(0)
    
        #plt.subplot(var_cnt + 1, len(dir_list), data_cnt+1)
        axs[var_cnt, data_packet_cnt].plot(Time_Vec,Value_Vec,'b.')
    #exit(0)

for var_cnt in range(0,len(VARIABLES)):
    axs[var_cnt, 0].set(ylabel = VARIABLES[var_cnt])

for data_packet_cnt in range(0,len(dir_list)):
    axs[0, data_packet_cnt].set_title('Pck' + str(data_packet_cnt))

#for ax in axs.flat:
#    ax.label_outer()

plt.show()

exit(0)














