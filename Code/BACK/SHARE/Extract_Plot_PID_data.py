import numpy as np
from matplotlib import pyplot as plt 
import os
import json
import statsmodels.api as sm

def extract_PID_data(data, PROTOCOL,LABEL):
    
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
        elif LABEL == 'DPFDP': # DPF Diffrential Pressure (across DPF)
            PID_TAG = '3251'
        elif LABEL == 'SCRT':   # SCR Catalyst Temperature Before Catalyst (DPF out)
            PID_TAG = '4360'
        elif LABEL == 'SPEED': # Wheep based Vehicle Speed
            PID_TAG = '84'
        elif LABEL == 'DPFINT':# DPF in Temperature Before DPF (DOC out)
            PID_TAG = '4766'
        elif LABEL == 'IS': #  regen inhibited
            PID_TAG = '3703'        
        elif LABEL == 'FUEL USE': #  Total fuel used (high precision)
            PID_TAG = '5054'        
        elif LABEL == 'DISTANCE': #  Total distance travelled (high precision)
            PID_TAG = '245'

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
    #print(len(data))
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

    
    return Time_vec,Val_vec

if __name__ == '__main__':
    OBD_data_path = '1653912000000'
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
    
    #"ENGINE RPM"
    #"DPFDP" : DPF Diffrential Pressure (across DPF)
    #"THROTTLE"
    #"ENGINE LOAD"
    #"SCRT" : SCR Catalyst Temperature Before Catalyst (DPF out)
    #"SPEED" : Wheel based Vehicle Speed
    #"DPFINT" : DPF in Temperature Before DPF (DOC out)
    #"IS" :regen inhibition switch
    #"FUEL USE" : Total fuel used (high precision)
    #"DISTANCE" : Total distance travelled  (high precision)
    #"ATGMF" : Eaxuast Gas Flow Rate

    LABEL = 'DPFINT'

    X_Time, X_Value = extract_PID_data(OBD_data,'SAE',LABEL)
    plt.title("Time Series") 
    plt.xlabel("Time Stamp") 
    plt.ylabel(LABEL) 
    plt.plot(np.array(X_Time),np.array(X_Value)) 
    plt.show()

'''
    DATA = np.array(X_Value)
    sw_cycle,temp_trend = sm.tsa.filters.hpfilter(DATA, lamb=10)

    sw_cycle,Temp = sm.tsa.filters.hpfilter(DATA[0:200], lamb=10)

    DATA2 =  DATA
    DATA2[0:200,0] = Temp

    plt.plot(DATA,'+b') 
    plt.plot(temp_trend,'*g') 
    plt.plot(DATA2,'r') 
    plt.show()
'''