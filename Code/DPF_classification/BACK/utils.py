import numpy as np
import json
from matplotlib import pyplot as plt 
import os
from scipy import signal


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

    #print("PID TAG is-->" + LABEL + " ," +PROTOCOL,"Mapping is --->", PID_TAG)

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
                    #print(len(State))
                    #exit(0)
                    #print(data_cnt)
                    #for state_cnt in range(0,len(State)):
                    if PID_TAG in State:
                        #print("--------------------------------------------IN----------------------------------")
                        Time_vec.append(np.array(State[PID_TAG]['timestamp'], dtype=np.int64))
                        Val_vec.append(np.array(State[PID_TAG]['value'], dtype=float))

    #print("Number of time stamp avilables are--->",len(Val_vec))

    if (PLOT_FLAG == 1):            
        plt.title("Time Series") 
        plt.xlabel("Time Stamp") 
        plt.ylabel(LABEL) 
        plt.scatter(Time_vec,Val_vec) 
        plt.show()
                
    return Time_vec,Val_vec


def resample_PID_data(X_Value,Number_of_features):

    # DPFDP has minimum length DATA[1,:], so resample all other variables  w.r.t. DPFDP
    Samples = len(X_Value[1])
    #print(len(X_Value[0]))
    DATA = np.zeros((Number_of_features,Samples))
    TEMP = np.transpose(np.array(X_Value[1]))
 
    for cnt in range(0,Number_of_features):
        if (len(X_Value[cnt]) > 0):
            DATA[cnt,:] = np.transpose(np.array(signal.resample(X_Value[cnt], Samples)))

    DATA[1,:] = TEMP

    return DATA

def min_max_normalize_data(DATA):

    Features = DATA.shape[1]
    #print(Features)
    #exit(0)

    DATA_OUT = np.zeros(DATA.shape)

    for cnt in range(0,Features):

        Feat_Selected = DATA[:,cnt,:]

        MIN = min(Feat_Selected.flatten())
        #print(MIN)
        MAX = max(Feat_Selected.flatten())
        #print(MAX)

        DATA_OUT[:,cnt,:] = ((DATA[:,cnt,:] - MIN)/ (MAX-MIN))

        #exit(0)

    return DATA_OUT

def RPM_Contraint(DATA,RANGE):

    TEMP = DATA[0,:] # 0th is RPM
    nums = TEMP[(RANGE[0] < TEMP) & (TEMP <= RANGE[1])]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if ((RANGE[0] < TEMP[cnt]) and (TEMP[cnt] <= RANGE[1])):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT
    

def Throttle_Contraint(DATA,TH):

    TEMP = DATA[7,:] # 7th is Throttle
    nums = TEMP[(TH < TEMP)]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if (TH < TEMP[cnt]):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT



    