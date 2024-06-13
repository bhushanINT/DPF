
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
            PID_TAG = '91'
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
                    #print(len(State))
                    #exit(0)
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


def Data_bin(X_Time_Vec,X_Value_Vec,Y_Time_Vec,Y_Value_Vec,Num_Bin,Sample_per_Bin):

    Mod_Time_vec = []
    Mod_X_Val_vec = []
    Mod_Y_Val_vec = []

    Common_time_stamp = np.intersect1d(X_Time_Vec, Y_Time_Vec)
    print(len(Common_time_stamp))

    for idx_cnt in range(0,len(Common_time_stamp)):
        Mod_Time_vec.append(Common_time_stamp[idx_cnt])
        IDX1 = np.where(X_Time_Vec == Common_time_stamp[idx_cnt])
        IDY1 = np.where(Y_Time_Vec == Common_time_stamp[idx_cnt])
        #print(IDX1)
        #print(IDY1)
        Mod_X_Val_vec.append(X_Value_Vec[int(IDX1[0][0])])
        Mod_Y_Val_vec.append(Y_Value_Vec[int(IDY1[0][0])])

    BINS = np.arange(min(Mod_X_Val_vec),max(Mod_X_Val_vec), (max(Mod_X_Val_vec)/Num_Bin)).astype(int)
    print(BINS)
    BIN_X_Val_vec = []
    BIN_Y_Val_vec = []
    Flag_Vec = np.zeros(len(BINS)-1)
    for bin_cnt in range(0,len(BINS)-1):
        BIN_VAL_CNT = 0
        for time_cnt in range(0,len(Mod_Time_vec)):
            if ((Mod_X_Val_vec[time_cnt] > BINS[bin_cnt]) and (Mod_X_Val_vec[time_cnt] < BINS[bin_cnt+1]) and (BIN_VAL_CNT < Sample_per_Bin)):
                BIN_X_Val_vec.append(Mod_X_Val_vec[time_cnt])
                BIN_Y_Val_vec.append(Mod_Y_Val_vec[time_cnt])
                BIN_VAL_CNT = BIN_VAL_CNT +1

            if(BIN_VAL_CNT == Sample_per_Bin):
                Flag_Vec[bin_cnt] = 1

    return BIN_X_Val_vec,BIN_Y_Val_vec,Flag_Vec

def RPM_Contraint(RPM_IN,VAL_IN,Th):

    RPM_temp = [] #Temporary list
    VAL_temp = [] #Temporary list

    for cnt in range(0,len(RPM_IN)):
        if (np.array(RPM_IN[cnt]) > Th):
            #print('---------------------------------------')
            #print(RPM[cnt])
            RPM_temp.append(RPM_IN[cnt])
            VAL_temp.append(VAL_IN[cnt])

    return RPM_temp,VAL_temp

####################################  MAIN  #####################################

STATE_TYPE = "A" # A-Active N-Normal

SAMPLE_NUM = "0"  # Total 46 Samples avilable

X_Var = 'THROTTLE'
Y_Var = 'DPFDP'

Num_Bin = 20

Sample_per_Bin = 5


PATH = "D:/Work/Timeseries_models/DATA/DPF_data/" + STATE_TYPE + "_State/State" + SAMPLE_NUM + "/"   

dir_list = os.listdir(PATH)
print(dir_list)


fig, axs = plt.subplots(1, len(dir_list))

for data_packet_cnt in range(0,len(dir_list)):
    
    OBD_data_path = PATH + dir_list[data_packet_cnt] 
    print(OBD_data_path)
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]

    X_Time_Vec, X_Value_Vec = extract_PID_data(OBD_data,'SAE',X_Var, 0)
    #print(len(X_Time_Vec))
    #print(len(X_Value_Vec))

    Y_Time_Vec, Y_Value_Vec = extract_PID_data(OBD_data,'SAE',Y_Var, 0)
    #print(len(Y_Time_Vec))
    #print(len(Y_Value_Vec))

    BIN_X_Val_vec,BIN_Y_Val_vec,Flag_Vec = Data_bin(X_Time_Vec,X_Value_Vec,Y_Time_Vec,Y_Value_Vec,Num_Bin,Sample_per_Bin)
    print(len(BIN_X_Val_vec))
    print(len(BIN_Y_Val_vec))

    BIN_X_Val_vec_lim,BIN_Y_Val_vec_lim = RPM_Contraint(BIN_X_Val_vec,BIN_Y_Val_vec,600) #### check

    print(Flag_Vec)
        
    plt.subplot(1, len(dir_list), data_packet_cnt+1)
    if (len(dir_list) > 1):
        axs[data_packet_cnt].plot(BIN_X_Val_vec_lim,BIN_Y_Val_vec_lim,'b.')
    else:
        axs.plot(BIN_X_Val_vec_lim,BIN_Y_Val_vec_lim,'b.')


if (len(dir_list) > 1):
    axs[0].set(ylabel = Y_Var)

    for data_packet_cnt in range(0,len(dir_list)):
        axs[data_packet_cnt].set_title('Pck' + str(data_packet_cnt))
        axs[data_packet_cnt].set(xlabel = X_Var)

    #for ax in axs.flat:
    #    ax.label_outer()
else:
    axs.set(ylabel = Y_Var)
    axs.set_title('Pck' + str(data_packet_cnt))
    axs.set(xlabel = X_Var)


plt.show()
















