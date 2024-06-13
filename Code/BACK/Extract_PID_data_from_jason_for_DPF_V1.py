
# Python program to read json file

import numpy as np
import json
from matplotlib import pyplot as plt 
import datetime

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
                    #print(data_cnt)
                    for state_cnt in range(0,len(State)):
                        if PID_TAG in State:
                            #print("--------------------------------------------IN----------------------------------")
                            Time_vec.append(np.array(State[PID_TAG]['timestamp'], dtype=float))
                            Val_vec.append(np.array(State[PID_TAG]['value'], dtype=float))

    print("Number of time stamp avilables are--->",len(Val_vec))

    if (PLOT_FLAG == 1):            
        plt.title("Time Series") 
        plt.xlabel("Time Stamp") 
        plt.ylabel(LABEL) 
        plt.scatter(Time_vec,Val_vec) 
        plt.show()
                
    return Time_vec,Val_vec

def extract_PID_data_using_timestamp(OBD_data, Start_time, End_time, PROTOCOL,LABEL,PLOT_FLAG):

    Time_Vec, Value_Vec = extract_PID_data(OBD_data,PROTOCOL,LABEL,0)

    if (len(Time_Vec)):

        print(Start_time)
        print(End_time)

        print(Time_Vec[0])
        print(Time_Vec[-1])
        #exit(0)

        Time_indx = np.where(np.logical_and(np.array(Time_Vec)>=(np.array(Start_time)*1000), np.array(Time_Vec)<=(np.array(End_time)*1000)))

        print(np.array(Time_indx[0]).shape)

        if ((np.array(Time_indx[0]).shape[0]) == 0):
            print("No Overlap ----")
            Active_state_data = []
        else:
            Active_state_data = Value_Vec[int(Time_indx[0][0]):int(Time_indx[0][-1])]
    else:
        Active_state_data = []


    return Active_state_data

def resize_zoh_vec(s, N):

    y = np.zeros(N*len(s))

    for i in range(0, len(s)):
        y[i*N] = s[i]
    #print(y)
    for j in range(1, len(y)):
        if (y[j]==0):
            y[j] = y[j-1]
        else:
            y[j]=y[j]

    return y


def extract_timestamps_from_log_data(LOG_data,VEHICLE_ID,STATE_ID):

    State = LOG_data[0]['result']['output']

    state_start = []
    state_end = []

    STATE = STATE_ID

    if(STATE == 'active'):
        REF_STATE = 'removed'
    else:
        REF_STATE = 'active'

    Prev_State = "None"
    for state_cnt in range(0,len(State)):
        #print(State[state_cnt]['fault_log.status'])
        if (State[state_cnt]['vehicle_id'] == VEHICLE_ID):
            if ((State[state_cnt]['fault_log.status'] == REF_STATE) and  (Prev_State == STATE)):
                #print('---------------------------------------')
                SS = State[state_cnt]['fault_log.timestamp']
                S_date = datetime.datetime(int(SS[0:4]), int(SS[5:7]), int(SS[8:10]), int(SS[11:13]), int(SS[14:16]),int(SS[17:19]))
                ES = State[state_cnt-1]['fault_log.timestamp']
                E_date = datetime.datetime(int(ES[0:4]), int(ES[5:7]), int(ES[8:10]), int(ES[11:13]), int(ES[14:16]),int(ES[17:19]))
                #print(datetime.datetime.timestamp(S_date))
                #print(datetime.datetime.timestamp(E_date))

                TimeStamp_avail = datetime.datetime.timestamp(S_date) - datetime.datetime.timestamp(E_date)
                #print(TimeStamp_avail)
                #exit(0)
                if (TimeStamp_avail>5000):
                    state_end.append(datetime.datetime.timestamp(S_date))
                    state_start.append(datetime.datetime.timestamp(E_date))
            Prev_State = State[state_cnt]['fault_log.status']
        #print(Prev_State)
    return state_start, state_end




# Path of JSON file
LOG_PATH = 'D:/Work/Timeseries_models/DATA/Data2/response.json'
LOG_data = [json.loads(line) for line in open(LOG_PATH, 'r')]

veichle = ["888373821327802368","946314396685041664","858707647044517888","944887316176961536","893850664696807424","906841549801783296"]

A_state = []


for vc_indx in veichle:
    A_state_start, A_state_end = extract_timestamps_from_log_data(LOG_data,vc_indx,'active')
    #print('---------------------------------------')
    print(len(A_state_start))
    print(len(A_state_end))
    #print('---------------------------------------')
    OBD_PATH = 'D:/Work/Timeseries_models/DATA/Data2/obd-' + vc_indx + '.json'
    OBD_data = [json.loads(line) for line in open(OBD_PATH, 'r')]
    Protocol = OBD_data[0][0]['protocol'][0:3]
    print("Extracted OBD protocol is--->",Protocol)

    for data_frm_cnt in range(0,len(A_state_start)):
        data_frm_cnt
        DATA = extract_PID_data_using_timestamp(OBD_data, A_state_start[data_frm_cnt],A_state_end[data_frm_cnt],Protocol,'ENGINE RPM',0)
        if (len(DATA)):
            A_state.append(DATA)

#exit(0)

R_state = []

for vc_indx in veichle:
    R_state_start, R_state_end = extract_timestamps_from_log_data(LOG_data,vc_indx,'removed')
    print(len(R_state_start))
    print(len(R_state_end))
    OBD_PATH = 'D:/Work/Timeseries_models/DATA/Data2/obd-' + vc_indx + '.json'
    OBD_data = [json.loads(line) for line in open(OBD_PATH, 'r')]
    Protocol = OBD_data[0][0]['protocol'][0:3]
    print("Extracted OBD protocol is--->",Protocol)

    for data_frm_cnt in range(0,len(R_state_start)):
        data_frm_cnt
        DATA = extract_PID_data_using_timestamp(OBD_data, R_state_start[data_frm_cnt],R_state_end[data_frm_cnt],Protocol,'ENGINE RPM',0)
        if (len(DATA)):
            R_state.append(DATA)
    

print(len(A_state))
print(len(R_state))

#exit(0)

for i in range(len(A_state)):
    plt.figure()
    plt.plot(A_state[i])
    fig1 = plt.gcf()
    fig1.savefig("D:/Work/Timeseries_models/DATA/Data2/FIGs/A_state_RPM_%03d.png"%(i))

for i in range(len(R_state)):
    plt.figure()
    plt.plot(R_state[i])
    fig1 = plt.gcf()
    fig1.savefig("D:/Work/Timeseries_models/DATA/Data2/FIGs/R_state_RPM_%03d.png"%(i))

#exit(0)









