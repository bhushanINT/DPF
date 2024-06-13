
# Python program to read json file

import numpy as np
import json
from matplotlib import pyplot as plt 

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

    elif(PROTOCOL == 'ISO'):

        if LABEL == 'IMAP':
            PID_TAG = '0B, 87BC'#MODIFY the "if PID_TAG in State1:" loop (append for loop on top)  
        elif LABEL == 'ENGINE LOAD':
            PID_TAG = '4'
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
    #print(len(data))
    for data_cnt in range(0,len(data)):
        State = data[data_cnt]['pids']
        for state_cnt in range(0,len(State)):
            State1 = State[state_cnt]
            if PID_TAG in State1:
                Time_vec.append(np.array(State1[PID_TAG]['timestamp'], dtype=float))
                Val_vec.append(np.array(State1[PID_TAG]['value'], dtype=float))

    print("Number of time stamp avilables are--->",len(Val_vec))

    if (PLOT_FLAG == 1):            
        plt.title("Time Series") 
        plt.xlabel("Time Stamp") 
        plt.ylabel(LABEL) 
        plt.scatter(Time_vec,Val_vec) 
        plt.show()
                
    return Time_vec,Val_vec


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

# Path of JSON file
PATH='D:/Work/Timeseries_models/DATA/trips_data/obddata/1685507123000-1067765443051126784-1001789242914897920.json'
data = [json.loads(line) for line in open(PATH, 'r')]

Protocol = data[0]['protocol'][0:3]

print("Extracted OBD protocol is--->",Protocol)

Time_TR, Value_TR = extract_PID_data(data,Protocol,'BOOST',1)
#Time_RPM, Value_RPM = extract_PID_data(data,Protocol,'ENGINE RPM',0)
#print(Time_TR[0:10])
#print(Time_RPM[0:10])
#print(np.array(Time_TR[0:10])/np.array(Time_RPM[0:10]))
#plt.scatter(Time_TR[0:100],Time_RPM[0:100]) 
#plt.show()
#A = np.array([1,2,3,4,5])
#print(A)
#B = resize_zoh_vec(A, 3)
#print(B)







