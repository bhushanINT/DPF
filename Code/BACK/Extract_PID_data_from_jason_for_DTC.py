
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
    #PID_TAG = "3236"
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
LOG_PATH = 'D:/Work/Timeseries_models/DATA/Data2/response.json'
OBD_PATH = 'D:/Work/Timeseries_models/DATA/Data2/obd-888373821327802368.json'
LOG_data = [json.loads(line) for line in open(LOG_PATH, 'r')]
OBD_data = [json.loads(line) for line in open(OBD_PATH, 'r')]

Protocol = OBD_data[0][0]['protocol'][0:3]
print("Extracted OBD protocol is--->",Protocol)

#print(len(OBD_data[0][0]['pids'][0]))
#print(OBD_data[0][104]['pids'][6])
#exit(0)

Time_Vec, Value_Vec = extract_PID_data(OBD_data,Protocol,'IMAP',1)
#exit(0)


#for x in OBD_data[0][0]['pids']:
#    keys = x.keys()
#    print(keys)



#print(Time_TR[0:10])

#print(len(LOG_data[0]['result']['output']))

State = LOG_data[0]['result']['output']

AA_state_start = []
AA_state_end = []

STATE = "removed"
Prev_State = "None"
for state_cnt in range(0,len(State)):
    #print(State[state_cnt]['fault_log.status'])
    if ((State[state_cnt]['fault_log.status'] == STATE) and  (Prev_State == STATE)):
        #print('---------------------------------------')
        SS = State[state_cnt]['fault_log.timestamp']
        date = datetime.datetime(int(SS[0:4]), int(SS[5:7]), int(SS[8:10]), int(SS[11:13]), int(SS[14:16]),int(SS[17:19]))
        AA_state_end.append(datetime.datetime.timestamp(date))
        ES = State[state_cnt-1]['fault_log.timestamp']
        date = datetime.datetime(int(ES[0:4]), int(ES[5:7]), int(ES[8:10]), int(ES[11:13]), int(ES[14:16]),int(ES[17:19]))
        AA_state_start.append(datetime.datetime.timestamp(date))
    Prev_State = State[state_cnt]['fault_log.status']
#print(Prev_State)


print(Time_Vec[0])
print(Time_Vec[-1])

print(AA_state_start[0])
print(AA_state_end[0])

Time_indx = np.where(np.logical_and(np.array(Time_Vec)>=(np.array(AA_state_start[0])*1000), np.array(Time_Vec)<=(np.array(AA_state_end[0])*1000)))

print(np.array(Time_indx[0]).shape)

Active_state_data = Value_Vec[int(Time_indx[0][0]):int(Time_indx[0][-1])]

plt.hist(np.array(Active_state_data), color='lightgreen', ec='black', bins=25)
plt.show()


#from datetime import datetime
#date = datetime.datetime(2023, 6, 28, 17, 38,12)#YYYY/MM/DD/HR/MIN/SEC
#datetime.datetime.timestamp(date)
#2023-06-28T22:44:03.000Z
    
#Protocol = data[0]['protocol'][0:3]

#print("Extracted OBD protocol is--->",Protocol)

#Time_TR, Value_TR = extract_PID_data(data,Protocol,'BOOST',1)
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







