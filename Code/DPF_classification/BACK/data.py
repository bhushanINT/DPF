#https://www.geeksforgeeks.org/python-creating-3d-list/
import os
import glob
import numpy as np
import json
from matplotlib import pyplot as plt 
from scipy import signal
from utils import extract_PID_data, resample_PID_data, min_max_normalize_data, RPM_Contraint, Throttle_Contraint


PATH = "D:/Work/Timeseries_models/DATA/DPF_data/"

STATE_TYPE = ['A','N'] # A-Active N-Normal
VARIABLES = ['ENGINE RPM','DPFDP', 'ATGMF', 'SCRT', 'FUEL RATE', 'IMAP','ENGINE LOAD','THROTTLE']

Time_series_len = 32 # Number of samples considered in duty cycle
Number_of_features = len(VARIABLES) # Number mesured variable considered 
SAFE_BUFFER = 0.5
STEP_FACT = 32 # 1,2,4,8,16,32,64 to 128 --> Increse data
RPM_RANGE = np.array([1250,1800])
THROTTLE_CUTOFF = 0.75

def create_train_data(PATH,STATE_TYPE,VARIABLES):
    
    train_data_path = os.path.join(PATH, 'A_State/')

    dirs = os.listdir(train_data_path)

    COUNT = 0
    LABEL = []


    for state_type in STATE_TYPE:
        #for state_cnt in range(0,len(dirs)):
        for state_cnt in range(0,23):
            State_PATH = PATH + state_type + "_State/State" + str(state_cnt) + "/"
            pkt_list = os.listdir(State_PATH)

            for data_packet_cnt in range(0,len(pkt_list)):
                OBD_data_path = State_PATH + pkt_list[data_packet_cnt] 
                #print(OBD_data_path)
                OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
                T_L = []
                V_L = []
                for var_type in VARIABLES:
                    X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
                    #print(len(X_Value))
                    T_L.append(np.array((X_Time), dtype=np.int64))
                    V_L.append(np.array((X_Value), dtype=float))


                if (len(V_L[0]) > 100):
                    Resample_data = resample_PID_data(V_L,Number_of_features)
                    #print(Resample_data.shape)

                if (data_packet_cnt == 0):
                    DATA = Resample_data
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)

            #print(DATA.shape)
            #exit(0)

            BUF_LIM = int(SAFE_BUFFER*DATA.shape[1]) ##################BUFFER LIMIT
            print(BUF_LIM)
            #exit(0)
            np.save(PATH + 'Train_Data/' + str(COUNT) + '.npy', Resample_data[:,0:BUF_LIM])
            LABEL.append(state_type)
            COUNT = COUNT + 1
            #exit(0)

    np.save(PATH + 'Train_Data/' + 'LABEL.npy', LABEL)

    return 


def load_train_data():

    #create_train_data(PATH,STATE_TYPE,VARIABLES)
    #exit(0)

    Y = np.load(PATH + 'Train_Data/' + 'LABEL.npy')

    LABEL = []

    FLAG = 0

    for cnt in range(0,46): # set
        X_T = np.load(PATH + 'Train_Data/' + str(cnt) +'.npy')

        #print(X_T.shape)
        X_1 = RPM_Contraint(X_T,RPM_RANGE)
        #print(X_1.shape)
        X = Throttle_Contraint(X_1,THROTTLE_CUTOFF)
        #plt.plot(X[1,:]) 
        #plt.show()
        #print(X.shape,'-----------------')
        #exit(0)

        for i in range(0, X.shape[1], int(Time_series_len/STEP_FACT)):

            TEMP = X[:,i:i+Time_series_len]

            if (TEMP.shape[1]==Time_series_len):
                if (FLAG == 0):
                    DATA = TEMP
                    DATA = np.expand_dims(DATA, axis=0)
                    FLAG = 1
                    if (Y[cnt] == 'A'):
                        LABEL.append(1)
                    else:
                        LABEL.append(0)
                else:
                    DATA = np.concatenate((DATA,np.expand_dims(TEMP, axis=0)),axis=0)
                    if (Y[cnt] == 'A'):
                        LABEL.append(1)
                    else:
                        LABEL.append(0)


    print(DATA.shape)
    #print(DATA[1,0,1:10])
    DATA_NORM = min_max_normalize_data(DATA)
    #print(DATA_NORM[1,0,1:10])
    print(np.array(LABEL).shape)
    #exit(0)
    return DATA_NORM, np.array(LABEL)


def create_test_data(PATH,STATE_TYPE,VARIABLES):
    
    train_data_path = os.path.join(PATH, 'A_State/')

    dirs = os.listdir(train_data_path)

    COUNT = 0
    LABEL = []


    for state_type in STATE_TYPE:
        #for state_cnt in range(0,len(dirs)):
        for state_cnt in range(23,27):
            State_PATH = PATH + state_type + "_State/State" + str(state_cnt) + "/"
            pkt_list = os.listdir(State_PATH)

            for data_packet_cnt in range(0,len(pkt_list)):
                OBD_data_path = State_PATH + pkt_list[data_packet_cnt] 
                #print(OBD_data_path)
                OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
                T_L = []
                V_L = []
                for var_type in VARIABLES:
                    X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
                    #print(len(X_Value))
                    T_L.append(np.array((X_Time), dtype=np.int64))
                    V_L.append(np.array((X_Value), dtype=float))


                if (len(V_L[0]) > 100):
                    Resample_data = resample_PID_data(V_L,Number_of_features)
                    #print(Resample_data.shape)
                    #exit(0)

                if (data_packet_cnt == 0):
                    DATA = Resample_data
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)

            #print(DATA.shape)
            #exit(0)

            BUF_LIM = int(SAFE_BUFFER*DATA.shape[1]) ##################BUFFER LIMIT
            print(BUF_LIM)
            #exit(0)
            np.save(PATH + 'Test_Data/' + str(COUNT) + '.npy', Resample_data[:,0:BUF_LIM])
            LABEL.append(state_type)
            COUNT = COUNT + 1
            #exit(0)

    np.save(PATH + 'Test_Data/' + 'LABEL.npy', LABEL)

    return 


def load_test_data():

    #create_test_data(PATH,STATE_TYPE,VARIABLES)
    #exit(0)

    Y = np.load(PATH + 'Test_Data/' + 'LABEL.npy')

    LABEL = []

    FLAG = 0

    for cnt in range(0,8):
        X_T = np.load(PATH + 'Test_Data/' + str(cnt) +'.npy')
        X_1 = RPM_Contraint(X_T,RPM_RANGE)
        #print(X_T.shape)
        X = Throttle_Contraint(X_1,THROTTLE_CUTOFF)
        #plt.plot(X[1,:]) 

        for i in range(0, X.shape[1], Time_series_len):

            TEMP = X[:,i:i+Time_series_len]

            if (TEMP.shape[1]==Time_series_len):
                if (FLAG == 0):
                    DATA = TEMP
                    DATA = np.expand_dims(DATA, axis=0)
                    FLAG = 1
                    if (Y[cnt] == 'A'):
                        LABEL.append(1)
                    else:
                        LABEL.append(0)
                else:
                    DATA = np.concatenate((DATA,np.expand_dims(TEMP, axis=0)),axis=0)
                    if (Y[cnt] == 'A'):
                        LABEL.append(1)
                    else:
                        LABEL.append(0)


    print(DATA.shape)
    #print(DATA[1,0,1:10])
    DATA_NORM = min_max_normalize_data(DATA)
    #print(DATA_NORM[1,0,1:10])
    #print(np.array(LABEL).shape)
    #exit(0)
    return DATA_NORM, np.array(LABEL)

