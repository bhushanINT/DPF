#https://www.analyticsvidhya.com/blog/2023/02/various-techniques-to-detect-and-isolate-time-series-components-using-python/
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html
import numpy as np
import statsmodels.api as sm
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


from utils import extract_PID_data, resample_PID_data

VARIABLES = ['ENGINE RPM','DPFDP','THROTTLE','ENGINE LOAD']

RPM_RANGE = np.array([1250,1800])
THROTTLE_CUTOFF = 0.5
ENGINE_LOAD_CUTOFF = 90
Duty_cycle = 25

GOOD_STATE = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,22,23,24,25,26])

PATH = "D:/Work/Timeseries_models/DATA/DPF_data/"


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

    TEMP = DATA[2,:] # 7th is Throttle
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

def Engine_load_Contraint(DATA,TH):

    TEMP = DATA[3,:] # 7th is Throttle
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

def get_data_A(state_type):

    data_path = os.path.join(PATH, 'A_State/')
    dirs = os.listdir(data_path)

    DP_DATA = np.zeros((len(dirs),25000))
    DP_DATA_T = np.zeros((DP_DATA.shape[0],int(DP_DATA.shape[1]/Duty_cycle)+1))

    #for state_cnt in range(0,len(dirs)):
    for state_cnt in GOOD_STATE:
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
                #plt.plot(np.array(V_L[0]))
                #plt.ylim(0, 2500)
                #plt.show()
                Resample_data = resample_PID_data(V_L,len(VARIABLES)) # cross check by plot
                #plt.plot(Resample_data[0,:])
                #plt.ylim(0, 2500)
                #plt.show()
                #exit(0)
                if (data_packet_cnt == 0):
                    DATA = Resample_data
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)

        #print(DATA.shape)
        X_1 = RPM_Contraint(DATA,RPM_RANGE)
        #print(X_1.shape)
        X_2 = Throttle_Contraint(X_1,THROTTLE_CUTOFF)
        #print(X_2.shape)
        X = Engine_load_Contraint(X_2,ENGINE_LOAD_CUTOFF)
        print(X.shape)
        #exit(0)

        TT = X[1,0:np.count_nonzero(X[1,:])]
        #plt.plot(TT)
        if (len(TT)>10):
            sw_cycle,sw_trend = sm.tsa.filters.hpfilter(TT, lamb=100)
            #plt.plot(sw_trend)
            #plt.show()
            #exit(0)
            DP_DATA[state_cnt,0:len(sw_trend)] = sw_trend

        DP_cnt = 0
        for i in range(0, DP_DATA.shape[1], int(Duty_cycle)):
            TEMP = DP_DATA[state_cnt,i:i+Duty_cycle]
            DP_DATA_T[state_cnt, DP_cnt] = np.mean(TEMP)
            DP_cnt = DP_cnt + 1

    return DP_DATA_T

def process_A(DP_DATA,STATE,Mean_val):

    plt.title("Diffrential Pressure Series") 
    plt.xlabel("Ducty Cycle index") 
    plt.ylabel('Diffrential Pressure in PSI') 
    #plt.plot(DP_DATA[STATE,0:np.count_nonzero(DP_DATA[STATE,:])])   
    PLOT_Data = DP_DATA[STATE,0:np.count_nonzero(DP_DATA[STATE,:])]
    plt.plot(PLOT_Data)
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val)  
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val*1.1)
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val*0.9)    
    plt.show()


def get_data_N(state_type):

    data_path = os.path.join(PATH, 'N_State/')
    dirs = os.listdir(data_path)

    DP_DATA = np.zeros((len(dirs),25000))
    DP_DATA_T = np.zeros((DP_DATA.shape[0],int(DP_DATA.shape[1]/Duty_cycle)+1))

    #for state_cnt in range(0,len(dirs)):
    for state_cnt in GOOD_STATE:
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
                Resample_data = resample_PID_data(V_L,len(VARIABLES))
                if (data_packet_cnt == 0):
                    DATA = Resample_data
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)

        #print(DATA.shape)
        X_1 = RPM_Contraint(DATA,RPM_RANGE)
        #print(X_1.shape)
        X_2 = Throttle_Contraint(X_1,THROTTLE_CUTOFF)
        #print(X_2.shape)
        X = Engine_load_Contraint(X_2,ENGINE_LOAD_CUTOFF)
        print(X.shape)
        #exit(0)

        #plt.plot(X[1,:])    
        #plt.show()
        #exit(0)

        TT = X[1,0:np.count_nonzero(X[1,:])]
        #plt.plot(TT)
        if (len(TT)>10):
            sw_cycle,sw_trend = sm.tsa.filters.hpfilter(TT, lamb=100)
            #plt.plot(sw_trend)
            #plt.show()
            #exit(0)
            DP_DATA[state_cnt,0:len(sw_trend)] = sw_trend

        DP_cnt = 0
        for i in range(0, DP_DATA.shape[1], int(Duty_cycle)):
            TEMP = DP_DATA[state_cnt,i:i+Duty_cycle]
            DP_DATA_T[state_cnt, DP_cnt] = np.mean(TEMP)
            DP_cnt = DP_cnt + 1

    return DP_DATA_T

def process_N(DP_DATA,STATE,Mean_val):

    plt.title("Diffrential Pressure Series") 
    plt.xlabel("Ducty Cycle index") 
    plt.ylabel('Diffrential Pressure in PSI') 
    #plt.plot(DP_DATA[STATE,0:np.count_nonzero(DP_DATA[STATE,:])])   
    PLOT_Data = DP_DATA[STATE,0:np.count_nonzero(DP_DATA[STATE,:])]
    plt.plot(PLOT_Data)
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val)  
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val*1.1)
    plt.plot(np.ones(PLOT_Data.shape)*Mean_val*0.9)    
    plt.show()

def get_mean_band(state_type):

    data_path = os.path.join(PATH, 'N_State/')
    dirs = os.listdir(data_path)
    FLAG = 0

    #for state_cnt in range(0,len(dirs)):
    for state_cnt in range(16,21):
        State_PATH = PATH + state_type + "_State/State" + str(state_cnt) + "/"
        pkt_list = os.listdir(State_PATH)
        print('State cnt---',state_cnt)

        for data_packet_cnt in range(0,len(pkt_list)):
            OBD_data_path = State_PATH + pkt_list[data_packet_cnt] 
            OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
            T_L = []
            V_L = []
            for var_type in VARIABLES:
                X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
                #print(len(X_Value))
                T_L.append(np.array((X_Time), dtype=np.int64))
                V_L.append(np.array((X_Value), dtype=float))

            if (len(V_L[0]) > 100):
                Resample_data = resample_PID_data(V_L,len(VARIABLES))
                if (FLAG == 0):
                    DATA = Resample_data
                    FLAG = 1
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)

    #print(DATA.shape)
    X_1 = RPM_Contraint(DATA,RPM_RANGE)
    #print(X_1.shape)
    X_2 = Throttle_Contraint(X_1,THROTTLE_CUTOFF)
    #print(X_2.shape)
    X = Engine_load_Contraint(X_2,ENGINE_LOAD_CUTOFF)
    print(X.shape)
    #exit(0)

    plt.title("Diffrential Pressure Series") 
    plt.xlabel("Time index") 
    plt.ylabel('Diffrential Pressure in PSI')
    plt.plot(X[1,:])
    Mean_val = np.mean(X[1,:])
    print(Mean_val)
    plt.plot(np.ones(X[1,:].shape)*Mean_val)
    plt.plot(np.ones(X[1,:].shape)*Mean_val*1.1)
    plt.plot(np.ones(X[1,:].shape)*Mean_val*0.9)
    plt.show()
    exit(0)

if __name__ == '__main__':

    #get_mean_band('A')

    STATE = 'N'
    if (STATE == 'A'):
        DATA = get_data_A(STATE)
        for i in range(0,26):
            process_A(DATA,i,2.25)
    else:
        DATA = get_data_N(STATE)
        for i in range(0,26):
            process_N(DATA,i,2.25)

    





