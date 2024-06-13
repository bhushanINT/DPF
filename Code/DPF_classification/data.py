import os
import glob
import numpy as np
import json
import statsmodels.api as sm
from matplotlib import pyplot as plt 
from scipy import signal
from utils import extract_PID_data, resample_PID_data, retain_monotonicity, RPM_Contraint, Throttle_Contraint, Engine_load_Contraint,idle_Contraint,IS_Contraint,min_max_normalize_data


PATH = "D:/Work/Timeseries_models/DATA/FULL_DATA/"

VARIABLES = ["ENGINE RPM","DPFDP","THROTTLE","ENGINE LOAD", "SCRT","SPEED","DPFINT","IS","FUEL USE","DISTANCE"]

Scale_fact = 4
Time_series_len = 32*Scale_fact # Number of samples considered in duty cycle
Number_of_features = len(VARIABLES) # Number mesured variable considered 

RPM_RANGE = np.array([1250,1800])
THROTTLE_CUTOFF = 0.5
ENGINE_LOAD_CUTOFF = 90 
ENGINE_IDLE_SWITCH = 0
REGN_INHIBT_SWITCH = 0.5

DP_HIGH_MAX = 3.5          # Maximum Threshold for high diffrential pressure
DP_HIGH_MIN = 3            # Minimum Threshold for high diffrential pressure
DP_LOW_MAX = 0.75          # Maximum Threshold for low diffrential pressure
DP_LOW_MIN = 1             # Minimum Threshold for low diffrential pressur
SCR_TEMP_HIGH_MAX = 450    # Maximum Threshold for high SCR Temprature
SCR_TEMP_HIGH_MIN = 425    # Minimum Threshold for high SCR Temprature
SCR_TEMP_LOW_MIN = 300     # Maximum Threshold for low SCR Temprature
SCR_TEMP_LOW_MAX = 275     # Minimum Threshold for low SCR Temprature
Sample_window = 25         # Sample to be consider for analyesis
Num_H_DP_cnt = 1          # Number of intances crossing high diffrential pressure limit
Num_M_DP_cnt = 3           # Number of intances crossing low diffrential pressure limit
Num_H_T_cnt = 1            # Number of intances crossing high SCR Temprature limit
Num_M_T_cnt = 10           # Number of intances crossing low SCR Temprature limit


def state_detection(DP,temp_trend):

    FLAG = 'REMOVE'

    A_TS = []
    N_TS = []

    DP_level = np.zeros(len(DP))
    TEMP_level = np.zeros(len(DP))

    for cnt in range(Sample_window,len(DP)):

        DP_BUF = DP[cnt-Sample_window:cnt]
        TMP_BUF = temp_trend[cnt-Sample_window:cnt]

        if ((FLAG == 'REMOVE') and (DP[cnt] > DP_HIGH_MAX)):
            H_P = sum(DP_BUF>DP_HIGH_MAX)
            M_P = sum(DP_BUF>DP_HIGH_MIN)
            H_T = sum(TMP_BUF>SCR_TEMP_HIGH_MAX)
            M_T = sum(TMP_BUF>SCR_TEMP_HIGH_MIN)

            if ((H_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (H_T >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt)):
                FLAG = 'ACTIVE'
                A_TS.append(cnt)

        if ((FLAG == 'ACTIVE') and (DP[cnt] < DP_LOW_MAX)):
            L_P = sum(DP_BUF<DP_LOW_MAX)
            M_P = sum(DP_BUF<DP_LOW_MIN)
            L_T = sum(TMP_BUF<SCR_TEMP_LOW_MAX)
            M_T = sum(TMP_BUF<SCR_TEMP_LOW_MIN)

            if ((L_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (L_P >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt)):
                FLAG = 'REMOVE'
                N_TS.append(cnt)


    return np.array(A_TS), np.array(N_TS)


def quality_of_soot_burn(DP,temp_trend):

    Sample_window  = 50

    BURN_Quality = np.zeros(len(DP))
    DP_Fall = np.zeros(len(DP))


    for cnt in range(2*Sample_window,len(DP)):
        TMP_BUF = temp_trend[(cnt-2*Sample_window):cnt-Sample_window]
        H_T = sum(TMP_BUF>SCR_TEMP_HIGH_MAX)
        M_T = sum(TMP_BUF>SCR_TEMP_HIGH_MIN)

        DP_BUF_PRE = DP[(cnt-2*Sample_window):cnt-Sample_window]
        DP_BUF_POST = DP[cnt-Sample_window:cnt]

        Pre_HH_P = sum(DP_BUF_PRE>DP_HIGH_MAX)
        Pre_HM_P = sum(DP_BUF_PRE>DP_HIGH_MIN)

        Post_LH_P = sum(DP_BUF_POST<DP_LOW_MAX)
        Post_LM_P = sum(DP_BUF_POST<DP_LOW_MIN)

        DP_Pre_mean = np.mean(DP_BUF_PRE)
        DP_Post_mean = np.mean(DP_BUF_POST)

        DP_Fall[cnt] = DP_Pre_mean - DP_Post_mean

        if ((H_T > 3) and (M_T > 7) and (Pre_HH_P > 3) and (Pre_HM_P > 7) and (Post_LH_P > 3) and (Post_LM_P > 7)):
            BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 1
        elif ((M_T > 7) and (Pre_HH_P > 1) and (Pre_HM_P > 7) and (Post_LH_P > 1) and (Post_LM_P > 7)):
            BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 1
        elif ((M_T > 5) and (Pre_HM_P > 5) and (Post_LM_P > 5)):
            BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 0.66
        elif ((M_T > 5) and (Pre_HM_P > 3) and (Post_LM_P > 3)):
            BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 0.33
        elif ((M_T > 7) and (Pre_HM_P > 7) and (Post_LM_P < 1)): # Missed Burning
            BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = -1

    return BURN_Quality, DP_Fall

def create_train_data(PATH,VARIABLES):
    

    for vehicle_cnt in range(1,10):

        Data_PATH = PATH + "V" + str(vehicle_cnt) + "/"
        pkt_list = os.listdir(Data_PATH)

        for data_packet_cnt in range(0,len(pkt_list)):
            OBD_data_path = Data_PATH + pkt_list[data_packet_cnt] 
            OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
            T_L = []
            V_L = []
            for var_type in VARIABLES:
                X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
                T_L.append(np.array((X_Time), dtype=np.int64))
                V_L.append(np.array((X_Value), dtype=float))

            if (len(V_L[1]) > 0):
                Temp_time = np.array(T_L[1])
                Resample_data = resample_PID_data(V_L,Number_of_features)
                Resample_data[8,:] = retain_monotonicity(Resample_data[8,:])
                Resample_data[9,:] = retain_monotonicity(Resample_data[9,:])

                if (data_packet_cnt == 0):
                    DATA = Resample_data
                    T1 = Temp_time 
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)
                    T1 = np.concatenate((T1,Temp_time),axis=0)

        X_1,T2 = RPM_Contraint(DATA,T1,np.array(RPM_RANGE))
        X_2,T3 = Throttle_Contraint(X_1,T2,THROTTLE_CUTOFF)
        X_3,T4 = Engine_load_Contraint(X_2,T3,ENGINE_LOAD_CUTOFF)
        X_4,T5 = idle_Contraint(X_3,T4,ENGINE_IDLE_SWITCH)
        X,T6 = IS_Contraint(X_4,T5,REGN_INHIBT_SWITCH)

        np.save(PATH + 'Train_Data/' + str(vehicle_cnt) + '.npy', X)

    return 


def load_train_data():

    #create_train_data(PATH,VARIABLES)
    #exit(0)

    FLAG = 0 
    LABEL = []

    for cnt in range(1,10): # set

        X = np.load(PATH + 'Train_Data/' + str(cnt) +'.npy')
        sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[4,:], lamb=10)
        A_TS, N_TS= state_detection(X[1,:],temp_trend)
        print(A_TS)
        print(N_TS)
        sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[6,:], lamb=10)
        BURN_Quality,DP_Fall = quality_of_soot_burn(X[1,:],temp_trend)

        for st_cnt in range(0,min(len(A_TS),len(N_TS))):
            if(st_cnt == 0):
                X_T = X[:,0:A_TS[st_cnt]]
                BURN_Quality_T = BURN_Quality[0:A_TS[st_cnt]]
                DP_Fall_T = DP_Fall[0:A_TS[st_cnt]]
            else:
                X_T = X[:,N_TS[st_cnt-1]:A_TS[st_cnt]]
                BURN_Quality_T = BURN_Quality[N_TS[st_cnt-1]:A_TS[st_cnt]]
                DP_Fall_T = DP_Fall[N_TS[st_cnt-1]:A_TS[st_cnt]]

            print(X_T.shape[1])

            for aug_cnt in range(0,int(Time_series_len),2):#AGUMENTATION LOOP
                DUTY_CYCLES = int(X_T.shape[1]/Time_series_len)
                count = 1
                for i in range(0+aug_cnt, X_T.shape[1], int(Time_series_len)):

                    TEMP_DP = X_T[1,i:i+Time_series_len]
                    TEMP_TM = X_T[6,i:i+Time_series_len]
                    TEMP_BURN_Quality = BURN_Quality_T[i:i+Time_series_len]
                    TEMP_DP_Fall = DP_Fall_T[i:i+Time_series_len]

                    ############### STATE INPUTS ##########
                    #H_P = sum(TEMP_DP>DP_HIGH_MAX)
                    #if(H_P >= Num_H_DP_cnt*Scale_fact):
                    #    TEMP_HP = np.ones([1,Time_series_len])
                    #else:
                    #    TEMP_HP = np.zeros([1,Time_series_len])

                    TEMP_HP = 1*(TEMP_DP>DP_HIGH_MAX)

                    #M_P = sum(TEMP_DP>DP_HIGH_MIN)
                    #if(M_P >= Num_M_DP_cnt*Scale_fact):
                    #    TEMP_MP = np.ones([1,Time_series_len])
                    #else:
                    #    TEMP_MP = np.zeros([1,Time_series_len])

                    TEMP_MP = 1*(TEMP_DP>DP_HIGH_MIN)

                    #H_T = sum(TEMP_TM>SCR_TEMP_HIGH_MAX)
                    #if(H_T >= Num_H_T_cnt*Scale_fact):
                    #    TEMP_HT = np.ones([1,Time_series_len])
                    #else:
                    #    TEMP_HT = np.zeros([1,Time_series_len])

                    TEMP_HT = 1*(TEMP_TM>SCR_TEMP_HIGH_MAX)

                    #M_T = sum(TEMP_TM>SCR_TEMP_HIGH_MIN)
                    #if(M_T >= Num_M_T_cnt*Scale_fact):
                    #    TEMP_MT = np.ones([1,Time_series_len])
                    #else:
                    #    TEMP_MT = np.zeros([1,Time_series_len])

                    TEMP_MT = 1*(TEMP_TM>SCR_TEMP_HIGH_MIN)
                    #######################################

                    TEMP = np.array([TEMP_DP,TEMP_TM])

                    if (TEMP.shape[1]==Time_series_len):
                        #print(TEMP.shape)
                        #print(TEMP_HP.shape)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_HP, axis=0)),axis=0)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_MP, axis=0)),axis=0)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_HT, axis=0)),axis=0)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_MT, axis=0)),axis=0)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_BURN_Quality, axis=0)),axis=0)
                        TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_DP_Fall, axis=0)),axis=0)

                    
                    if (TEMP.shape[1]==Time_series_len):
                        if (FLAG == 0):
                            DATA = TEMP
                            DATA = np.expand_dims(DATA, axis=0)
                            FLAG = 1
                            LABEL.append(DUTY_CYCLES - count)
                            count = count + 1
                        else:
                            DATA = np.concatenate((DATA,np.expand_dims(TEMP, axis=0)),axis=0)
                            LABEL.append(DUTY_CYCLES - count)
                            count = count + 1

        #print('---------------------------',DATA.shape)

    print(DATA.shape)
    DATA_NORM = min_max_normalize_data(DATA)
    print(np.array(LABEL).shape)
    print(np.array(LABEL))
    print(max(np.array(LABEL)))
    #exit(0)
    return DATA_NORM, np.array(LABEL)


def create_test_data(PATH,VARIABLES):
    
    for vehicle_cnt in range(10,14):

        Data_PATH = PATH + "V" + str(vehicle_cnt) + "/"
        pkt_list = os.listdir(Data_PATH)

        for data_packet_cnt in range(0,len(pkt_list)):
            OBD_data_path = Data_PATH + pkt_list[data_packet_cnt] 
            OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
            T_L = []
            V_L = []
            for var_type in VARIABLES:
                X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
                T_L.append(np.array((X_Time), dtype=np.int64))
                V_L.append(np.array((X_Value), dtype=float))

            if (len(V_L[1]) > 0):
                Temp_time = np.array(T_L[1])
                Resample_data = resample_PID_data(V_L,Number_of_features)
                Resample_data[8,:] = retain_monotonicity(Resample_data[8,:])
                Resample_data[9,:] = retain_monotonicity(Resample_data[9,:])

                if (data_packet_cnt == 0):
                    DATA = Resample_data
                    T1 = Temp_time 
                else:
                    DATA = np.concatenate((DATA,Resample_data),axis=1)
                    T1 = np.concatenate((T1,Temp_time),axis=0)

        X_1,T2 = RPM_Contraint(DATA,T1,np.array(RPM_RANGE))
        X_2,T3 = Throttle_Contraint(X_1,T2,THROTTLE_CUTOFF)
        X_3,T4 = Engine_load_Contraint(X_2,T3,ENGINE_LOAD_CUTOFF)
        X_4,T5 = idle_Contraint(X_3,T4,ENGINE_IDLE_SWITCH)
        X,T6 = IS_Contraint(X_4,T5,REGN_INHIBT_SWITCH)

        np.save(PATH + 'Test_Data/' + str(vehicle_cnt) + '.npy', X)

    return 


def load_test_data():

    #create_test_data(PATH,VARIABLES)
    #exit(0)

    FLAG = 0 
    LABEL = []

    for cnt in range(10,14): # set range(10,14):

        X = np.load(PATH + 'Test_Data/' + str(cnt) +'.npy') # set Test_Data
        sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[4,:], lamb=10)
        A_TS, N_TS= state_detection(X[1,:],temp_trend)
        print(A_TS)
        print(N_TS)

        sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[6,:], lamb=10)
        BURN_Quality,DP_Fall = quality_of_soot_burn(X[1,:],temp_trend)

        for st_cnt in range(0,min(len(A_TS),len(N_TS))): # +1 for test time to cover last state
            if(st_cnt == 0):
                X_T = X[:,0:A_TS[st_cnt]]
                BURN_Quality_T = BURN_Quality[0:A_TS[st_cnt]]
                DP_Fall_T = DP_Fall[0:A_TS[st_cnt]]
            else:
                X_T = X[:,N_TS[st_cnt-1]:A_TS[st_cnt]]
                BURN_Quality_T = BURN_Quality[N_TS[st_cnt-1]:A_TS[st_cnt]]
                DP_Fall_T = DP_Fall[N_TS[st_cnt-1]:A_TS[st_cnt]]

            print(X_T.shape[1])

            count = 1
            offset = 0

            for i in range(0+offset, X_T.shape[1], int(Time_series_len/1)):

                TEMP_DP = X_T[1,i:i+Time_series_len]
                TEMP_TM = X_T[6,i:i+Time_series_len]
                TEMP_BURN_Quality = BURN_Quality_T[i:i+Time_series_len]
                TEMP_DP_Fall = DP_Fall_T[i:i+Time_series_len]
               ############### STATE INPUTS ##########
                #H_P = sum(TEMP_DP>DP_HIGH_MAX)
                #if(H_P >= Num_H_DP_cnt):
                #    TEMP_HP = np.ones([1,Time_series_len])
                #else:
                #    TEMP_HP = np.zeros([1,Time_series_len])

                TEMP_HP = 1*(TEMP_DP>DP_HIGH_MAX)

                #M_P = sum(TEMP_DP>DP_HIGH_MIN)
                #if(M_P >= Num_M_DP_cnt):
                #    TEMP_MP = np.ones([1,Time_series_len])
                #else:
                #    TEMP_MP = np.zeros([1,Time_series_len])

                TEMP_MP = 1*(TEMP_DP>DP_HIGH_MIN)

                #H_T = sum(TEMP_TM>SCR_TEMP_HIGH_MAX)
                #if(H_T >= Num_H_T_cnt):
                #    TEMP_HT = np.ones([1,Time_series_len])
                #else:
                #    TEMP_HT = np.zeros([1,Time_series_len])

                TEMP_HT = 1*(TEMP_TM>SCR_TEMP_HIGH_MAX)

                #M_T = sum(TEMP_TM>SCR_TEMP_HIGH_MIN)
                #if(M_T >= Num_M_T_cnt):
                #    TEMP_MT = np.ones([1,Time_series_len])
                #else:
                #    TEMP_MT = np.zeros([1,Time_series_len])

                TEMP_MT = 1*(TEMP_TM>SCR_TEMP_HIGH_MIN)
                #######################################
                TEMP = np.array([TEMP_DP,TEMP_TM])

                if (TEMP.shape[1]==Time_series_len):
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_HP, axis=0)),axis=0)
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_MP, axis=0)),axis=0)
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_HT, axis=0)),axis=0)
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_MT, axis=0)),axis=0)
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_BURN_Quality, axis=0)),axis=0)
                    TEMP = np.concatenate((TEMP,np.expand_dims(TEMP_DP_Fall, axis=0)),axis=0)


                DUTY_CYCLES = int(X_T.shape[1]/Time_series_len)
                #print(TEMP.shape)
                #exit(0)

                if (TEMP.shape[1]==Time_series_len):
                    if (FLAG == 0):
                        DATA = TEMP
                        DATA = np.expand_dims(DATA, axis=0)
                        FLAG = 1
                        LABEL.append(DUTY_CYCLES - count)
                        count = count + 1
                    else:
                        DATA = np.concatenate((DATA,np.expand_dims(TEMP, axis=0)),axis=0)
                        LABEL.append(DUTY_CYCLES - count)
                        count = count + 1

        #print('---------------------------',DATA.shape)

    print(DATA.shape)
    DATA_NORM = min_max_normalize_data(DATA)
    print(np.array(LABEL).shape)
    #exit(0)
    return DATA_NORM, np.array(LABEL)


