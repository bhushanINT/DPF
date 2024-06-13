import numpy as np
import json
import yaml
import statsmodels.api as sm
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches
import os
from utils_new import extract_PID_data, resample_PID_data
from state_logic import quality_of_soot_burn_v1


# Read configuration file
Config_path = "D:/Work/Timeseries_models/Code/DPF_classification/State_logic1/config.yml"
with open(Config_path, 'r') as file:
    Config = yaml.safe_load(file)


# Extract configuration informatioon
VEHICLE = Config['vehicle_num']
PATH = "D:/Work/Timeseries_models/DATA/FULL_DATA/" + VEHICLE + "/"
VARIABLES = Config['VARIABLES']
RPM_RANGE = Config['RPM_RANGE']
THROTTLE_CUTOFF = Config['THROTTLE_CUTOFF']
ENGINE_LOAD_CUTOFF = Config['ENGINE_LOAD_CUTOFF']
ENGINE_IDLE_SWITCH = Config['ENGINE_IDLE_SWITCH']

print(Config['Thresholds']['DPFINT_TEMP_HIGH_MIN'])

# Data Read
dir_list = os.listdir(PATH)
FLAG = 0
for data_packet_cnt in range(0,len(dir_list)):
    OBD_data_path = PATH + dir_list[data_packet_cnt] 
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
    T_L = []
    V_L = []
    for var_type in VARIABLES:
        X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
        T_L.append(np.array((X_Time), dtype=np.int64))
        V_L.append(np.array((X_Value), dtype=float))

    if (len(V_L[0]) > 0):
        Temp_time = np.array(T_L[0]) # its DP time so no need to resample
        Resample_data = resample_PID_data(V_L,len(VARIABLES)) # cross check by plot

        if ((len(V_L[0])>0) and (FLAG==0)):
            DATA = Resample_data
            T1 = Temp_time 
            FLAG = 1
        else:
            if(len(V_L[0])>0):
                DATA = np.concatenate((DATA,Resample_data),axis=1)
                T1 = np.concatenate((T1,Temp_time),axis=0)

print(DATA.shape)
print(len(T1))
fig, axs1 = plt.subplots(2)
fig.suptitle('DP, TEMP')
axs1[0].plot(DATA[0,:],'r')
axs1[1].plot(DATA[1,:],'r')
#plt.show()

AR_conditions_met = np.zeros(len(T1))
AR_initiated = np.zeros(len(T1))
DP_block = np.zeros(len(T1))


for time_cnt in range(0,len(T1)-1):
    if((DATA[3,time_cnt] > RPM_RANGE[0]) and (DATA[3,time_cnt] < RPM_RANGE[1]) and (DATA[4,time_cnt] > THROTTLE_CUTOFF) and (DATA[5,time_cnt] > ENGINE_LOAD_CUTOFF) and (DATA[6,time_cnt] > ENGINE_IDLE_SWITCH)):
        AR_conditions_met[time_cnt] = 1
    if(DATA[1,time_cnt] > Config['Thresholds']['DPFINT_TEMP_HIGH_MAX']):
        AR_initiated[time_cnt] = 1
    if(DATA[0,time_cnt] > Config['Thresholds']['DP_HIGH_MAX']):
        DP_block[time_cnt] = 1


fig, axs2 = plt.subplots(3)
fig.suptitle('AR_conditions_met, AR_initiated, DP_block')
axs2[0].plot(T1,AR_conditions_met)
axs2[1].plot(T1,AR_initiated)
axs2[2].plot(T1,DP_block)
#plt.show()

 
Hour_cnt = 0

AR_conditions_met_frac = np.zeros(len(range(T1[0],T1[-1],3600000)))
AR_initiated_crop_frac = np.zeros(len(range(T1[0],T1[-1],3600000)))
DP_block_crop_frac = np.zeros(len(range(T1[0],T1[-1],3600000)))

for time_cnt in range(T1[0],T1[-1],3600000):

    Start_TS = time_cnt
    End_TS = Start_TS + 3600000

    IDX = np.where(np.logical_and(T1>=Start_TS, T1<=End_TS))  

    AR_conditions_met_crop = AR_conditions_met[IDX]
    AR_initiated_crop = AR_initiated[IDX]
    DP_block_crop = DP_block[IDX]

    print(AR_conditions_met_crop.shape)

    if (len(IDX)>0):
        AR_conditions_met_frac[Hour_cnt] = sum(AR_conditions_met_crop)/len(IDX)
        AR_initiated_crop_frac[Hour_cnt] = sum(AR_initiated_crop)/len(IDX)
        DP_block_crop_frac[Hour_cnt] = sum(DP_block_crop)/len(IDX)
        Hour_cnt = Hour_cnt + 1

Regen_attempt_ratio = AR_initiated_crop_frac/AR_conditions_met_frac
Regen_attempt_ratio[np.isnan(Regen_attempt_ratio)] = 0

fig, axs3 = plt.subplots(4)
fig.suptitle('AR_conditions_met_frac, AR_initiated_crop_frac, DP_block_crop_frac')
axs3[0].plot(AR_conditions_met_frac,'r')
axs3[1].plot(AR_initiated_crop_frac,'r')
axs3[2].plot(DP_block_crop_frac,'r')
axs3[3].plot(Regen_attempt_ratio,'r')
plt.show()


Regen_attempt_ratio = AR_initiated_crop_frac/AR_conditions_met_frac

exit(0)











