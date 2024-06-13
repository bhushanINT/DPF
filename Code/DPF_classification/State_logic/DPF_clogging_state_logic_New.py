import numpy as np
import json
import yaml
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches
import os
from utils_new import extract_PID_data, resample_PID_data, RPM_Contraint, Throttle_Contraint, Engine_load_Contraint, idle_Contraint, IS_Contraint
from utils_new import speed_Contraint
from state_logic import quality_of_soot_burn_v2

# Read configuration file
Config_path = "D:/Work/Timeseries_models/Code/DPF_classification/State_logic/config_new.yml"
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
REGN_INHIBT_SWITCH = Config['REGN_INHIBT_SWITCH']


# Data Read
#PATH = "D:/Work/Timeseries_models/DATA/P244B/Data_Mahindra/849980124446064640/"
dir_list = os.listdir(PATH)
#print(dir_list[30:60])

FLAG = 0

#for data_packet_cnt in range(0,len(dir_list)):
for data_packet_cnt in range(0,1):
    #OBD_data_path = PATH + dir_list[data_packet_cnt]
    OBD_data_path = "D:/Work/Timeseries_models/DATA/FULL_DATA/MangoDB_data/979722067546996736"
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
    T_L = []
    V_L = []
    for var_type in VARIABLES:
        X_Time, X_Value = extract_PID_data(OBD_data,'SAE',var_type, 0)
        T_L.append(np.array((X_Time), dtype=np.int64))
        V_L.append(np.array((X_Value), dtype=float))
    if (len(V_L[0]) > 0):

        Temp_time = np.array(T_L[1]) # its DP time so no need to resample
        Resample_data = resample_PID_data(V_L,len(VARIABLES)) # cross check by plot

        if ((len(V_L[0])>0) and (FLAG==0)):
            DATA = Resample_data
            T1 = Temp_time 
            FLAG = 1
        else:
            if(len(V_L[0])>0):
                DATA = np.concatenate((DATA,Resample_data),axis=1)
                T1 = np.concatenate((T1,Temp_time),axis=0)


Active_Regen = np.zeros(len(DATA[6,:]))

for cnt in range (5,len(DATA[6,:])):
    T = DATA[6,cnt-4:cnt]
    #print(T.shape)
    Temp_val = np.max(T)
    if (Temp_val > 550):
        Active_Regen[cnt] = 1

fig, ax1 = plt.subplots()
ax1.plot(T1,DATA[6,:],'g.')
ax1.plot(T1,Active_Regen*650,'r')

X_1,T2 = RPM_Contraint(DATA,T1,np.array(RPM_RANGE))
X_2,T3 = Throttle_Contraint(X_1,T2,THROTTLE_CUTOFF)
X_3,T4 = Engine_load_Contraint(X_2,T3,ENGINE_LOAD_CUTOFF)
X_4,T5 = idle_Contraint(X_3,T4,ENGINE_IDLE_SWITCH)
X,T6 = IS_Contraint(X_4,T5,REGN_INHIBT_SWITCH)

fig, ax2 = plt.subplots()
ax2.plot(T6,X[6,:],'g.')

fig, ax3 = plt.subplots()
ax3.plot(T6,X[1,:],'g.')



BURN_Quality,ALERT = quality_of_soot_burn_v2(T6,X[1,:],T1,Active_Regen,'979722067546996736')



fig, ax4 = plt.subplots(2)
ax4[0].plot(T6,BURN_Quality,'g.')
ax4[1].plot(T6,ALERT,'r.')

plt.show()
exit(0)





