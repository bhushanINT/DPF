import numpy as np
import json
import yaml
from matplotlib import pyplot as plt 
import os
from utils_EBT import extract_PID_data, resample_PID_data, RPM_Contraint, Throttle_Contraint, Engine_load_Contraint, idle_Contraint, IS_Contraint
from state_logic_EBT import quality_of_soot_burn_v2
import requests

# Read configuration file
Config_path = "D:/Work/Timeseries_models/Code/DPF_classification/State_logic/EBT/config_EBT.yml"
with open(Config_path, 'r') as file:
    Config = yaml.safe_load(file)


# Extract configuration informatioon
VARIABLES = Config['VARIABLES']
RPM_RANGE = Config['RPM_RANGE']
THROTTLE_CUTOFF = Config['THROTTLE_CUTOFF']
ENGINE_LOAD_CUTOFF = Config['ENGINE_LOAD_CUTOFF']
ENGINE_IDLE_SWITCH = Config['ENGINE_IDLE_SWITCH']
REGN_INHIBT_SWITCH = Config['REGN_INHIBT_SWITCH']

Regen_Temp = Config['Thresholds']['DPFINT_TEMP_HIGH_MAX']

'''
TT = np.zeros(1000)
cnt = 0
for inc in np.arange(0, 1, 0.001):
    TT[cnt] = 1- np.exp(-5*inc)
    cnt = cnt+1
plt.plot(TT)
plt.show()
exit(0)
'''

# Data Read
#OBD_data_path = 'D:/Work/Timeseries_models/DATA/TH_DATA/EBT/T/1281412584410447872'
#OBD_data_path = 'D:/Work/Timeseries_models/DATA/TH_DATA/TATA/T/1241025102523400192'
#OBD_data_path = 'D:/Work/Timeseries_models/DATA/TH_DATA/AL/T/1223989316162682880'
#OBD_data_path = 'D:/Work/Timeseries_models/DATA/TH_DATA/Mahindra/T/1191352654433878016'
#OBD_data_path = 'D:/Work/Timeseries_models/DATA/TH_DATA/Eicher/2110/1073560749432897536'
OBD_data_path = 'D:/Work/FRP/DATA/FOTA/EBT/Volvo - Cummins ISX15/1281385415655292928'

FLAG = 0

for data_packet_cnt in range(0,1):
    OBD_data = [json.loads(line) for line in open(OBD_data_path, 'r')]
    T_L = []
    V_L = []
    # EXtract Required PIDs from Jason
    for var_type in VARIABLES:
        X_Time, X_Value = extract_PID_data(OBD_data,'SAE_AVG_SPN',var_type, 0)
        T_L.append(np.array((X_Time), dtype=np.int64))
        V_L.append(np.array((X_Value), dtype=float))
    if (len(V_L[0]) > 0):

        Temp_time = np.array(T_L[1]) # its DP time so no need to resample
        Resample_data = resample_PID_data(V_L,len(VARIABLES)) # Resample other PIDs w.r.t DP

        if ((len(V_L[0])>0) and (FLAG==0)):
            DATA = Resample_data
            T1 = Temp_time 
            FLAG = 1
        else:
            if(len(V_L[0])>0):
                DATA = np.concatenate((DATA,Resample_data),axis=1)
                T1 = np.concatenate((T1,Temp_time),axis=0)

print(T1[0])
print(T1[-1])
#plt.plot(DATA[0,:],'b')
#plt.show()
#exit(0)

# Active Regenration detection
Active_Regen = np.zeros(len(DATA[5,:]))

for cnt in range (5,len(DATA[5,:])):
    T = DATA[5,cnt-4:cnt]
    #print(T.shape)
    Temp_val = np.max(T)
    if (Temp_val > Regen_Temp):
        Active_Regen[cnt] = 1

# Plot Active Regenration status
fig, ax1 = plt.subplots()
ax1.plot(T1,DATA[5,:],'g.')
ax1.plot(T1,Active_Regen*550,'r')
ax1.set_ylabel('DPF input temprature')
ax1.set_xlabel('Time')
ax1.legend(['DPF input temprature', 'Active regenration status'])

#DATA = DATA[:,1::10]
#T1  = T1 [1::10]

X_1,T2 = RPM_Contraint(DATA,T1,np.array(RPM_RANGE))
X_2,T3 = Throttle_Contraint(X_1,T2,THROTTLE_CUTOFF)
X_3,T4 = Engine_load_Contraint(X_2,T3,ENGINE_LOAD_CUTOFF)
X,T6 = idle_Contraint(X_3,T4,ENGINE_IDLE_SWITCH)
#X,T6 = IS_Contraint(X_4,T5,REGN_INHIBT_SWITCH)

#''' # HIGH FREQUENCY COMPENSASTION

X = X[:,1::10]
T6  = T6 [1::10]
print(X.shape)
print(T6.shape)
print(int(len(T6)/2))
#exit(0)
#''' # HIGH FREQUENCY COMPENSASTION

#print(T1[1:11]-T1[0:10])
#print(T6[1:11]-T6[0:10])
#exit(0)


fig, ax2 = plt.subplots()
ax2.plot(T6,X[5,:],'g.')

fig, ax3 = plt.subplots()
ax3.plot(T6,X[1,:],'g.')
ax3.plot(T6,(X[4,:]>50)+1,'r.')
ax3.plot(T6,X[5,:]>425,'k.')

#plt.show()
#exit(0)
print(np.mean(X[1,:]))


BURN_Quality,ALERT,DP_FALL = quality_of_soot_burn_v2(T6,X[1,:],T1,Active_Regen,'1225922848434946048',Config['Thresholds'])

print(BURN_Quality)

fig, ax4 = plt.subplots(4)
ax4[0].plot(T6,BURN_Quality,'g.')
#ax4[0].plot(T6,DP_FALL,'r')
ax4[0].set_xlabel('BURN_Quality')
ax4[1].plot(T6,ALERT,'r.')
ax4[1].set_xlabel('ALERT LEVEL')
ax4[2].plot(T6,X[8,:],'r')
ax4[2].set_xlabel('Soot Load')
ax4[3].plot(T6,DP_FALL,'g')
ax4[3].axhline(y=0, xmin=0, xmax=len(T6), color='r',linestyle='-')
ax4[3].set_xlabel('DP_FALL')

plt.show()
exit(0)






