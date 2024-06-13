import numpy as np
import json
import yaml
import statsmodels.api as sm
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches
import os
from utils_new import extract_PID_data, resample_PID_data, RPM_Contraint, Throttle_Contraint, Engine_load_Contraint, idle_Contraint, IS_Contraint, find_nearest
from utils_new import retain_monotonicity,zoh_vec,segments_fit, speed_Contraint
from state_logic import state_detection, quality_of_soot_burn_v1, get_milage, get_spectrogram
from scipy.signal import hilbert

# Read configuration file
Config_path = "D:/Work/Timeseries_models/Code/DPF_classification/State_logic/config.yml"
with open(Config_path, 'r') as file:
    Config = yaml.safe_load(file)


# Extract configuration informatioon
VEHICLE = Config['vehicle_num']
PATH = "D:/Work/Timeseries_models/DATA/FULL_DATA/" + VEHICLE + "/"
VARIABLES = Config['VARIABLES']
RPM_RANGE = Config['RPM_RANGE']
AVG_SPEED_RANGE = Config['AVG_SPEED_RANGE']
THROTTLE_CUTOFF = Config['THROTTLE_CUTOFF']
ENGINE_LOAD_CUTOFF = Config['ENGINE_LOAD_CUTOFF']
ENGINE_IDLE_SWITCH = Config['ENGINE_IDLE_SWITCH']
REGN_INHIBT_SWITCH = Config['REGN_INHIBT_SWITCH']
active_TS = np.array(Config['vehicle'][VEHICLE]['active_TS'])
removed_TS = np.array(Config['vehicle'][VEHICLE]['removed_TS'])


# Data Read
dir_list = os.listdir(PATH)
print(dir_list[30:60])
#exit(0)
for data_packet_cnt in range(0,len(dir_list)):
#for data_packet_cnt in range(30,60):
    OBD_data_path = PATH + dir_list[data_packet_cnt] 
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
        Temp_time = np.array(T_L[1]) # its DP time so no need to resample
        #plt.plot(np.array(V_L[8]))
        #plt.show()
        Resample_data = resample_PID_data(V_L,len(VARIABLES)) # cross check by plot
        Resample_data[8,:] = retain_monotonicity(Resample_data[8,:])
        Resample_data[9,:] = retain_monotonicity(Resample_data[9,:])
        #plt.plot(Resample_data[8,:])
        #plt.show()
        #exit(0)

        if (data_packet_cnt == 0):
            DATA = Resample_data
            T1 = Temp_time 
        else:
            DATA = np.concatenate((DATA,Resample_data),axis=1)
            T1 = np.concatenate((T1,Temp_time),axis=0)

#plt.plot(DATA[5,:],'r')
#print(np.mean(DATA[5,:]))
#plt.plot(np.convolve(DATA[5,:], np.ones(30)/30,'same'))
#plt.show()
#exit(0)

X_1,T2 = RPM_Contraint(DATA,T1,np.array(RPM_RANGE))
X_2,T3 = Throttle_Contraint(X_1,T2,THROTTLE_CUTOFF)
X_3,T4 = Engine_load_Contraint(X_2,T3,ENGINE_LOAD_CUTOFF)
X_4,T5 = idle_Contraint(X_3,T4,ENGINE_IDLE_SWITCH)
#X_4,T5 = speed_Contraint(X_3,T4,AVG_SPEED_RANGE)
X,T6 = IS_Contraint(X_4,T5,REGN_INHIBT_SWITCH)

print(X.shape)
print(T6.shape)
#XX,TT1 = sm.tsa.filters.hpfilter(X[4,:], lamb=100)
#XX,TT2 = sm.tsa.filters.hpfilter(X[6,:], lamb=100)
#plt.plot(TT1,'b')
#plt.plot(X[10,:],'g')
#plt.plot(X[6,:],'r')
#plt.show()
#exit(0)


sw_cycle,dp_trend = sm.tsa.filters.hpfilter(X[1,:], lamb=100)
Slope_of_dp = np.gradient(dp_trend)
sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[4,:], lamb=10)

A_TS, N_TS,DP_level,TEMP_level = state_detection(X[1,:],temp_trend,Slope_of_dp,Config['Thresholds'])
print(A_TS)
print(N_TS)

print('TIME RESOLUTION',T6[1:21]-T6[0:20])
#print(T6[20436])  
#plt.plot(DP_level,'b')
#plt.plot(TEMP_level,'r')

#### PLOTING BOSCH
fig, ax1 = plt.subplots()
ax1.plot(X[1,:],'b')
ax1.set_ylabel('Differential pressure (kPa)')
ax1.yaxis.label.set_color('blue') 
ax1.set_xlabel('Time Samples')

#ax1.plot(dp_trend,'k')
ax1.axhline(y=Config['Thresholds']['DP_HIGH_MAX'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_HIGH_MIN'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_LOW_MAX'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_LOW_MIN'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')

A_indx = np.zeros(active_TS.shape)
N_indx = np.zeros(removed_TS.shape)
for cnt in range (0,active_TS.shape[0]):
    VAL,IDX = find_nearest(T6, active_TS[cnt])
    A_indx[cnt] = IDX
    VAL,IDX = find_nearest(T6, removed_TS[cnt])
    N_indx[cnt] = IDX
    ax1.axvline(x = A_indx[cnt] , ymin=0, ymax=3, color='r')
    ax1.axvline(x = N_indx[cnt] , ymin=0, ymax=3, color='g')

ax2 = ax1.twinx() 
ax2.plot(X[4,:],'m')
ax2.set_ylabel('DPF in Temperature (°C)')
ax2.yaxis.label.set_color('magenta') 
#ax2.plot(temp_trend,'k')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_HIGH_MAX'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_HIGH_MIN'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_LOW_MAX'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_LOW_MIN'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
plt.show()



#########################SPECTROGRAM#############
#RET = get_spectrogram(X[1,:],A_indx,N_indx)
#exit(0)
#################################################

#### PLOTING OUR
'''
fig, ax1 = plt.subplots()
ax1.plot(X[1,:],'b')
ax1.set_ylabel('Differential pressure (kPa)')
ax1.yaxis.label.set_color('blue') 
ax1.set_xlabel('Time Samples')

#ax1.plot(dp_trend,'k')
ax1.axhline(y=Config['Thresholds']['DP_HIGH_MAX'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_HIGH_MIN'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_LOW_MAX'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')
ax1.axhline(y=Config['Thresholds']['DP_LOW_MIN'], xmin=0, xmax=len(X[1,:]), color='r',linestyle='--')

for cnt in range (0,len(N_TS)):
    ax1.axvline(x = A_TS[cnt] , ymin=0, ymax=3, color='r')
    ax1.axvline(x = N_TS[cnt] , ymin=0, ymax=3, color='g')

ax2 = ax1.twinx() 
ax2.plot(X[4,:],'m')
ax2.set_ylabel('DPF in Temperature (°C)')
ax2.yaxis.label.set_color('magenta') 
#ax2.plot(temp_trend,'k')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_HIGH_MAX'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_HIGH_MIN'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_LOW_MAX'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
ax2.axhline(y=Config['Thresholds']['SCR_TEMP_LOW_MIN'], xmin=0, xmax=len(X[4,:]), color='y',linestyle='--')
plt.show()
'''


sw_cycle,temp_trend = sm.tsa.filters.hpfilter(X[6,:], lamb=10)
#plt.plot(X[6,:],'b')
#plt.plot(temp_trend,'r')
#plt.show()
#exit(0)
#BURN_Quality,DP_Fall = quality_of_soot_burn(X[1,:],temp_trend,Config['Thresholds'])
BURN_Quality,DP_Fall = quality_of_soot_burn_v1(X[1,:],X[10,:],temp_trend,Config['Thresholds'])
Milage_trend, Milage_trend_ZOH= get_milage(X[8,:],X[9,:],100) # CHECK for speed filter make 25


fig, ax3 = plt.subplots()
ax3.plot(X[1,:],'r')
ax3.set_ylabel('Differential pressure (kPa)', fontsize = 18)
ax3.yaxis.label.set_color('red')  
ax3.set_xlabel('Time Samples', fontsize = 18)

col =[]
for i in range(0, len(BURN_Quality)):
    if (BURN_Quality[i]<=0):
        col.append('white')  
    elif ((BURN_Quality[i]>0) and (BURN_Quality[i]<0.5)):
        col.append('yellow')
    elif ((BURN_Quality[i]>=0.5) and (BURN_Quality[i]<0.8)):
        col.append('orange')
    elif ((BURN_Quality[i]>=0.8) and (BURN_Quality[i] <= 1) ):
        col.append('green')
    elif ((BURN_Quality[i] == 2)):
        col.append('black')
    elif ((BURN_Quality[i] == 3)):
        col.append('red')

for i in range(len(BURN_Quality)):
    ax3.plot(i,0.1,c = col[i],marker='>')

white_patch = mpatches.Patch(color='white', label='Good State (No Soot Burn needed)')
yellow_patch = mpatches.Patch(color='yellow', label='Low Soot Burn')
orange_patch = mpatches.Patch(color='orange', label='Medium Soot Burn')
green_patch = mpatches.Patch(color='green', label='Good Soot Burn')
black_patch = mpatches.Patch(color='black', label='Incomplete Soot Burn')
red_patch = mpatches.Patch(color='red', label='Soot Burn Failed')
ax3.legend(handles=[white_patch, yellow_patch, orange_patch, green_patch, black_patch, red_patch])

ax4 = ax3.twinx() 
ax4.plot(temp_trend,'y')
ax4.set_ylabel('Temprature (°C)', fontsize = 18)
ax4.yaxis.label.set_color('yellow') 

#for cnt in range (0,active_TS.shape[0]):
#    ax3.axvline(x = A_indx[cnt] , ymin=0, ymax=3, color='r')
#    ax3.axvline(x = N_indx[cnt] , ymin=0, ymax=3, color='g')

plt.show()


#### EVELOPE PLOT
'''
SIG = X[1,:]
X1 = np.array(range(0,len(SIG)))
z = np.polyfit(X1, SIG, 1)
#print(z)
FIT = z[0]*X1 + z[1]
plt.plot(SIG, label='signal')
plt.plot(FIT, label='trend')
plt.ylabel('Differential pressure (kPa)')
plt.xlabel('Time Samples')
plt.legend(["Differential pressure (kPa)", "Trend"])

plt.show()
'''

########## MILAGE TREND IN BURN CASE ######
Good_milage = -5*np.zeros(len(BURN_Quality))
Bad_milage = -5*np.zeros(len(BURN_Quality))

for i in range(0, len(BURN_Quality)):
    if(BURN_Quality[i] == 0):
        Good_milage[i] = Milage_trend_ZOH[i]
    elif((BURN_Quality[i] > 1)): 
        Bad_milage[i] = Milage_trend_ZOH[i]

Good_milage_ZOH = zoh_vec(Good_milage)
Bad_milage_ZOH = zoh_vec(Bad_milage)

Good_milage_mean = -5*np.zeros(len(BURN_Quality))
Bad_milage_mean = -5*np.zeros(len(BURN_Quality))

Mil_win = 2500
for frame_cnt in range(0, len(BURN_Quality)-Mil_win,Mil_win):
    Good_milage_mean[frame_cnt+int(Mil_win/2)] = np.mean(Good_milage_ZOH[frame_cnt:frame_cnt+Mil_win])
    Bad_milage_mean[frame_cnt+int(Mil_win/2)] = np.mean(Bad_milage_ZOH[frame_cnt:frame_cnt+Mil_win])


#############################################
ALERT = np.zeros(len(BURN_Quality))
BURN_TH = 50 # CHECK for speed filter make 25
for i in range(BURN_TH, len(BURN_Quality)):
    Percentage_milage_drop = ((Good_milage_ZOH[i] - Bad_milage_ZOH[i])/Good_milage_ZOH[i])*100
    if((sum(BURN_Quality[i-50:i])) > 125 and (Percentage_milage_drop > 15)):  # CHECK for speed filter make 75
        ALERT[i] = 2

#############################################


fig, ax5 = plt.subplots()

ax5.plot(temp_trend,'c')
ax5.set_ylabel('Temprature (°C)')
ax5.yaxis.label.set_color('cyan') 
ax5.axis(ymin=0,ymax=700)

ax6 = ax5.twinx()

ax6.plot(X[1,:],'g')
ax6.set_xlabel('Time Samples')
ax6.set_ylabel('Different pressure (kPa)')
ax6.yaxis.label.set_color('green')

#ax6.plot(Milage_trend,'m*')
ax6.plot(Good_milage_mean,"bP")
ax6.plot(Bad_milage_mean,"rX")

#ax6.plot(Good_milage_ZOH,"k.")
#ax6.plot(Bad_milage_ZOH,"r.")

ax6.plot(ALERT,"r")
ax6.axis(ymin=0.1,ymax=10)

for i in range(len(BURN_Quality)):
    ax6.plot(i,0.2,c = col[i],marker='>')

ax6.legend(handles=[white_patch, yellow_patch, orange_patch, green_patch, black_patch, red_patch])

plt.show()


### DP ..progression in good state burn