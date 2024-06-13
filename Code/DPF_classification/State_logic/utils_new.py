import numpy as np
from matplotlib import pyplot as plt 
import os
from scipy import signal
from scipy import optimize

def extract_PID_data(data, PROTOCOL,LABEL,PLOT_FLAG):
    
    if (PROTOCOL == 'SAE'):

        if LABEL == 'IMAP':
            PID_TAG = '106'
        elif LABEL == 'ENGINE LOAD':
            PID_TAG = '92'
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
        elif LABEL == 'DPFDP': # DPF Diffrential Pressure (across DPF)
            PID_TAG = '3251'
        elif LABEL == 'SCRT':   # SCR Catalyst Temperature Before Catalyst (DPF out)
            PID_TAG = '4360'
        elif LABEL == 'SPEED': # Wheep based Vehicle Speed
            PID_TAG = '84'
        elif LABEL == 'DPFINT':# DPF in Temperature Before DPF (DOC out)
            PID_TAG = '4766'
        elif LABEL == 'IS': #  regen inhibited
            PID_TAG = '3703'        
        elif LABEL == 'FUEL USE': #  Total fuel used (high precision)
            PID_TAG = '5054'        
        elif LABEL == 'DISTANCE': #  Total distance travelled (high precision)
            PID_TAG = '245'

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
    #print(len(data))
    for data_cnt in range(0,len(data[0])):
        if "pids" in data[0][data_cnt]:
            if len(data[0][data_cnt]['pids'])>0:
                for sub_pid_cnt in range(0,len(data[0][data_cnt]['pids'])):  #this loop
                    State = data[0][data_cnt]['pids'][sub_pid_cnt]
                    #print(State)
                    #print(data_cnt)
                    #for state_cnt in range(0,len(State)):
                    if PID_TAG in State:
                        #print("--------------------------------------------IN----------------------------------")
                        Time_vec.append(np.array(State[PID_TAG]['timestamp'], dtype=np.int64))
                        Val_vec.append(np.array(State[PID_TAG]['value'], dtype=float))

    print("Number of time stamp avilables are--->",len(Val_vec))

    if (PLOT_FLAG == 1):            
        plt.title("Time Series") 
        plt.xlabel("Time Stamp") 
        plt.ylabel(LABEL) 
        plt.scatter(Time_vec,Val_vec) 
        plt.show()
                
    return Time_vec,Val_vec

def resample_PID_data(X_Value,Number_of_features):

    # DPFDP has minimum length DATA[1,:], so resample all other variables  w.r.t. DPFDP
    Samples = len(X_Value[1])
    #print(len(X_Value[0]))
    DATA = np.zeros((Number_of_features,Samples))
    TEMP = np.transpose(np.array(X_Value[1]))
 
    for cnt in range(0,Number_of_features):
        if (len(X_Value[cnt]) > 0):
            DATA[cnt,:] = np.transpose(np.array(signal.resample(X_Value[cnt], Samples)))

    DATA[1,:] = TEMP

    return DATA

def retain_monotonicity(In):

    OUT = np.zeros(len(In))

    Start = In[0]
    OUT[0] = Start

    for cnt in range(1,len(In)):
        value = In[cnt]

        if value <= Start:
            OUT[cnt] = Start
        else:
            Start = value
            OUT[cnt] = Start
  
    return OUT

def RPM_Contraint(DATA,T1,RANGE):

    TEMP = DATA[0,:] # 0th is RPM
    nums = TEMP[(RANGE[0] < TEMP) & (TEMP <= RANGE[1])]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if ((RANGE[0] < TEMP[cnt]) and (TEMP[cnt] <= RANGE[1])):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T1[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT
    

def Throttle_Contraint(DATA,T2,TH):

    TEMP = DATA[2,:] # 7th is Throttle
    nums = TEMP[(TH < TEMP)]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if (TH < TEMP[cnt]):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T2[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT

def Engine_load_Contraint(DATA,T3,TH):

    TEMP = DATA[3,:] # 7th is Throttle
    nums = TEMP[(TH < TEMP)]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if (TH < TEMP[cnt]):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T3[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT

def idle_Contraint(DATA,T4,TH):

    TEMP = DATA[5,:] 
    nums = TEMP[(TH < TEMP)]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if (TH < TEMP[cnt]):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T4[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT

def speed_Contraint(DATA,T4,RANGE):

    TEMP = DATA[5,:]
    TEMP = np.convolve(TEMP, np.ones(30)/30,'same')
    nums = TEMP[(RANGE[0] < TEMP) & (TEMP <= RANGE[1])] 
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if ((RANGE[0] < TEMP[cnt]) and (TEMP[cnt] <= RANGE[1])):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T4[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT

def IS_Contraint(DATA,T5,TH):

    TEMP = DATA[7,:] 
    nums = TEMP[(TH > TEMP)]
    Samples = len(nums)
    #print(Samples)
    DATA_OUT = np.zeros((DATA.shape[0],Samples))
    T_OUT = np.zeros(Samples)

    out_cnt = 0

    for cnt in range(0,DATA.shape[1]):
        if (TH > TEMP[cnt]):
            for var_cnt in range(0,DATA.shape[0]):
                DATA_OUT[var_cnt,out_cnt] = DATA[var_cnt,cnt]
            T_OUT[out_cnt] = T5[cnt]
            out_cnt = out_cnt + 1

    return DATA_OUT,T_OUT

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def zoh_vec(V):

    Idx = np.nonzero(V)
    #print(V[Idx[0][0]])

    V[0] = V[Idx[0][0]]

    for i in range(1, len(V)):
        if (V[i]==0):
            V[i] = V[i-1]
        else:
            V[i]=V[i]

    return V

    
#https://stackoverflow.com/questions/70710906/how-to-make-a-piecewise-linear-fit-in-python-with-some-constant-pieces
def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)