import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import datetime
import math
import requests
import json
import tarfile
import os
import time

def DownloadFile(url,local_filename):
    #local_filename = url.split('/')[-1]
    headers = {'Intangles-User-Token': 'vP1ROc1fGJIEHzdDze5V47o80E71J_gHfZv6z_eyWu_opoNVRhtjMBkuu3H5lX8v'}
    #headers = {'Intangles-User-Token': 'djpCuPPVwctFtKIHSWpUiRTcPMvDDyYV8FX9RSXwJ3cHcFVCoBXa3pD3O1cNdznI'}
    r = requests.get(url, headers=headers)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                print("WRITING...........")
                f.write(chunk)
    return 

def ExtractJason(local_filename,Temp_tar_file,out_path):
    LINK_data = [json.loads(line) for line in open(local_filename, 'r')]

    if "s3_obddata_results" in LINK_data[0]['results']['data']:   #Enable to unbrake data download
        OBD_LINKS = LINK_data[0]['results']['data']['s3_obddata_results']
        os.mkdir(out_path)
        for link_cnt in range(0,len(OBD_LINKS)):
            Link = OBD_LINKS[link_cnt]
            print(Link)
            response = requests.get(Link, stream=True)
            time.sleep(2)
            print("response.....................",response)
            if response.status_code == 200:
                with open(Temp_tar_file, 'wb') as f:
                    f.write(response.raw.read())

            if(os.path.exists(Temp_tar_file)):
                # open file
                file = tarfile.open(Temp_tar_file)
                # extracting file
                file.extractall(out_path)
                file.close()
                os.remove(Temp_tar_file)
                time.sleep(0)

        os.remove(local_filename)
        time.sleep(0)

    return
 


# Path of CSV file
LOG_PATH = 'D:/Work/Timeseries_models/DATA/DPF_data/Log_data.csv'
local_filename = "D:/Work/Timeseries_models/DATA/DPF_data/test.json"
Temp_tar_file = 'D:/Work/Timeseries_models/DATA/DPF_data/test.tar.gz'

df = pd.read_csv(LOG_PATH)

veichle = pd.Series(df['vehicle_id'].unique()).sort_values()
print(len(veichle))
#exit(0)


Pre_fault_buff = 0.2
Fault_buff = 0.5

Data_cnt = 2
for veichle_cnt in range(9,10):#len(veichle)):
    print('----------------------------------------------------------',veichle_cnt)
    veichle_select = df[(df['vehicle_id'] == veichle[veichle_cnt])]
    veichle_select = veichle_select.reset_index()
    Transition_col = veichle_select["transition"]
    Engine_run_col = veichle_select["engine_run_hrs_delta"]
    Time_stamp_col = veichle_select["timestamp"]
    #print(veichle_select.shape[0])
    for state_cnt in range(0,veichle_select.shape[0]):
        if ((Transition_col[state_cnt] == 'active-removed') and ((state_cnt+1) < veichle_select.shape[0])):
            #if(math.isnan(Engine_run_col[state_cnt+1]) or  (Engine_run_col[state_cnt] < 1) or (Engine_run_col[state_cnt+1] < 1)):
            if((Engine_run_col[state_cnt] < 1) or (Engine_run_col[state_cnt+1] < 1)):
                print("--------Engine Run time Less Than 1 Hour---------")
            else:

                print(Time_stamp_col[state_cnt])
                SS = Time_stamp_col[state_cnt]
                S_date = datetime.datetime(int(SS[0:4]), int(SS[5:7]), int(SS[8:10]), int(SS[11:13]), int(SS[14:16]),int(SS[17:19]))
                Time_stamp_mili = datetime.datetime.timestamp(S_date)*1000
                print(Time_stamp_mili)

                SS_pre = Time_stamp_col[state_cnt-1]
                S_date_pre = datetime.datetime(int(SS_pre[0:4]), int(SS_pre[5:7]), int(SS_pre[8:10]), int(SS_pre[11:13]), int(SS_pre[14:16]),int(SS_pre[17:19]))
                Pre_Time_stamp_mili = datetime.datetime.timestamp(S_date_pre)*1000

                SS_post = Time_stamp_col[state_cnt+1]
                S_date_post = datetime.datetime(int(SS_post[0:4]), int(SS_post[5:7]), int(SS_post[8:10]), int(SS_post[11:13]), int(SS_post[14:16]),int(SS_post[17:19]))
                Post_Time_stamp_mili = datetime.datetime.timestamp(S_date_post)*1000

                Fault_time = Time_stamp_mili - Pre_Time_stamp_mili
                Healthy_time = Post_Time_stamp_mili - Time_stamp_mili
                #print(Fault_time)
                #print(Healthy_time)
                #exit(0)

                
                Fault_start_time  = ((Fault_time) + (Pre_fault_buff * Fault_time))
                
                A_state_start = (Time_stamp_mili - Fault_start_time) 
                A_state_end = (Time_stamp_mili - (Fault_buff * Fault_time))
                N_state_start = Time_stamp_mili
                N_state_end = Time_stamp_mili + (Fault_buff * Healthy_time)

                print(A_state_start)
                print(A_state_end)
                # FAULT STATE
                URL = "https://apis.intangles.com/vehicle/" + str(int(veichle[veichle_cnt])) +"/obd_data/"+ str(int(A_state_start)) + "/" + str(int(A_state_end)) +"?fetch_result_from_multiple_sources=true"
                print(URL)
                print('----------------------------------------------------------',veichle_cnt)
                DownloadFile(URL,local_filename)
                out_path = 'D:/Work/Timeseries_models/DATA/DPF_data/A_State/' + "State" + str(Data_cnt)
                ExtractJason(local_filename,Temp_tar_file,out_path)
                # NORMAL STATE
                URL = "https://apis.intangles.com/vehicle/" + str(int(veichle[veichle_cnt])) +"/obd_data/"+ str(int(N_state_start)) + "/" + str(int(N_state_end)) +"?fetch_result_from_multiple_sources=true"
                print(URL)
                print('----------------------------------------------------------',veichle_cnt)
                DownloadFile(URL,local_filename)
                out_path = 'D:/Work/Timeseries_models/DATA/DPF_data/N_State/' + "State" + str(Data_cnt)
                ExtractJason(local_filename,Temp_tar_file,out_path)
                Data_cnt = Data_cnt + 1

                #exit(0)
                













