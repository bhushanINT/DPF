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
    headers = {'Intangles-User-Token': 'YtEMoiCiJSFbRs7eToZw0-sh4tfAZeEfDBLVeoyALuwbKFqNIEgCXoeeA7Jt9Hiy'}
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
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for link_cnt in range(0,len(OBD_LINKS)):
            Link = OBD_LINKS[link_cnt]
            print(Link)
            response = requests.get(Link, stream=True)
            time.sleep(0)
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
local_filename = "D:/Work/FRP/test.json"
Temp_tar_file = 'D:/Work/FRP/test.tar.gz'
veichle_ID = '941321887898664960'

URL = "https://apis.intangles.com/vehicle/" + veichle_ID +"/obd_data/"+ str(int(1652491611000)) + "/" + str(int(1653542290000)) +"?fetch_result_from_multiple_sources=true"
print(URL)
DownloadFile(URL,local_filename)
out_path = 'D:/Work/FRP/DATA/941321887898664960/'
ExtractJason(local_filename,Temp_tar_file,out_path)

exit(0)
                













