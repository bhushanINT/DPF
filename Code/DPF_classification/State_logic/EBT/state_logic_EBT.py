import numpy as np
import requests
from matplotlib import pyplot as plt 



#https://dieselnet.com/tech/dpf_regen.php

def quality_of_soot_burn_v2(DP_TS,DP,TS,Active_Regen,veichle_ID,Thresholds):

	Sample_window  = Thresholds['Sample_window']
	DP_L1 = Thresholds['DP_Level_1']
	DP_L2 = Thresholds['DP_Level_2']
	DP_L3 = Thresholds['DP_Level_3']
	DP_L4 = Thresholds['DP_Level_4']
	

	State_change = Active_Regen[1:-1] - Active_Regen[0:-2]

	AR_Start_IDX =  np.nonzero(State_change == 1)
	AR_Start_TS = TS[AR_Start_IDX]
	AR_End_IDX =  np.nonzero(State_change == -1)
	AR_End_TS = TS[AR_End_IDX]

	print('---------------------------------------------------------')
	print(AR_Start_TS)
	print(AR_End_TS)

	DP_ACT = np.zeros(len(DP))

	for cnt in range(0,len(DP)):
		TEMP = DP_TS[cnt]
		#print(TEMP)
		for idx in range(0,len(AR_Start_TS)):
			#print(AR_Start_TS[idx])
			#print(AR_End_TS[idx])
			if ((TEMP>=AR_Start_TS[idx]) and (TEMP<=AR_End_TS[idx])):
				DP_ACT[cnt] = 1
 

	BURN_Quality = np.zeros(len(DP))
	DP_FALL = np.zeros(len(DP))

	ALERT = -1*np.ones(len(DP))

	for cnt in range(2*Sample_window,len(DP)):

		DP_BUF_PRE = DP[(cnt-2*Sample_window):cnt-Sample_window]
		DP_BUF_POST = DP[cnt-Sample_window:cnt]

		#'''
		DP_BUF_PRE_5 = np.percentile(DP_BUF_PRE, 5)
		DP_BUF_PRE_95 = np.percentile(DP_BUF_PRE, 95)
		DP_BUF_POST_5 = np.percentile(DP_BUF_POST, 5)
		DP_BUF_POST_95 = np.percentile(DP_BUF_POST, 95)

		CNT_NUM = 0
		PRE_SUM = 0
		for cnt1 in range(0,len(DP_BUF_PRE)):
			if((DP_BUF_PRE[cnt1]>DP_BUF_PRE_5) and (DP_BUF_PRE[cnt1]<DP_BUF_PRE_95)):
				PRE_SUM = PRE_SUM + DP_BUF_PRE[cnt1]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Pre = np.mean(DP_BUF_PRE)
		else:
			DP_Pre = PRE_SUM/CNT_NUM

		CNT_NUM = 0
		POST_SUM = 0
		for cnt2 in range(0,len(DP_BUF_POST)):
			if((DP_BUF_POST[cnt2]>DP_BUF_POST_5) and (DP_BUF_POST[cnt2]<DP_BUF_POST_95)):
				POST_SUM = POST_SUM + DP_BUF_POST[cnt2]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Post = np.mean(DP_BUF_POST)
		else:
			DP_Post = POST_SUM/CNT_NUM
		#'''

		if(DP_Pre==0):
			DP_Fall_perct = 0
		else:
			DP_Fall_perct = ((DP_Pre - DP_Post)/DP_Pre) 

		DP_FALL[cnt-Sample_window] = DP_Fall_perct 


		# SET Alert ir-respective of ative regen
		if(DP_Post > DP_L4):
			ALERT[cnt-Sample_window] = 2
		if(DP_L4 > DP_Post > DP_L3):
			ALERT[cnt-Sample_window] = 1

		if(DP_ACT[cnt-Sample_window]==1): # For passive Regen Just (Temp > 450) and (Flowrate>300)
			print('REGEN ON.......................................')
			BURN_Quality[cnt-Sample_window] = DP_Fall_perct
			#if (BURN_Quality[cnt]<=0): 
			if ((BURN_Quality[cnt-Sample_window]<=0) and (DP_Pre > DP_L1)): #Just to make sure not every attempt is classified as failed
				BURN_Quality[cnt-Sample_window] = 2
			elif ((BURN_Quality[cnt-Sample_window]<=-0.5)): # even if DP_pre is less but DP is rising significantly
				BURN_Quality[cnt-Sample_window] = 2
			elif (BURN_Quality[cnt-Sample_window]>0):
				BURN_Quality[cnt-Sample_window] = 1 - np.exp(-5*(BURN_Quality[cnt-Sample_window]))

		BURN_Quality[BURN_Quality < 0] = 0 # if Pre DP was less # Make it 0.1 to avoid miss event

		if((BURN_Quality[cnt-Sample_window]==2) and (DP_L2 < DP_Post < DP_L3)):
			ALERT[cnt-Sample_window] = 1
		elif ((BURN_Quality[cnt-Sample_window]==2) and (DP_Post > DP_L3)):
			ALERT[cnt-Sample_window] = 2


		if(DP_Post < DP_L1):
			ALERT[cnt-Sample_window] = 0		 


	return BURN_Quality,ALERT,DP_FALL
	

def quality_of_soot_burn_v2_back(DP_TS,DP,TS,Active_Regen,veichle_ID,Thresholds):

	Sample_window  = Thresholds['Sample_window']
	DP_L1 = Thresholds['DP_Level_1']
	DP_L2 = Thresholds['DP_Level_2']
	DP_L3 = Thresholds['DP_Level_3']
	DP_L4 = Thresholds['DP_Level_4']
	

	State_change = Active_Regen[1:-1] - Active_Regen[0:-2]

	AR_Start_IDX =  np.nonzero(State_change == 1)
	AR_Start_TS = TS[AR_Start_IDX]
	AR_End_IDX =  np.nonzero(State_change == -1)
	AR_End_TS = TS[AR_End_IDX]

	print('---------------------------------------------------------')
	print(AR_Start_TS)
	print(AR_End_TS)

	'''
	TS_Diff = 0
	for cnt in range(0,len(AR_End_TS)-1):
		Temp_Diff = AR_Start_TS[cnt+1] - AR_End_TS[cnt]
		if (Temp_Diff > TS_Diff):
			TS_Diff = Temp_Diff
			BASE_START_TS = AR_End_TS[cnt]
			BASE_END_TS = AR_Start_TS[cnt+1]

	URL = 'http://algo-internal-apis.intangles-aws-us-east-1.intangles.us:1234/vehicle/' + veichle_ID +  "/summary/" +str(int(BASE_START_TS)) + "/" + str(int(BASE_END_TS)) 
	r = requests.get(URL,stream=True)
	data = r.json()
	ML = data['result']['mileage']
	print('BASE Milage-------------------------',ML)

	# ACTIVE REGEN MERGE LOGIC
	FLAG = 0
	for cnt in range(0,len(AR_End_TS)-1):
		Temp_var = AR_Start_TS[cnt+1] - AR_End_TS[cnt]
		if((Temp_var<2000000) and (FLAG == 0)): # 33 min
			IDX = [cnt]
			FLAG = 1
		elif((Temp_var<2000000) and (FLAG == 1)):
			IDX = np.append(IDX,cnt)

	AR_Start_TS_Temp = np.delete(AR_Start_TS, IDX+1)
	AR_End_TS_Temp = np.delete(AR_End_TS, IDX)

	print(np.array(IDX))
	# ACTIVE REGEN MERGE LOGIC

	#print(AR_Start_TS)
	#print(AR_End_TS)

	#exit(0)


	for cnt in range(0,len(AR_Start_TS_Temp)):
		URL = 'http://algo-internal-apis.intangles-aws-us-east-1.intangles.us:1234/vehicle/' + veichle_ID +  "/summary/" +str(int(AR_Start_TS_Temp[cnt])) + "/" + str(int(AR_End_TS_Temp[cnt])) 
		r = requests.get(URL,stream=True)
		data = r.json()
		ML = data['result']['mileage']
		print('Milage-------------------------',ML)
	'''


	DP_ACT = np.zeros(len(DP))

	for cnt in range(0,len(DP)):
		TEMP = DP_TS[cnt]
		#print(TEMP)
		for idx in range(0,len(AR_Start_TS)):
			#print(AR_Start_TS[idx])
			#print(AR_End_TS[idx])
			if ((TEMP>=AR_Start_TS[idx]) and (TEMP<=AR_End_TS[idx])):
				DP_ACT[cnt] = 1
 
	#plt.plot(DP_TS,DP,'g.')
	#plt.plot(DP_TS,DP_ACT,'r')
	#plt.show()


	BURN_Quality = np.zeros(len(DP))
	DP_FALL = np.zeros(len(DP))

	ALERT = -1*np.ones(len(DP))

	for cnt in range(2*Sample_window,len(DP)):

		DP_BUF_PRE = DP[(cnt-2*Sample_window):cnt-Sample_window]
		DP_BUF_POST = DP[cnt-Sample_window:cnt]

		#'''
		DP_BUF_PRE_5 = np.percentile(DP_BUF_PRE, 5)
		DP_BUF_PRE_95 = np.percentile(DP_BUF_PRE, 95)
		DP_BUF_POST_5 = np.percentile(DP_BUF_POST, 5)
		DP_BUF_POST_95 = np.percentile(DP_BUF_POST, 95)

		CNT_NUM = 0
		PRE_SUM = 0
		for cnt1 in range(0,len(DP_BUF_PRE)):
			if((DP_BUF_PRE[cnt1]>DP_BUF_PRE_5) and (DP_BUF_PRE[cnt1]<DP_BUF_PRE_95)):
				PRE_SUM = PRE_SUM + DP_BUF_PRE[cnt1]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Pre = np.mean(DP_BUF_PRE)
		else:
			DP_Pre = PRE_SUM/CNT_NUM

		CNT_NUM = 0
		POST_SUM = 0
		for cnt2 in range(0,len(DP_BUF_POST)):
			if((DP_BUF_POST[cnt2]>DP_BUF_POST_5) and (DP_BUF_POST[cnt2]<DP_BUF_POST_95)):
				POST_SUM = POST_SUM + DP_BUF_POST[cnt2]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Post = np.mean(DP_BUF_POST)
		else:
			DP_Post = POST_SUM/CNT_NUM
		#'''

		if(DP_Pre==0):
			DP_Fall_perct = 0
		else:
			DP_Fall_perct = ((DP_Pre - DP_Post)/DP_Pre) 

		DP_FALL[cnt-Sample_window] = DP_Fall_perct 


		# SET Alert ir-respective of ative regen
		if(DP_Post > DP_L4):
			ALERT[cnt-Sample_window] = 2
		if(DP_L4 > DP_Post > DP_L3):
			ALERT[cnt-Sample_window] = 1

		if(DP_ACT[cnt-Sample_window]==1): # For passive Regen Just (Temp > 450) and (Flowrate>300)
			print('REGEN ON.......................................')
			BURN_Quality[cnt-Sample_window] = DP_Fall_perct
			#if (BURN_Quality[cnt]<=0): 
			if ((BURN_Quality[cnt-Sample_window]<=0) and (DP_Pre > DP_L1)): #Just to make sure not every attempt is classified as failed
				BURN_Quality[cnt-Sample_window] = 2
			elif ((BURN_Quality[cnt-Sample_window]<=-0.5)): # even if DP_pre is less but DP is rising significantly
				BURN_Quality[cnt-Sample_window] = 2
			elif (BURN_Quality[cnt-Sample_window]>0):
				BURN_Quality[cnt-Sample_window] = 1 - np.exp(-5*(BURN_Quality[cnt-Sample_window]))
			'''
			elif ((BURN_Quality[cnt]>-0.10) and (BURN_Quality[cnt]<0.33/Scale_fact)):
				#BURN_Quality[cnt] = 0.33
				BURN_Quality[cnt] = 1 - np.exp(-5*(BURN_Quality[cnt]+0.10))
			elif ((BURN_Quality[cnt]>=0.33/Scale_fact) and (BURN_Quality[cnt]<0.66/Scale_fact)):
				#BURN_Quality[cnt] = 0.66
				BURN_Quality[cnt] = 1 - np.exp(-5*(BURN_Quality[cnt]))
			elif (BURN_Quality[cnt]>=0.66/Scale_fact):
				#BURN_Quality[cnt] = 0.99
				BURN_Quality[cnt] = 1 - np.exp(-5*(BURN_Quality[cnt]))
			'''

		BURN_Quality[BURN_Quality < 0] = 0 # if Pre DP was less # Make it 0.1 to avoid miss event

		if((BURN_Quality[cnt-Sample_window]==2) and (DP_L2 < DP_Post < DP_L3)):
			ALERT[cnt-Sample_window] = 1
		elif ((BURN_Quality[cnt-Sample_window]==2) and (DP_Post > DP_L3)):
			ALERT[cnt-Sample_window] = 2


		if(DP_Post < DP_L1):
			ALERT[cnt-Sample_window] = 0		 


	return BURN_Quality,ALERT,DP_FALL

def quality_of_soot_burn_v2_r2(DP_TS,DP,TS,Active_Regen,veichle_ID,Thresholds):

	Sample_window  = Thresholds['Sample_window']
	DP_L1 = Thresholds['DP_Level_1']
	DP_L2 = Thresholds['DP_Level_2']
	DP_L3 = Thresholds['DP_Level_3']
	DP_L4 = Thresholds['DP_Level_4']
	

	State_change = Active_Regen[1:-1] - Active_Regen[0:-2]

	AR_Start_IDX =  np.nonzero(State_change == 1)
	AR_Start_TS = TS[AR_Start_IDX]
	AR_End_IDX =  np.nonzero(State_change == -1)
	AR_End_TS = TS[AR_End_IDX]

	print('---------------------------------------------------------')
	print(AR_Start_TS)
	print(AR_End_TS)


	DP_ACT = np.zeros(len(DP))

	for cnt in range(0,len(DP)):
		TEMP = DP_TS[cnt]
		#print(TEMP)
		for idx in range(0,len(AR_Start_TS)):
			#print(AR_Start_TS[idx])
			#print(AR_End_TS[idx])
			if ((TEMP>=AR_Start_TS[idx]) and (TEMP<=AR_End_TS[idx])):
				DP_ACT[cnt] = 1
 

	State_change_DP = DP_ACT[1:-1] - DP_ACT[0:-2]
	AR1_Start_IDX =  np.nonzero(State_change_DP == 1)
	AR1_Start_TS = DP_TS[AR1_Start_IDX]
	AR1_End_IDX =  np.nonzero(State_change_DP == -1)
	AR1_End_TS = DP_TS[AR1_End_IDX]


	print('---------------------------------------------------------')
	print(AR1_Start_IDX)
	print(AR1_End_IDX)

	#fig, ax5 = plt.subplots()
	#ax5.plot(DP_TS[0:-2],State_change_DP,'g.')
	#ax5.plot(DP_TS,DP_ACT,'r')
	#plt.show()
	#exit(0)


	BURN_Quality = np.zeros(len(DP))
	DP_FALL = np.zeros(len(DP))

	ALERT = -1*np.ones(len(DP))

	for cnt in range(2,len(AR1_Start_IDX[0])):

		DP_BUF_PRE = DP[int(AR1_Start_IDX[0][cnt])-25:int(AR1_Start_IDX[0][cnt])]
		DP_BUF_POST = DP[int(AR1_End_IDX[0][cnt]):int(AR1_End_IDX[0][cnt])+25]

		#'''
		DP_BUF_PRE_5 = np.percentile(DP_BUF_PRE, 5)
		DP_BUF_PRE_95 = np.percentile(DP_BUF_PRE, 95)
		DP_BUF_POST_5 = np.percentile(DP_BUF_POST, 5)
		DP_BUF_POST_95 = np.percentile(DP_BUF_POST, 95)

		CNT_NUM = 0
		PRE_SUM = 0
		for cnt1 in range(0,len(DP_BUF_PRE)):
			if((DP_BUF_PRE[cnt1]>DP_BUF_PRE_5) and (DP_BUF_PRE[cnt1]<DP_BUF_PRE_95)):
				PRE_SUM = PRE_SUM + DP_BUF_PRE[cnt1]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Pre = np.mean(DP_BUF_PRE)
		else:
			DP_Pre = PRE_SUM/CNT_NUM

		CNT_NUM = 0
		POST_SUM = 0
		for cnt2 in range(0,len(DP_BUF_POST)):
			if((DP_BUF_POST[cnt2]>DP_BUF_POST_5) and (DP_BUF_POST[cnt2]<DP_BUF_POST_95)):
				POST_SUM = POST_SUM + DP_BUF_POST[cnt2]
				CNT_NUM = CNT_NUM +1

		if(CNT_NUM==0):
			DP_Post = np.mean(DP_BUF_POST)
		else:
			DP_Post = POST_SUM/CNT_NUM
		#'''

		if(DP_Pre==0):
			DP_Fall_perct = 0
		else:
			DP_Fall_perct = ((DP_Pre - DP_Post)/DP_Pre) 

		print(1 - np.exp(-5*DP_Fall_perct))
		DP_FALL[np.nonzero(DP_TS == AR1_Start_TS[cnt])] = 1 - np.exp(-5*DP_Fall_perct)


	return BURN_Quality,ALERT,DP_FALL

