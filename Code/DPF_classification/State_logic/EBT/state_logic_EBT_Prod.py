import numpy as np


def soot_burn_quantification(DP,AR_status,Thresholds):

	Sample_window  = Thresholds['Sample_window']
	DP_L1 = Thresholds['DP_Level_1']
	DP_L2 = Thresholds['DP_Level_2']
	DP_L3 = Thresholds['DP_Level_3']
	DP_L4 = Thresholds['DP_Level_4']
	

	ALERT = -1
	BURN_Quality = 0
	ALERT_SAMPLE = int(len(DP)/2)

	DP_BUF_PRE = DP[0:25]
	DP_BUF_POST = DP[25:50]
	print(DP.shape)
	print(DP_BUF_PRE.shape)
	print(DP_BUF_POST.shape)


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

	if(DP_Pre==0):
		DP_Fall_perct = 0
	else:
		DP_Fall_perct = ((DP_Pre - DP_Post)/DP_Pre) 

	# SET Alert ir-respective of ative regen
	if(DP_Post > DP_L4):
		ALERT = 2
	if(DP_L4 > DP_Post > DP_L3):
		ALERT = 1

	if(AR_status==1): # Active regenration
		BURN_Quality = DP_Fall_perct
		if ((BURN_Quality<=0) and (DP_Pre > DP_L1)): #Just to make sure not every attempt is classified as failed
			BURN_Quality = 2
		elif ((BURN_Quality<=-0.5)): # even if DP_pre is less but DP is rsing significantly
			BURN_Quality = 2
		elif (BURN_Quality>0):
			BURN_Quality= 1 - np.exp(-5*(BURN_Quality))

	if(BURN_Quality < 0):
		BURN_Quality = 0

	if((BURN_Quality==2) and (DP_L2 < DP_Post < DP_L3)):
		ALERT = 1
	elif ((BURN_Quality==2) and (DP_Post > DP_L3)):
		ALERT = 2


	if(DP_Post < DP_L1):
		ALERT = 0		 


	return BURN_Quality,ALERT,ALERT_SAMPLE




def soot_burn_quantification_R2(DP_pre_buf,DP_post_buf,AR_Complete,Thresholds):

	Sample_window  = Thresholds['Sample_window']
	DP_L1 = Thresholds['DP_Level_1'] #Multipty by 1.15 to handle higher sampling Frequency
	DP_L2 = Thresholds['DP_Level_2'] #Multipty by 1.15 to handle higher sampling Frequency 
	DP_L3 = Thresholds['DP_Level_3'] #Multipty by 1.15 to handle higher sampling Frequency
	DP_L4 = Thresholds['DP_Level_4'] #Multipty by 1.15 to handle higher sampling Frequency
	

	ALERT = -1
	BURN_Quality = 0

	DP_BUF_PRE = DP_pre_buf
	DP_BUF_PRE_5 = np.percentile(DP_BUF_PRE, 5)
	DP_BUF_PRE_95 = np.percentile(DP_BUF_PRE, 95)

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


	print('.....................',DP_Pre) # comment


	# SET Alert ir-respective of ative regen
	if(DP_Pre > DP_L4):
		ALERT = 2
	if(DP_L4 > DP_Pre > DP_L3):
		ALERT = 1
	if(DP_Pre < DP_L1):
		ALERT = 0

	if(AR_Complete==1):
		DP_BUF_POST = DP_post_buf
		DP_BUF_POST_5 = np.percentile(DP_BUF_POST, 5)
		DP_BUF_POST_95 = np.percentile(DP_BUF_POST, 95)

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

		if(DP_Pre==0):
			DP_Fall_perct = 0
		else:
			DP_Fall_perct = ((DP_Pre - DP_Post)/DP_Pre) 

		BURN_Quality = DP_Fall_perct
		if ((BURN_Quality<=0) and (DP_Pre > DP_L2)): #DP_L2 #Just to make sure not every attempt is classified as failed 
			BURN_Quality = 2
		elif ((BURN_Quality<=-0.5) and (DP_Pre > DP_L1)): # even if DP_pre is less but DP is rising significantly
			BURN_Quality = 2
		elif (BURN_Quality>0):
			BURN_Quality= 1 - np.exp(-5*(BURN_Quality))

		if(BURN_Quality < 0):
			BURN_Quality = 0.1 # Make it 0.1 to avoid miss event

		if((BURN_Quality==2) and (DP_L2 < DP_Post < DP_L3)):
			ALERT = 1
		elif ((BURN_Quality==2) and (DP_Post > DP_L3)):
			ALERT = 2
		elif (DP_Post < DP_L1):
			ALERT = 0


	return BURN_Quality,ALERT
