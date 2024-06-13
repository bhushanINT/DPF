import numpy as np
from scipy import signal
import pywt
from matplotlib import pyplot as plt 

def state_detection(DP,temp_trend,Slope_of_dp,Thresholds):

	DP_HIGH_MAX = Thresholds['DP_HIGH_MAX']
	DP_HIGH_MIN = Thresholds['DP_HIGH_MIN']
	DP_LOW_MAX = Thresholds['DP_LOW_MAX']
	DP_LOW_MIN = Thresholds['DP_LOW_MIN']
	SCR_TEMP_HIGH_MAX = Thresholds['SCR_TEMP_HIGH_MAX']
	SCR_TEMP_HIGH_MIN = Thresholds['SCR_TEMP_HIGH_MIN']
	SCR_TEMP_LOW_MAX = Thresholds['SCR_TEMP_LOW_MAX']
	SCR_TEMP_LOW_MIN= Thresholds['SCR_TEMP_LOW_MIN']
	DP_SLOPE_HIGH = Thresholds['DP_SLOPE_HIGH']
	DP_SLOPE_LOW = Thresholds['DP_SLOPE_LOW']

	Sample_window  = Thresholds['Sample_window']

	Num_H_DP_cnt   = Thresholds['Hi_DP_cnt']
	Num_M_DP_cnt   = Thresholds['Med_DP_cnt']
	Num_H_T_cnt   = Thresholds['Hi_T_cnt']
	Num_M_T_cnt   = Thresholds['Med_T_cnt']
	Num_SLP_cnt   = Thresholds['SLP_T_cnt']

	FLAG = 'REMOVE'

	A_TS = []
	N_TS = []

	DP_level = np.zeros(len(DP))
	TEMP_level = np.zeros(len(DP))

	for cnt in range(Sample_window,len(DP)):

  		DP_BUF = DP[cnt-Sample_window:cnt]
  		TMP_BUF = temp_trend[cnt-Sample_window:cnt]
  		SLP_DP_BUF = Slope_of_dp[cnt-int(Sample_window/4):cnt]

  		if ((FLAG == 'REMOVE') and (DP[cnt] > DP_HIGH_MAX)):
  			H_P = sum(DP_BUF>DP_HIGH_MAX)
  			M_P = sum(DP_BUF>DP_HIGH_MIN)
  			H_T = sum(TMP_BUF>SCR_TEMP_HIGH_MAX)
  			M_T = sum(TMP_BUF>SCR_TEMP_HIGH_MIN)
  			SL = sum(SLP_DP_BUF>DP_SLOPE_HIGH)

  			#if ((H_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (H_T >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt) and (SL >= Num_SLP_cnt)):
  			if ((H_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (H_T >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt)):
  				FLAG = 'ACTIVE'
  				A_TS.append(cnt)

  		if ((FLAG == 'ACTIVE') and (DP[cnt] < DP_LOW_MAX)):
  			L_P = sum(DP_BUF<DP_LOW_MAX)
  			M_P = sum(DP_BUF<DP_LOW_MIN)
  			L_T = sum(TMP_BUF<SCR_TEMP_LOW_MAX)
  			M_T = sum(TMP_BUF<SCR_TEMP_LOW_MIN)
  			SL = sum(SLP_DP_BUF<DP_SLOPE_LOW)

  			#if ((L_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (L_P >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt) and (SL >= Num_SLP_cnt)):
  			if ((L_P >= Num_H_DP_cnt) and (M_P >= Num_M_DP_cnt) and (L_P >= Num_H_T_cnt) and (M_T >= Num_M_T_cnt)):
  				FLAG = 'REMOVE'
  				N_TS.append(cnt)

  		if (DP[cnt] > DP_HIGH_MAX):
  			DP_level[cnt] = 1
  		elif((DP[cnt] < DP_HIGH_MAX) and (DP[cnt] > DP_HIGH_MIN)):
  			DP_level[cnt] = 0.5

  		if (temp_trend[cnt] > SCR_TEMP_HIGH_MAX):
  			TEMP_level[cnt] = 1
  		elif((temp_trend[cnt] < SCR_TEMP_HIGH_MAX) and (temp_trend[cnt] > SCR_TEMP_HIGH_MIN)):
  			TEMP_level[cnt] = 0.5

	return np.array(A_TS), np.array(N_TS),DP_level,TEMP_level

#https://dieselnet.com/tech/dpf_regen.php

def quality_of_soot_burn(DP,temp_trend,Thresholds):

	Sample_window  = 2*Thresholds['Sample_window'] # 50 sample approx 1 hour of burn

	SCR_TEMP_HIGH_MAX = Thresholds['SCR_TEMP_HIGH_MAX']
	SCR_TEMP_HIGH_MIN = Thresholds['SCR_TEMP_HIGH_MIN']
	DP_HIGH_MAX = Thresholds['DP_HIGH_MAX']
	DP_HIGH_MIN = Thresholds['DP_HIGH_MIN']
	DP_LOW_MAX = Thresholds['DP_LOW_MAX']
	DP_LOW_MIN = Thresholds['DP_LOW_MIN']

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
			BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 2

	return BURN_Quality, DP_Fall

#https://dieselnet.com/tech/dpf_regen.php
def quality_of_soot_burn_v1(DP,EFR,temp_trend,Thresholds):

	Sample_window  = 2*Thresholds['Sample_window'] # 50 sample approx 1 hour of burn

	SCR_TEMP_HIGH_MAX = Thresholds['SCR_TEMP_HIGH_MAX']
	SCR_TEMP_HIGH_MIN = Thresholds['SCR_TEMP_HIGH_MIN']
	DP_HIGH_MAX = Thresholds['DP_HIGH_MAX']
	DP_HIGH_MIN = Thresholds['DP_HIGH_MIN']
	DP_LOW_MAX = Thresholds['DP_LOW_MAX']
	DP_LOW_MIN = Thresholds['DP_LOW_MIN']
	EXT_FLOW_RATE_TH = Thresholds['EXT_FLOW_RATE_TH']

	BURN_Quality = np.zeros(len(DP))
	DP_Fall = np.zeros(len(DP))


	for cnt in range(2*Sample_window,len(DP)):
		TMP_BUF = temp_trend[(cnt-2*Sample_window):cnt-Sample_window]
		H_T = sum(TMP_BUF>SCR_TEMP_HIGH_MAX)
		M_T = sum(TMP_BUF>SCR_TEMP_HIGH_MIN)

		EFR_BUF = EFR[(cnt-2*Sample_window):cnt-Sample_window]
		H_FR = sum(EFR_BUF>EXT_FLOW_RATE_TH)


		DP_BUF_PRE = DP[(cnt-2*Sample_window):cnt-Sample_window]
		DP_BUF_POST = DP[cnt-Sample_window:cnt]

		Pre_HH_P = sum(DP_BUF_PRE>DP_HIGH_MAX)
		Pre_HM_P = sum(DP_BUF_PRE>DP_HIGH_MIN)

		Post_LH_P = sum(DP_BUF_POST<DP_LOW_MAX)
		Post_LM_P = sum(DP_BUF_POST<DP_LOW_MIN)

		Post_HM_P = sum(DP_BUF_POST>DP_HIGH_MIN) ### TEMP

		DP_Pre_max = np.max(DP_BUF_PRE)
		DP_Post_min = np.min(DP_BUF_POST)

		DP_Fall_perct = ((DP_Pre_max - DP_Post_min)/DP_Pre_max)

		if ((H_T >= 1) and (M_T >= 5) and (Pre_HH_P >= 1) and (Pre_HM_P >= 5) and (H_FR >= 5)): # DPF IS BLOCK AND BURN CONDITIONS MET and BURN HAPPENED
			BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = DP_Fall_perct
		elif ((Pre_HM_P > 5) and (M_T <= 1) and (H_FR <=1)): # DPF IS BLOCK BUT BURN CONDITIONS ARE NOT MET (TRAFFIC)
			BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 2
		elif ((H_T >= 1) and (M_T >= 5) and (Pre_HM_P >= 5) and (Post_LM_P <= 1) and (H_FR >= 5)): # DPF IS BLOCK AND BURN CONDITIONS MET BUT BURN NOT HAPPENED
		#elif ((M_T >= 5) and (Pre_HM_P >= 3) and (Post_HM_P >= 3) and (H_FR >= 5)):
			BURN_Quality[(cnt-2*Sample_window):cnt-Sample_window] = 3

	return BURN_Quality, DP_Fall


def get_milage(Fuel_use,dist_travel,window):

	#window = 100

	OUT = -5*np.ones(len(Fuel_use))
	OUT1 = np.zeros(len(Fuel_use))

	for cnt in range(window,len(Fuel_use),window):
		Milage = (dist_travel[cnt]-dist_travel[cnt-window])/(Fuel_use[cnt]-Fuel_use[cnt-window])
		OUT[cnt-50] = Milage		
		OUT1[cnt-window:cnt] = Milage

	return OUT, OUT1

def get_spectrogram(DP,A_indx,N_indx):

	for cnt in range(1,len(N_indx)-1):
		print(N_indx[cnt])
		print(N_indx[cnt+1])
		sig = DP[int(N_indx[cnt]):int(N_indx[cnt+1])]
		#f, t, Sxx = signal.spectrogram(sig, 1/35, scaling='spectrum')
		#plt.pcolormesh(t, f, np.log10(Sxx))
		#plt.xlabel('Time')
		#plt.ylabel('Frequency')
		w = pywt.Wavelet('db2')
		coef, freqs=pywt.cwt(sig,np.arange(1,129),wavelet=w)
		plt.matshow(abs(coef))
		plt.colorbar(label="Coeff Val")
		plt.show()

	return 1