from tkinter import *
from tkinter import ttk 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import tensorflow as tf # tf version 1.13 ;   h5py version 2.9.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ctypes 
#ctypes.windll.shcore.SetProcessDpiAwareness(1)

CURR_DIR = os.getcwd()
from sys import platform
if platform == "linux" or platform == "linux2":
    defaultfile = CURR_DIR+'/out.csv'
elif platform == "win32":
    defaultfile = CURR_DIR+r'\out.csv'

def click_generate():

	np_res = res_var.get() # Number of residential profiles
	np_ind = ind_var.get() # Number of industrial profiles
	res_nsamples = numsamples_var.get() # Number of samples per sampling period
	res_nperiods = numperiods_var.get() # Number of sampling periods
	res_unit = resunits_var.get() # Units of sampling period ('Seconds','Minutes','Hours','Days','Weeks')
	agg_type = agg_var.get() # Aggregation type (mean, min, max)
	len_number = tlen_var.get() # Length of output time 
	len_unit = units_var.get() # Units of output time ('Seconds','Minutes','Hours','Days','Weeks','Year')
	season = seas_var.get() # Season ('Winter','Spring','Summer','Fall')
	out_path = txt_out.get() # Output path and file name

	# Define time resolution 
	res_unit_int = getperiod(res_unit)
	timeperiod = res_unit_int*res_nperiods/res_nsamples # period in seconds of sampling resolution
	# Define total output length
	len_unit_int = getlenunit(len_unit)
	timelength = len_unit_int*len_number # length of profile to generate, in seconds

	# Define load types for all profiles to generate 
	ltypes = np.zeros(np_res+np_ind) # indicate type of load for each profile to be generated
	ltypes[:np_res]=1 # the first np_res entries are one (indicating res load), while the last np_ind entries are zeros (indicating ind load)

	# Define season
	season_int = getseason(season)


	#text_out = 'time length:' + str(timelength)
	#lbl_clicked = Label(window, text=text_out).pack()


	click_size()
	

	for i in range(len(ltypes)):	
		print('### Load number '+str(i+1) + '  ###')
		currentport.insert(0, 'Creating load '+str(i+1)+'/'+str(len(ltypes)))
		window.update_idletasks()
		ltype = int(ltypes[i])
		if timelength == 31536000:
			#print('GENERATING ONE COMPLETE YEAR')
			# Generate year-long
			print('generating year-long profile')
			out_y = gen_year(ltype,1)
			genperiod = 604800
			if timeperiod <= 604800:
				print('generating week-long profiles')
				n_weeks = 53
				x_pred_week = gen_week_completeyear(ltype)# Generate week-long profiles according to seasons
				out_yw = combine_yearweek(n_weeks,x_pred_week,out_y)# combine with year+week profile
				genperiod = 3600
			else:
				out_yw = np.copy(out_y)
				out_yw = out_yw.flatten()
			if timeperiod < 3600:
				print('generating hour-long profiles')
				n_hours = len(out_yw)-1
				x_pred = gen_hour(n_hours) # Generate hour-long profiles
				out_ywh = combine_weekhour(n_hours,out_yw,x_pred)# Combine year+week+hour
				genperiod = 30
			else:
				out_ywh = out_yw[:-3]
			if timeperiod < 30:
				print('generating 30sec-long profiles')
				n_30sec = len(out_ywh)# number of 30sec profiles = len(out_ywh)
				x_pred_30sec = gen_30sec(n_30sec) # Generate 30sec-long profiles
				out_ywhp = combine_hour30sec(n_30sec,x_pred_30sec,out_ywh) # Combine year+week+hour+pmu
				genperiod = 1/30
			else:
				out_ywhp = np.copy(out_ywh)

		if timelength < 31536000 and timelength > 30: # If less than 1 year, we neglect yearly pattern and only generate one season
			n_weeks = math.ceil(timelength/604800)+1
			# Generate week based on season and type
			print('Generating week-long profiles..')
			x_pred_week = gen_week(ltype,n_weeks,season_int)
			print('.. combining week-long profiles')
			out_yw = combine_weeks(n_weeks,x_pred_week) #apply filter between weeks and flatten			
			genperiod = 3600
			if timeperiod < 3600:
				n_hours = len(out_yw)-4
				# Generate hour-long profiles
				# Combine year+week+hour
				print('Generating hour-long profiles..')
				x_pred = gen_hour(n_hours) # Generate hour-long profiles
				print('.. combining hour-long profiles')
				out_ywh = combine_weekhour_new(n_hours,out_yw,x_pred)# Combine year+week+hour
				#out_ywh_old = combine_weekhour(n_hours,out_yw[2:],x_pred)# ## for testing only (linear weekhour combination)
				genperiod = 30
			else:
				print(out_yw.shape)				
				out_ywh = out_yw[:-1]
			if timeperiod < 30:
				print('Generating 30sec-long profiles..')
				n_30sec = len(out_ywh)# number of 30sec profiles = len(out_ywh)
				x_pred_30sec = gen_30sec(n_30sec) # Generate 30sec-long profiles
				print('.. combining 30sec-long profiles')
				out_ywhp = combine_hour30sec(n_30sec,x_pred_30sec,out_ywh) # Combine year+week+hour+pmu
				genperiod = 1/30
			else:
				out_ywhp = np.copy(out_ywh)

		if timelength <= 30:
				print('Generating 30sec-long profile')
				n_30sec = 1# number of 30sec profiles = len(out_ywh)
				out_ywhp = gen_30sec(n_30sec) # Generate 30sec-long profiles
				out_ywhp = out_ywhp.flatten()
				genperiod = 1/30

		
				
		#print('out_ywhp size')
		#print(out_ywhp.shape)

		# Trim to the right time length (remove the excess at the end)
		if timelength > 30 and timelength < 172800:
			rr = random.randint(0, int((timelength)/genperiod))
			out = out_ywhp[rr:rr+int(timelength/genperiod)]
			#out_old = out_ywh_old[rr:rr+int(timelength/genperiod)] ## for testing only (linear weekhour combination)
		#elif timelength==31536000:
		#	print('length of generated: '+str(len(out_ywhp)))
		#	out = out_ywhp[:8760]
		else:
			out = out_ywhp[:int(timelength/genperiod)]

			
		#print('output size')
		#print(out.shape)
		#plt.figure()
		#plt.plot(out)
		#plt.savefig('syn_year.png')
		#plt.close()
		
		#print(out)
		
		print('Aggregating..')
		out = aggregate(timeperiod,genperiod,out,agg_type)

		print('output size')
		print(out.shape)
		print('Saving figure..')
		fig, axs = plt.subplots(1,1)
		fig.set_size_inches(14, 7.2)
		axs.plot(out)
		#axs.plot(out_old)
		#fig = plt.figure(figsize=(10,15))
		#fig.plot(out)
		#fig.plot(out_old)
		if platform == "linux" or platform == "linux2":
			fig.savefig('plots/out_plot_'+str(i)+'.png')
		elif platform == "win32":
			fig.savefig('plots\out_plot_'+str(i)+'.png')
		#fig.close()

		#print(out)	

		#out = out.reshape((len(out),1))
		print('Storing new profile..')
		if i == 0:
			out_total = np.empty((len(out),len(ltypes)))
					
		out_total[:,i] = out[:,0]

	
	print('size of total in MB=  '+str(out_total.nbytes*3.125/1000000))	

	np.savetxt(out_path, out_total, delimiter=',')	
	currentport.insert(0,'Ready')


def click_size():
	global txt_size
	np_res = res_var.get() # Number of residential profiles
	np_ind = ind_var.get() # Number of industrial profiles
	res_nsamples = numsamples_var.get() # Number of samples per sampling period
	res_nperiods = numperiods_var.get() # Number of sampling periods
	res_unit = resunits_var.get() # Units of sampling period ('Seconds','Minutes','Hours','Days','Weeks')
	agg_type = agg_var.get() # Aggregation type (mean, min, max)
	len_number = tlen_var.get() # Length of output time 
	len_unit = units_var.get() # Units of output time ('Seconds','Minutes','Hours','Days','Weeks','Year')
	season = seas_var.get() # Season ('Winter','Spring','Summer','Fall')
	out_path = txt_out.get() # Output path and file name

	# Define time resolution 
	res_unit_int = getperiod(res_unit)
	timeperiod = res_unit_int*res_nperiods/res_nsamples # period in seconds of sampling resolution
	# Define total output length
	len_unit_int = getlenunit(len_unit)
	timelength = len_unit_int*len_number # length of profile to generate, in seconds

	n_samples_total = (timelength/timeperiod)*(np_res+np_ind)
	n_bytes_total = n_samples_total*8
	size_total = n_bytes_total*3.125
	txt_size_out = format_bytes(size_total)
	print(txt_size_out)
	txt_size.delete(0,END)
	txt_size.insert(0,txt_size_out)
	

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return "{:.2f}".format(size)+' '+str(power_labels[n])+'B'


def aggregate(timeperiod,genperiod,out,agg_type):

	genlength = int(len(out))*genperiod
	print('length in seconds of generated:')
	print(genlength)
	enddate = pd.to_datetime('00:00:00')+pd.to_timedelta(genlength,unit='s')
	if genperiod < 1:
		freq_gen = "{:.0f}".format(genperiod*1000000000)
		freq_gen = freq_gen+'n'

		freq_out = "{:.0f}".format(timeperiod*1000000000)
		freq_out = freq_out+'n'	

	else:
		freq_gen = str(genperiod)+'s'
		freq_out = str(int(timeperiod))+'s'
	print('frequency')
	print(freq_gen)
	rangeindex = pd.date_range(start = pd.Timestamp('0:00:00'), end= enddate,freq=freq_gen)
	rangeindex = rangeindex.delete(-1)
	
	
	out_pd = pd.DataFrame(out,index=rangeindex)

	if agg_type == 'Mean':
		out_f = out_pd.resample(freq_out).mean()
	elif agg_type == 'Minimum':
		out_f = out_pd.resample(freq_out).min()
	elif agg_type == 'Maximum':
		out_f = out_pd.resample(freq_out).max()


	return out_f.to_numpy()



def combine_weekhour(n_hours,out_yw,x_pred):
	out_ywh = np.zeros((n_hours*120))
	for i in range(n_hours):
		slope = (out_yw[i+1]-out_yw[i])/120
		mean = (out_yw[i+1]+ out_yw[i])/2
		mi = out_yw[i]        
		for j in range(120):
			out_ywh[i*120+j] = mi + j*slope + x_pred[i,j]*mean
	return out_ywh

def combine_weekhour_new(n_hours,out_yw,x_pred):
	out_ywh = np.zeros((n_hours*120))
	x = np.array(range(5))
	print(out_yw.shape)
	for i in range(n_hours):
		h = i+2
		#print(x)
		#print(out_yw[h-2:h+3])
		z = np.polyfit(x,out_yw[h-2:h+3], 4)
		p = np.poly1d(z)
		for j in range(120):
			out_ywh[i*120+j]=p((240+j)/120) + x_pred[i,j]*out_yw[h]
	return out_ywh


def gen_hour(n_hours):
	n_noise = 25
	g_learning_rate = 0.0001
	scaling_factor_hour = 0.4915 # This was the max value by which all of trX was scaled before training GAN. Generated profiles need to be scaled back by this factor
	## Generator
	# load json and create model
	json_file = open('g_model_hour.json', 'r')
	loaded_g_model_json = json_file.read()
	json_file.close()
	loaded_g_model = tf.keras.models.model_from_json(loaded_g_model_json)
	loaded_g_model.load_weights("g_model_hour.h5")
	loaded_g_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=g_learning_rate, decay=g_learning_rate/10))
	## Generate data 
	z_test = sample_Z(n_hours, n_noise)
	z_test_ones = np.zeros((n_hours, 1))
	z_test_ones[:, 0] = 1    
	x_pred = loaded_g_model.predict([z_test, z_test_ones])
	x_pred = x_pred*scaling_factor_hour

	return x_pred

   

def combine_yearweek(n_weeks,x_pred_week_n,n):
	### Combine weeks with year

	out_yw = np.zeros((n_weeks,168))
	for i in range(n_weeks):
		if i==52:
			out_yw[i,:] = x_pred_week_n[i,:]/np.mean(x_pred_week_n[i,:])*n[0,i-1]
		else:
			out_yw[i,:] = x_pred_week_n[i,:]/np.mean(x_pred_week_n[i,:])*n[0,i]
	out_yw = out_yw.flatten()
	out_yw = out_yw[:8764]

	#filter_coeffs_summer_1 = [-0.17305649,  1.01926796,  0.15955196, -0.02145443]
	#filter_coeffs_summer_2 = [ 0.10098675,  0.36221281,  0.63368745, -0.12147718]
	#for i in range(14):
		#out_yw[(27+i)*168-1] = out_yw[(27+i)*168-3]*filter_coeffs_summer_1[0] + out_yw[(27+i)*168-2]*filter_coeffs_summer_1[1] + out_yw[(27+i)*168+1]*filter_coeffs_summer_1[2] + out_yw[(27+i)*168+2]*filter_coeffs_summer_1[3]
		#out_yw[(27+i)*168] = out_yw[(27+i)*168-3]*filter_coeffs_summer_2[0] + out_yw[(27+i)*168-2]*filter_coeffs_summer_2[1] + out_yw[(27+i)*168+1]*filter_coeffs_summer_2[2] + out_yw[(27+i)*168+2]*filter_coeffs_summer_2[3]
	
	filter_coeffs = [-0.11796032, 0.40664512, 0.5, 0.27327281, -0.06260389]
	for i in range(n_weeks-1):
		out_yw[(i+1)*168-2] = out_yw[(i+1)*168-4]*filter_coeffs[0] + out_yw[(i+1)*168-3]*filter_coeffs[1] + out_yw[(i+1)*168-2]*filter_coeffs[2] + out_yw[(i+1)*168-1]*filter_coeffs[3] + out_yw[(i+1)*168]*filter_coeffs[4] 
	out_yw[(i+1)*168-1] = out_yw[(i+1)*168-3]*filter_coeffs[0] + out_yw[(i+1)*168-2]*filter_coeffs[1] + out_yw[(i+1)*168-1]*filter_coeffs[2] + out_yw[(i+1)*168]*filter_coeffs[3] + out_yw[(i+1)*168+1]*filter_coeffs[4] 
	out_yw[(i+1)*168] = out_yw[(i+1)*168-2]*filter_coeffs[0] + out_yw[(i+1)*168-1]*filter_coeffs[1] + out_yw[(i+1)*168]*filter_coeffs[2] + out_yw[(i+1)*168+1]*filter_coeffs[3] + out_yw[(i+1)*168+2]*filter_coeffs[4]
	out_yw[(i+1)*168+1] = out_yw[(i+1)*168-1]*filter_coeffs[0] + out_yw[(i+1)*168]*filter_coeffs[1] + out_yw[(i+1)*168+1]*filter_coeffs[2] + out_yw[(i+1)*168+2]*filter_coeffs[3] + out_yw[(i+1)*168+3]*filter_coeffs[4]


	return out_yw

def combine_weeks(n_weeks,x_pred_week_n):

	out_yw = x_pred_week_n.flatten()
	

	#filter_coeffs_summer_1 = [-0.17305649,  1.01926796,  0.15955196, -0.02145443]
	#filter_coeffs_summer_2 = [ 0.10098675,  0.36221281,  0.63368745, -0.12147718]
	#for i in range(n_weeks-1):
		#out_yw[(i+1)*168-1] = out_yw[(i+1)*168-3]*filter_coeffs_summer_1[0] + out_yw[(i+1)*168-2]*filter_coeffs_summer_1[1] + out_yw[(i+1)*168+1]*filter_coeffs_summer_1[2] + out_yw[(i+1)*168+2]*filter_coeffs_summer_1[3]
		#out_yw[(i+1)*168] = out_yw[(i+1)*168-3]*filter_coeffs_summer_2[0] + out_yw[(i+1)*168-2]*filter_coeffs_summer_2[1] + out_yw[(i+1)*168+1]*filter_coeffs_summer_2[2] + out_yw[(i+1)*168+2]*filter_coeffs_summer_2[3]

	filter_coeffs = [-0.11796032, 0.40664512, 0.5, 0.27327281, -0.06260389]
	for i in range(n_weeks-1):
		out_yw[(i+1)*168-2] = out_yw[(i+1)*168-4]*filter_coeffs[0] + out_yw[(i+1)*168-3]*filter_coeffs[1] + out_yw[(i+1)*168-2]*filter_coeffs[2] + out_yw[(i+1)*168-1]*filter_coeffs[3] + out_yw[(i+1)*168]*filter_coeffs[4] 
	out_yw[(i+1)*168-1] = out_yw[(i+1)*168-3]*filter_coeffs[0] + out_yw[(i+1)*168-2]*filter_coeffs[1] + out_yw[(i+1)*168-1]*filter_coeffs[2] + out_yw[(i+1)*168]*filter_coeffs[3] + out_yw[(i+1)*168+1]*filter_coeffs[4] 
	out_yw[(i+1)*168] = out_yw[(i+1)*168-2]*filter_coeffs[0] + out_yw[(i+1)*168-1]*filter_coeffs[1] + out_yw[(i+1)*168]*filter_coeffs[2] + out_yw[(i+1)*168+1]*filter_coeffs[3] + out_yw[(i+1)*168+2]*filter_coeffs[4]
	out_yw[(i+1)*168+1] = out_yw[(i+1)*168-1]*filter_coeffs[0] + out_yw[(i+1)*168]*filter_coeffs[1] + out_yw[(i+1)*168+1]*filter_coeffs[2] + out_yw[(i+1)*168+2]*filter_coeffs[3] + out_yw[(i+1)*168+3]*filter_coeffs[4]

	out_yw = out_yw[:-164]

	return out_yw


def gen_week_completeyear(ltype):
	n_noise = 25
	n_types = 6
	g_learning_rate_week = 0.00005
	scaling_factor_week = 5.606 # This is the max value by which all of trX was scaled before training GAN. Generated profiles are scaled back by this factor
    
	## Generator
	# load json and create model
	json_file = open('g_model_week.json', 'r')
	loaded_g_model_json = json_file.read()
	json_file.close()
	loaded_g_model = tf.keras.models.model_from_json(loaded_g_model_json)
	loaded_g_model.load_weights("g_model_week.h5")
	loaded_g_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=g_learning_rate_week, decay=g_learning_rate_week/10))
	## Generate data 
	z_test = sample_Z(53, n_noise)
	z_test_ones = np.zeros((53, n_types))
	z_test_ones[:13, 0] = 1
	z_test_ones[13:27, 1] = 1
	z_test_ones[27:41, 2] = 1
	z_test_ones[41:, 3] = 1
	z_test_ones[:,4+ltype] = 1
	x_pred_week = loaded_g_model.predict([z_test, z_test_ones])
	print('number of negatives')
	for i in range(52):
		if sum(1 for number in x_pred_week[i,:] if number < 0.05)>0:
			j = 0
			while j == 0:
				z_test = sample_Z(1, n_noise)
				print('size of z_test:  ' + str(z_test.shape))
				print('size of z_ones:  ' + str(z_test_ones[i,:].shape))
				print(z_test_ones[i,:])
				x_pred_week[i,:] = loaded_g_model.predict([z_test, np.reshape(z_test_ones[i,:],(1,6))])
				if sum(1 for number in x_pred_week[i,:] if number < 0.05)==0:
					j=1
				
		#plt.figure()
		#plt.plot(x_pred_week[i,:])
		#plt.savefig('plots/negative_plot_'+str(i)+'.png')
		#plt.close()
			

	print(sum(1 for number in x_pred_week.flatten() if number < 0))
	x_pred_week = x_pred_week *scaling_factor_week*0.5

	return x_pred_week

def gen_30sec(n_30sec):
	n_noise = 25
	g_learning_rate = 0.0001
	scaling_factor_30sec_min = 0.964711 # 
	scaling_factor_30sec_max = 1.072352
	## Generator
	# load json and create model
	json_file = open('g_model_30sec.json', 'r')
	loaded_g_model_json = json_file.read()
	json_file.close()
	loaded_g_model = tf.keras.models.model_from_json(loaded_g_model_json)
	loaded_g_model.load_weights("g_model_30sec.h5")
	loaded_g_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=g_learning_rate, decay=g_learning_rate/10))
	## Generate data 
	z_test = sample_Z(n_30sec, n_noise)
	z_test_ones = np.zeros((n_30sec, 1))
	z_test_ones[:, 0] = 1    
	x_pred = loaded_g_model.predict([z_test, z_test_ones])
	x_pred = x_pred*(scaling_factor_30sec_max-scaling_factor_30sec_min)+scaling_factor_30sec_min

	return x_pred

def combine_hour30sec(n_30sec,x_pred_30sec,out_ywh):
	### Combine 30sec profiles with hours
	out_ywhp = np.zeros((n_30sec,900))
	for i in range(n_30sec):
		out_ywhp[i,:] = x_pred_30sec[i,:]/np.mean(x_pred_30sec[i,:])*out_ywh[i]
	out_ywhp = out_ywhp.flatten()
	return out_ywhp

def gen_week(ltype,n_weeks,season):
	print('reading model...')
	n_noise = 25
	n_types = 6
	g_learning_rate_week = 0.00005
	scaling_factor_week = 5.606 # This was the max value by which all of trX was scaled before training GAN. Generated profiles need to be scaled back by this factor
	## Generator
	# load json and create model
	json_file = open('g_model_week.json', 'r')
	loaded_g_model_json = json_file.read()
	json_file.close()
	loaded_g_model = tf.keras.models.model_from_json(loaded_g_model_json)
	loaded_g_model.load_weights("g_model_week.h5")
	loaded_g_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=g_learning_rate_week, decay=g_learning_rate_week/10))
	## Generate data 
	print('generating..')
	z_test = sample_Z(n_weeks, n_noise)
	z_test_ones = np.zeros((n_weeks, n_types))
	#####
	z_test_ones[:, season] = 1
	z_test_ones[:,4+ltype] = 1
	x_pred_week = loaded_g_model.predict([z_test, z_test_ones])
	x_pred_week = x_pred_week *scaling_factor_week*0.5
	print('shape of x_pred')
	print(x_pred_week.shape)
	
	return x_pred_week



def gen_year(ltype,n_years):
	ft = 12 # Number of features to use (rows of V^T)
	if ltype == 1:
		sv = np.load('sv_res.npy')
		pd = np.load('pd_res.npy')
	elif ltype == 0:
		sv = np.load('sv_ind.npy')
		pd = np.load('pd_ind.npy')
	U_new = np.zeros((n_years,ft))
	for i in range(n_years):
		for j in range(ft):
			U_new[i,j] = random.gauss(pd[j,0],pd[j,1])
	#print(U_new.shape)
	#print(sv.shape)
	syn = np.dot(U_new,sv)
	#print(syn)

	return syn
			


	
def sample_Z(m, n): 
	return np.random.normal(loc=0,scale=0.2,size=[m, n])	
		
	
def getseason(season):
	if season == 'Winter':
		season_int = 0
	elif season == 'Spring':
		season_int = 1
	elif season == 'Summer':
		season_int = 2
	elif season == 'Fall':
		season_int = 3

	return int(season_int)
		
		
def getperiod(res_unit):
	if res_unit == 'Seconds':
		res_unit_int = 1
	elif res_unit == 'Minutes':
		res_unit_int = 60
	elif res_unit == 'Hours':
		res_unit_int = 3600
	elif res_unit == 'Days':
		res_unit_int = 86400
	elif res_unit == 'Weeks':
		res_unit_int = 604800

	return res_unit_int

def getlenunit(len_unit):
	if len_unit == 'Seconds':
		len_unit_int = 1
	elif len_unit == 'Minutes':
		len_unit_int = 60
	elif len_unit == 'Hours':
		len_unit_int = 3600
	elif len_unit == 'Days':
		len_unit_int = 86400
	elif len_unit == 'Weeks':
		len_unit_int = 604800
	elif len_unit == 'Year':
		len_unit_int = 31536000

	return len_unit_int

def click_reset():
	txt_out.delete(0,END)
	txt_out.insert(0,defaultfile)
	

window = Tk()

window.title('LoadGAN App')
window.geometry('650x750')
#window.configure(bg='#68BCF2')

s = ttk.Style()
s.configure('TFrame')

frame_title = ttk.Frame(window, style ='TFrame')
frame_title.pack(side=TOP,fill=X,pady=0)

frame_numb = ttk.Frame(window, style ='TFrame')
frame_numb.pack(side=TOP,fill=X,pady=10)

frame_tres = ttk.Frame(window, style ='TFrame')
frame_tres.pack(side=TOP,fill=X,pady=10)

frame_aggr = ttk.Frame(window, style ='TFrame')
frame_aggr.pack(side=TOP,fill=X,pady=10)

frame_tlen = ttk.Frame(window, style ='TFrame')
frame_tlen.pack(side=TOP,fill=X,pady=10)

frame_seas = ttk.Frame(window, style ='TFrame')
frame_seas.pack(side=TOP,fill=X,pady=10)

frame_output = ttk.Frame(window, style ='TFrame')
frame_output.pack(side=TOP,fill=X,pady=10)

frame_size = ttk.Frame(window, style ='TFrame')
frame_size.pack(side=TOP,fill=X,pady=10)

frame_gen = ttk.Frame(window, style ='TFrame')
frame_gen.pack(side=TOP,fill=X,pady=10)

frame_bottom = ttk.Frame(window, style ='TFrame')
frame_bottom.pack(side=BOTTOM,fill=X,pady=10)


fontsize = 15

## Title
lbl_title = Label(frame_title, text='LoadGAN', font=("Arial Bold", 50),borderwidth=2,relief='ridge').pack(pady=30)


## Number of profiles
lbl_number = Label(frame_numb, text='Number of profiles:', font=("Arial Bold", fontsize),padx=20,anchor=W).pack(side=LEFT)
lbl_res = Label(frame_numb, text='Residential', font=("Arial", fontsize),padx=5).pack(side=LEFT)
res_var = IntVar(frame_numb,value=1)
txt_res = Entry(frame_numb, textvariable=res_var, width=5,font=("Arial", fontsize),justify='center').pack(side=LEFT)
lbl_ind = Label(frame_numb, text='Industrial', font=("Arial", fontsize),width=10,anchor=E).pack(side=LEFT,anchor=W)
ind_var = IntVar(frame_numb,value=0)
txt_ind = Entry(frame_numb, textvariable=ind_var, font=("Arial", fontsize),width=5, justify='center').pack(side=LEFT)


## Time resolution
lbl_tres = Label(frame_tres, text='Time resolution:', font=("Arial Bold", fontsize),padx=20,anchor=W).pack(side=LEFT)
 # new (select any number of samples / period)
numsamples_var = IntVar(frame_tres, value = 2)
txt_tres = Entry(frame_tres, textvariable=numsamples_var, width=3,justify='center',font=("Arial", fontsize)).pack(side=LEFT,anchor=E,padx=0)
lbl_tres2 = Label(frame_tres, text='samples /', font=("Arial", fontsize)).pack(side=LEFT,anchor=W, padx=00)
numperiods_var = IntVar(frame_tres, value = 1)
txt_tres2 = Entry(frame_tres, textvariable=numperiods_var, width=2,justify='center',font=("Arial", fontsize)).pack(side=LEFT,padx=0)
resunits_var = StringVar(frame_tres,value='Minutes')
resunits_tres = ttk.Combobox(frame_tres, values=['Seconds','Minutes','Hours','Days','Weeks'], textvariable = resunits_var,font=("Arial", fontsize),width=8).pack(side=LEFT)
 # old (only initial 4 levels)
#var_tres = IntVar(frame_tres, value=2)
#c1 = Radiobutton(frame_tres, variable = var_tres, value = 1, text='1 sample/week', font=("Arial", fontsize)).pack(anchor=W)
#c2 = Radiobutton(frame_tres, variable = var_tres, value = 2, text='1 sample/hour',font=("Arial", fontsize)).pack(anchor=W)
#c3 = Radiobutton(frame_tres, variable = var_tres, value = 3, text='2 samples/minute',font=("Arial", fontsize)).pack(anchor=W)
#c4 = Radiobutton(frame_tres, variable = var_tres, value = 4, text='30 samples/second',font=("Arial", fontsize)).pack(anchor=W)

## Aggregation
lbl_aggr = Label(frame_aggr, text='Aggregation type:', font=("Arial Bold", fontsize),padx=20,anchor=W).pack(side=LEFT)
agg_var = StringVar(frame_aggr,value='Mean')
agg_aggr = ttk.Combobox(frame_aggr, values=['Minimum','Mean','Maximum'], textvariable = agg_var,font=("Arial", fontsize),width=8, justify='center').pack(side=LEFT)
lbl_aggr2 = Label(frame_aggr, text='value over the sampling period', font=("Arial", fontsize)).pack(side=LEFT,anchor=W, padx=00)

## Time length
lbl_tlen = Label(frame_tlen, text='Time length:', font=("Arial Bold", fontsize),padx=20,anchor=W).pack(side=LEFT)
tlen_var = IntVar(frame_tlen, value = 1)
txt_tlen = Entry(frame_tlen, textvariable=tlen_var, width=5,justify='center',font=("Arial", fontsize)).pack(side=LEFT,padx=35)
units_var = StringVar(frame_tlen,value='Days')
units_tlen = ttk.Combobox(frame_tlen, values=['Seconds','Minutes','Hours','Days','Weeks','Year'], textvariable = units_var,font=("Arial", fontsize),width=10).pack(side=LEFT)
lbl2_tlen = Label(frame_tlen, text='(up to 1 year)', font=("Arial", fontsize),padx=10,anchor=W).pack(side=LEFT)


## Season
lbl_seas = Label(frame_seas, text='Season:', font=("Arial Bold", fontsize),padx=20,anchor=W).pack(side=LEFT,anchor=W)
seas_var = StringVar(frame_seas,value='Summer')
seas_drop = ttk.Combobox(frame_seas, values=['Winter','Spring','Summer','Fall'], textvariable = seas_var, width=10, font=("Arial", fontsize)).pack(side=LEFT)
lbl_tlen2 = Label(frame_seas, text='(ignored when creating full year)', font=("Arial", fontsize)).pack(side=BOTTOM,anchor=W, padx=10)


## Output filename
lbl_out = Label(frame_output, text='Output file name:', font=("Arial Bold", fontsize),padx=20,anchor=W)#.pack(side=LEFT,anchor=W)
lbl_out.grid(row=0,column=0)
out_var = StringVar(frame_output,value=defaultfile)
txt_out = Entry(frame_output, textvariable=out_var, width=50,justify='left',font=("Arial", 10))
txt_out.grid(row=0,column=1)#.pack(side=LEFT)
btn_reset = Button(frame_output, text='Reset path', font=("Arial", 10), command=click_reset)#.pack(side=BOTTOM)
btn_reset.grid(row=1,column=1,sticky=E)

## Output size
lbl_size = Label(frame_size, text='Estimated output file size:', font=("Arial Bold", fontsize),padx=20,anchor=W)#.pack(side=LEFT,anchor=W)
lbl_size.grid(row=0,column=0)
size_var = StringVar(frame_size,value='0 MB')
txt_size = Entry(frame_size, textvariable=size_var, width=20,justify='left',font=("Arial", 10))
txt_size.grid(row=0,column=1)#.pack(side=LEFT)
btn_size = Button(frame_size, text='Update', font=("Arial", 10), command=click_size)#.pack(side=BOTTOM)
btn_size.grid(row=0,column=2,sticky=E)


## Generate
btn_generate = Button(frame_gen, text='Generate data', bg='green', fg='black', font=("Arial Bold", 25), command=click_generate).pack(side=BOTTOM,pady=20)

#txt_info = StringVar(window,value='Ready')
#lbl_clicked = Label(window, text=txt_info).pack()

currentport = Listbox(window,height=1,bg='#DEE6ED',justify='center')
currentport.pack()
currentport.insert(0,'Ready')

## Bottom
lbl_bottom = Label(frame_bottom, text='Andrea Pinceti, Arizona State University, 2021').pack()
#lbl_bottom.grid(row=0, column=1,sticky=E)
lbl_bottom = Label(frame_bottom, text='a.pinceti@asu.edu, https://github.com/apince/LoadGAN').pack()
#lbl_bottom.grid(row=1, column=1,sticky=E)





window.mainloop()


