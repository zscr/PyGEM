#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:00:39 2019

@author: zoescrewvala
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
#from datetime import datetime

#%% CHECKING HINDCAST SIM WITH CAL DATA
ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/ERA-Interim/R1_ERA-Interim_c1_ba1_1sets_1980_2017.nc')
ds2 = pd.read_csv(os.getcwd() + '/../DEMs/larsen/larsen2015_supplementdata_wRGIIds.csv')
mb_sim = ds.variables['massbaltotal_glac_monthly'].values[:,:,0]
cal_mb = ds2.loc[:,'mb_mwea']
sim_dates = pd.date_range(start='10/1/1979', end='9/1/2017', freq='MS')

#%% CANWELL GLACIER # start date - 5/24/05 / end date - 5/24/13
start_date = '2005-05-01'
end_date = '2013-05-01'
glac_name = 'Canwell'
glac_idx = 11
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()

#%% BLACK RAPIDS GLACIER # start date - 5/19/95 / end date - 5/19/13
start_date = '1995-05-01'
end_date = '2013-05-01'
glac_name = 'Black Rapids'
glac_idx = 24
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()

#%% GULKANA GLACIER # start date - 5/17/95 / end date - 5/17/13
start_date = '1995-05-01'
end_date = '2013-05-01'
glac_name = 'Gulkana'
glac_idx = 25
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()

#%% KENNICOTT GLACIER # start date - 6/16/00 / end date - 6/16/13
start_date = '2000-06-01'
end_date = '2013-06-01'
glac_name = 'Kennicott'
glac_idx = 27
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()
#%% SEWARD GLACIER # start date - 8/26/00 / end date - 8/26/13
start_date = '2000-08-01'
end_date = '2013-08-01'
glac_name = 'Seward'
glac_idx = 41
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()
#%% MATANUSKA GLACIER # start date - 9/6/04 / end date - 9/6/12
start_date = '2004-09-01'
end_date = '2012-09-01'
glac_name = 'Matanuska'
glac_idx = 56
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()
#%% VALDEZ GLACIER # start date - 8/25/00 / end date - 8/25/13
start_date = '2000-08-01'
end_date = '2013-08-01'
glac_name = 'Valdez'
glac_idx = 60
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()
#%% COLUMBIA GLACIER # start date - 9/6/04 / end date - 9/6/13
start_date = '2004-09-01'
end_date = '2013-09-01'
glac_name = 'Columbia'
glac_idx = 75
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()
#%% MENDENHALL GLACIER # start date - 8/26/00 / end date - 8/26/13
start_date = '2000-08-01'
end_date = '2013-08-01'
glac_name = 'Mendenhall'
glac_idx = 108
#==========================================^change for each glacier / vSame for each glacier
t1_idx = sim_dates.get_loc(start_date)
t2_idx = sim_dates.get_loc(end_date)
years = (t2_idx - t1_idx + 1)/12
mb_sim_glac = mb_sim[glac_idx]
glac_sum = (np.sum(mb_sim_glac[t1_idx:(t2_idx + 1)]))/years
glac_cal = cal_mb[glac_idx]
diff = glac_sum - glac_cal

print('Simulation mass balance for ' + glac_name + ' Glacier: ' + str(glac_sum))
print('Calibration data mass balance for ' + glac_name + ' Glacier: ' + str(glac_cal))
print('Difference between simulation and calibration: ' + str(diff))
print()