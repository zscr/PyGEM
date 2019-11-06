#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:15:15 2019

@author: zoescrewvala
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
#from datetime import datetime
import matplotlib.pyplot as plt

# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
#import pygemfxns_massbalance as massbalance
#import class_climate
import class_mbdata
from datetime import datetime
#%% CHECKING HINDCAST SIM WITH CAL DATA ~ 16 sec

startTime = datetime.now()

# open sim data -- csv file
ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/merged/ERA5/R1--all--ERA5_c4_ba1_1sets_1980_2018.nc')
df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)

# Glacier selection
rgi_regionsO1 = [1]
#rgi_glac_number = ['14683']
rgi_glac_number = input.get_same_glaciers(os.getcwd() + '/../Output/cal_opt4/')
startyear = 1980
endyear = 2018

# Select glaciers
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', rgi_glac_number=rgi_glac_number)
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.hyps_filepath, input.hyps_filedict, input.hyps_colsdrop)
# Determine dates_table_idx that coincides with data  rgi_regionsO1
dates_table = modelsetup.datesmodelrun(startyear, endyear, spinupyears=0)

elev_bins = main_glac_hyps.columns.values.astype(int)
elev_bin_interval = elev_bins[1] - elev_bins[0]

#cal_datasets = ['larsen']

cal_data = pd.DataFrame()
#for dataset in cal_datasets:

print(datetime.now() - startTime)

#%%  ~1 min 49 sec
startTime = datetime.now()

cal_subset = class_mbdata.MBData(name='braun') #, rgi_region=rgi_regionsO1[0]
cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
cal_data = cal_data.append(cal_subset_data, ignore_index=True)
cal_data = cal_data.sort_values(['glacno', 't1_idx'])
cal_data.reset_index(drop=True, inplace=True)

ds2 = cal_data
#df2 = pd.DataFrame(ds2.glacier_table.values, columns=ds2.glac_attrs.values)

#ds2 = pd.read_csv(os.getcwd() + '/../DEMs/larsen/larsen2015_supplementdata_wRGIIds.csv')
#ds2 = pd.read_csv(os.getcwd() + '/../DEMs/McNabb_data/wgms_dv/Alaska_dV_17jun_preprocessed.csv')
larsen_glac = input.get_same_glaciers(os.getcwd() + '/../Output/cal_opt1/reg1')
larsen_glac = ['RGI60-01.' + x for x in larsen_glac]
#glac_idxs = ds2.RGIId
#glac_idxs = glac_idxs.index.tolist(larsen_glac)
mb_sim = ds.variables['massbaltotal_glac_monthly'].values[:,:,0]
#mb_cal = ds2.mb_mwe
num_glac = ds2.shape[0]

# get index of start dates for all glaciers -- t1_idx is a list
t1_idx = ds2.t1_idx.values
t2_idx = ds2.t2_idx.values.astype(int)
years = (t2_idx - t1_idx + 1)/12
sigma = ds2.mb_mwe_err/years


#%% ~0.2 sec

# get yearly mass balance sum from simulation -- change to processing as matrix not list
mb_subset_sum = np.empty(num_glac)
zscore = np.empty(num_glac)
diff = np.empty(num_glac)
for i in range(0, num_glac - 1):
     if i%100==0:
         print(i)
     mb_partial = mb_sim[i, t1_idx[i]:t2_idx[i] + 1]
     mb_subset_sum[i] = (np.sum(mb_partial))
mb_sim_mwea = mb_subset_sum/years
#    mb_subset_sum[i] = (np.sum(mb_partial)/years[i])
mb_cal = ds2.mb_mwe/years
##    print(mb_subset_sum[i], mb_cal[i])
diff = mb_sim_mwea - mb_cal
zscore = diff/sigma
#    print(larsen_glac[i] + ': ' + str(zscore[i]))

#    print(zscore[i])

print(datetime.now() - startTime)

#%% X-Y plot ~0.5 sec
startTime = datetime.now()

x_values = mb_cal
y_values = mb_subset_sum
x2_values = [-6, 1]
y2_values = [-6, 1]

fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
ax[0,0].scatter(x_values, y_values, color='k', zorder=2, s=2)
ax[0,0].plot(x2_values, y2_values, color='b', linewidth=1, zorder=2, label='1:1 line')
ax[0,0].text(0.5, 1.05, 'Checking ERA-Interim Hindcast', size=10, horizontalalignment='center', verticalalignment='top', transform=ax[0,0].transAxes)
ax[0,0].set_xlabel('Mass balance calibration data [m w.e. yr^-1]', size=12)
ax[0,0].set_ylabel('Simulation-produced mass balance [m w.e. yr^-1]', size=12)
ax[0,0].set_xlim(left=-6, right=1)
ax[0,0].set_ylim(top=1, bottom=-6)

# save figure
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/plots/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'hindcast_check_plot_xy.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)

print(datetime.now() - startTime)

##%% HISTOGRAM PLOT
#diff = mb_subset_sum - mb_cal
#zscore = diff/sigma
##x_values = np.linspace(1, num_glac, num_glac)
#num_off = 0
#for i  in range(0, 115):
#    if zscore[i] > 10 or zscore[i] < -10:
#        print(larsen_glac[i] + ': ' + str(zscore[i]))
#        num_off = num_off + 1
#print(str(num_off))
#
#fig, ax = plt.subplots()
#ax.hist(zscore, bins=num_glac)
#ax.text(0.5, 1.05, 'Checking ERA-Interim Hindcast', size=10, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
#ax.set_xlabel('Z-score', size=10)
#ax.set_ylabel('Number of glaciers', size=10)
## save figure
#fig.set_size_inches(4, 4)
#figure_fp = os.getcwd() + '/../Output/plots/'
#if os.path.exists(figure_fp) == False:
#    os.makedirs(figure_fp)
#figure_fn = 'hindcast_check_plot_hist.png'
#fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)

##%% UNCERTAINTY PLOT
##fig, ax = plt.subplots()
##ax.scatter(np.absolute(diff), np.absolute(sigma), s=2)
##ax.text(0.5, 1.05, 'Checking ERA-Interim Hindcast', size=10, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
##ax.set_xlabel('Difference between mass balance calibration data and simulation-produced mass balance', size=10)
##ax.set_ylabel('Uncertainty of mass balance measurements', size=10)
##
### save figure
##fig.set_size_inches(4, 4)
##figure_fp = os.getcwd() + '/../Output/plots/'
##if os.path.exists(figure_fp) == False:
##    os.makedirs(figure_fp)
##figure_fn = 'hindcast_check_plot_uncertainties.png'
##fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)





###
## dates
#sim_dates = pd.date_range(start='10/1/1979', end='9/1/2017', freq='MS')
#sim_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(sim_dates.values)]
#sim_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in sim_dates]
#
#start_dates = ds2.loc[:,'date0']
#start_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(start_dates.values)]
#start_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in start_dates]
#
#end_dates = ds2.loc[:,'date1']
#end_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(end_dates.values)]
#end_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in end_dates]
    
    #for i in range(0, num_glac):
#    for j in range(0, len(sim_dates)):
#        if start_dates[i].year == sim_dates[j].year and start_dates[i].month == sim_dates[j].month:
#            t1_idx[i] = j
#            print(start_dates[i] + ' ' + sim_dates[j])
#        if end_dates[i].year == sim_dates[j].year and end_dates[i].month == sim_dates[j].month:
#            t2_idx[i] = j



