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
from datetime import datetime
import matplotlib.pyplot as plt
#%% CHECKING HINDCAST SIM WITH CAL DATA

# open data
ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/ERA-Interim/R1_ERA-Interim_c1_ba1_1sets_1980_2017.nc')
ds2 = pd.read_csv(os.getcwd() + '/../DEMs/larsen/larsen2015_supplementdata_wRGIIds.csv')
mb_sim = ds.variables['massbaltotal_glac_monthly'].values[:,:,0]
mb_cal = ds2.loc[:,'mb_mwea']

# dates
sim_dates = pd.date_range(start='10/1/1979', end='9/1/2017', freq='MS')
sim_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(sim_dates.values)]
sim_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in sim_dates]

start_dates = ds2.loc[:,'date0']
start_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(start_dates.values)]
start_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in start_dates]

end_dates = ds2.loc[:,'date1']
end_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(end_dates.values)]
end_dates = [datetime.strptime(x, '%Y-%m-%d').date() for x in end_dates]

# get index of start dates for all glaciers -- t1_idx is a list
t1_idx = np.empty(ds2.shape[0], dtype=np.int)
t2_idx = np.empty(ds2.shape[0], dtype=np.int)
for i in range(0, ds2.shape[0]):
    for j in range(0, len(sim_dates)):
        if start_dates[i].year == sim_dates[j].year and start_dates[i].month == sim_dates[j].month:
            t1_idx[i] = j
        if end_dates[i].year == sim_dates[j].year and end_dates[i].month == sim_dates[j].month:
            t2_idx[i] = j

# get yearly mass balance sum from simulation
years = (t2_idx - t1_idx + 1)/12
glac_sum = np.empty(ds2.shape[0])
for i in range(0, ds2.shape[0]):
    glac_sum[i] = (np.sum(mb_sim[i, t1_idx[i]:(t2_idx[i] + 1)]))/years[i]

# X-Y plot
x_values = mb_cal
y_values = glac_sum
y2_values = mb_cal

fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
ax[0,0].scatter(x_values, y_values, color='k', zorder=2, s=2)
ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='1:1 line')
ax[0,0].text(0.5, 1.05, 'Checking ERA-Interim Hindcast', size=10, horizontalalignment='center', verticalalignment='top', transform=ax[0,0].transAxes)
ax[0,0].set_xlabel('Mass balance calibration data [m w.e. yr^-1]', size=12)
ax[0,0].set_ylabel('Simulation-produced mass balance [m w.e. yr^-1]', size=12)
ax[0,0].set_xlim(left=-6, right=1)
ax[0,0].set_ylim(top=1, bottom=-6)
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/plots/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'hindcast_check_plot.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)









