#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:51:16 2019

@author: zoescrewvala
"""
import os
#import cartopy
import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import time
#%% X-Y PLOT
ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/ERA-Interim/R1_ERA-Interim_c1_ba1_1sets_1980_2017.nc')
mb = ds.variables['massbaltotal_glac_monthly'].values[:,:,0]
ds2 = pd.read_csv(os.getcwd() + '/../DEMs/larsen/larsen2015_supplementdata_wRGIIds.csv')
start_dates = ds2.loc[:,'date0']
start_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(start_dates.values)]
#start_dates = [datetime.strptime(x, '%Y-%m-%d') for x in start_dates]

end_dates = ds2.loc[:,'date1']
end_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(end_dates.values)]

sim_dates = pd.date_range(start='10/1/1979', end='9/1/2017', freq='MS')
sim_dates = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(sim_dates.values)]


i = 0
j = 0
num_years = 0
sum_temp = 0
mb_sum = []
test = np.ones((116, 456), dtype=int)
for i in range(0, test.shape[0]):
    for j in range(0, test.shape[1]):
        if (time.strptime(sim_dates[j], '%Y-%m-%d') > time.strptime(start_dates[i], '%Y-%m-%d')) and (time.strptime(sim_dates[j], '%Y-%m-%d') < time.strptime(end_dates[i], '%Y-%m-%d')):
            sum_temp = sum_temp + mb[i, j]
            num_years = num_years + 1/12
    sum_temp = sum_temp/num_years
    mb_sum.extend([sum_temp])
    sum_temp = 0
    num_years = 0
    
### USING TEST, mb_sum SHOULD BE A LIST OF 116 VALUES EACH WITH AN INT < OR = 456
    
    np.sum(mb, axis=1)
    
cal_mb = ds2.loc[:,'mb_mwea']
time = ds.time.values

#time = ds.variables['year'].values[:]

# X,Y values
x_values = mb_sum
y_values = cal_mb
y2_values = mb_sum

# Set up your plot (and/or subplots)
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
             
# Plot
#  zorder controls the order of the plots (higher zorder plots on top)
#  label used to automatically generate legends (legends can be done manually for more control)
#ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=2, label='Mass Balance')
ax[0,0].scatter(x_values, y_values, color='k', zorder=2, s=2)
ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='1:1 line')

# Fill between
#  fill between is useful for putting colors between plots (e.g., error bounds)
#ax[0,0].fill_between(x, y_low, y_high, facecolor='k', alpha=0.2, zorder=1)

# Text
#  text can be used to manually add labels or to comment on plot
#  transform=ax.transAxes means the x and y are between 0-1
ax[0,0].text(0.5, 1.1, 'Checking ERA-Interim Hindcast', size=10, horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('Mass balance data used for calibration [m w.e. yr^-1]', size=12)
#ax[0,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), time_values_annual[t1_idx:t2_idx].max())
#ax[0,0].xaxis.set_tick_params(labelsize=12)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('Simulation-produced mass balance [m w.e. yr^-1]', size=12)
#ax[0,0].set_ylim(0,1.1)
#ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
#ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Tick parameters
#  controls the plotting of the ticks
#ax[0,0].yaxis.set_ticks_position('both')
#ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')
#ax[0,0].tick_params(axis='both', which='minor', labelsize=12, direction='inout')               
    
# Example Legend
# Option 1: automatic based on labels
ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
               frameon=False)
# Option 2: manually define legend
#leg_lines = []
#labels = ['plot1', 'plot2']
#label_colors = ['k', 'b']
#for nlabel, label in enumerate(labels):
#    line = Line2D([0,1],[0,1], color=label_colors[nlabel], linewidth=1)
#    leg_lines.append(line)
#ax[0,0].legend(leg_lines, labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#               handletextpad=0.25, borderpad=0, frameon=False)

# Save figure
#  figures can be saved in any format (.jpg, .png, .pdf, etc.)
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/plots/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'hindcast_check_plot.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)