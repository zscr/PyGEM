#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:02:42 2019

@author: zoescrewvala
"""

import os
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr

#%% X-Y PLOT
#ds = Dataset(os.getcwd() + '/Output/simulations/CanESM2/R1_CanESM2_rcp26_c1_ba1_1sets_2000_2100.nc')
sim_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
rcp_list = ['26', '45', '85']
color_list = ['b', 'k', 'r']
vol = []
end_index = 121 # last timestep

#%% Normalized volume plot
# set up plot
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})

for j in range(len(rcp_list)):
    vol_regional_all = np.zeros((len(sim_list),122), dtype=float)
    for i  in range(len(sim_list)):
        ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/' + sim_list[i] + '/R1_' + sim_list[i] + '_rcp' + rcp_list[j] + '_c1_ba1_1sets_1980_2100.nc')
    #    time = np.append(arr=time, values=ds.variables['year_plus1'].values[:])
        time = ds.variables['year_plus1'].values[:]
#        start_index = np.where(time == 2017)
    #    vol = np.append(arr=vol, values=ds.variables['volume_glac_annual'].values[:,:,0])
        vol = ds.variables['volume_glac_annual'].values[:,:,0]
        vol_regional = np.sum(vol, axis=0)
        vol_regional_norm = vol_regional/vol_regional[0]
        vol_regional_all[i] = vol_regional_norm
    vol_regional_average = np.average(vol_regional_all, axis=0)
    std = np.std(vol_regional_all, axis=0)
    x_values=time[37:121]
    y_values=vol_regional_average[37:121]
    error = std[37:121]
#    ax[0,0].scatter(time, vol_regional, color='k', zorder=2, s=2)
    ax[0,0].plot(x_values, y_values, color=color_list[j], linewidth=1, zorder=2, label=rcp_list[j])
    plt.fill_between(x_values, y_values-error, y_values+error, alpha=0.3, linewidth=0.4, color=color_list[j], linestyle = '--')
    ds.close()
    
#%%
#dens_ice = 917 # in kg/m^3
#mb = ds.loc[:,'mb_mwea']
#area = ds.loc[:,'area']
#mb_uncertainty = ds.loc[:,'mb_mwea_sigma']

# variables for vol over time plot
# variables
#time = ds.variables['year_plus1'].values[:]
#vol = ds.variables['volume_glac_annual'].values[:,:,0]
#vol_regional = np.sum(vol, axis=0)
#vol_init = np.sum(vol[:,:,0][:,-1])
#vol_norm = vol_norm/vol_init

# X,Y values
#x_values = time  
#y_values = vol_regional/vol_regional[0]
#y2_values = ds.loc[...]

# Set up your plot (and/or subplots)
#fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
             
# Plot
#  zorder controls the order of the plots (higher zorder plots on top)
#  label used to automatically generate legends (legends can be done manually for more control)
#ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=2, label='plot1')
#ax[0,0].scatter(x_values, y_values, color='k', zorder=2, s=2)

#ax[0,0].scatter(x_values, y_values[7,:], color='m', zorder=2, s=2)

#ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='plot2')

# Fill between
#  fill between is useful for putting colors between plots (e.g., error bounds)
#ax[0,0].fill_between(x, y_low, y_high, facecolor='k', alpha=0.2, zorder=1)

# Text
#  text can be used to manually add labels or to comment on plot
#  transform=ax.transAxes means the x and y are between 0-1
ax[0,0].text(0.5, 0.99, 'All RCPs, averaged for 10 GCMs', size=10, horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('Year', size=12)
#ax[0,0].set_xlim(0,1.1)
#ax[0,0].xaxis.set_tick_params(labelsize=12)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('Volume (-)', size=12)
ax[0,0].set_ylim(0,1.1)
#ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))x
#ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Tick parameters
#  controls the plotting of the ticks
#ax[0,0].yaxis.set_ticks_position('both')
#ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')
#ax[0,0].tick_params(axis='both', which='minor', labelsize=12, direction='inout')               
    
# Example Legend
# Option 1: automatic based on labels
#ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
#               frameon=False)
# Option 2: manually define legend
leg_lines = []
labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
label_colors = ['b', 'k', 'r']
for nlabel, label in enumerate(labels):
    line = Line2D([0,1],[0,1], color=label_colors[nlabel], linewidth=1)
    leg_lines.append(line)
ax[0,0].legend(leg_lines, labels, loc=(0.05,0.05), fontsize=10, 
  labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)

# Save figure
#  figures can be saved in any format (.jpg, .png, .pdf, etc.)
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/plots/vol_norm/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'vol_norm_allrcps_avg.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
