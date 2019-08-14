#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:53:40 2019

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
# alphabetical order -->
#sim_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
sim_list = ['CSIRO-Mk3-6-0', 'CNRM-CM5', 'GISS-E2-R', 'GFDL-ESM2M', 'CCSM4', 'MPI-ESM-LR', 'NorESM1-M', 'CanESM2', 'GFDL-CM3', 'IPSL-CM5A-LR']

rcp = '26'
massbal = []
massbal_regional = []
massbal_regional_all = np.zeros((len(sim_list),122), dtype=float)

# set up plot
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})

for i  in range(len(sim_list)):
# specific GCM
    ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/' + sim_list[i] + '/R1_' + sim_list[i] + '_rcp' + rcp + '_c1_ba1_1sets_1980_2100.nc')
    time = ds.variables['year_plus1'].values[:]
    vol = ds.variables['volume_glac_annual'].values[:,:,0]
    massbal_regional = np.sum(vol, axis=0)
    massbal_regional_norm = vol_regional/vol_regional[0]
    ax[0,0].plot(time[37:], vol_regional_norm[37:], linewidth=1, zorder=2, label=sim_list[i])
# GCM averages background
    massbal_regional_all[i] = vol_regional_norm
vol_regional_average = np.average(vol_regional_all, axis=0)
std = np.std(vol_regional_all, axis=0)
x_values=time[37:121]
y_values=vol_regional_average[37:121]
error = std[37:121]
ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=3, label='Average +/- st. dev.')
plt.fill_between(x_values, y_values-error, y_values+error, color='k', alpha=0.2, linewidth=0.4)
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
#ax[0,0].text(0.5, 0.99, 'RCP 8.5', size=10, horizontalalignment='center', verticalalignment='top', 
#             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('Year', size=15)
#ax[0,0].set_xlim(0,1.1)
ax[0,0].xaxis.set_tick_params(labelsize=15)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('Mass Balance (-)', size=15)
ax[0,0].set_ylim(0,1.1)
#ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))x
#ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax[0,0].yaxis.set_tick_params(labelsize=15)

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
fig.set_size_inches(8, 8)
figure_fp = os.getcwd() + '/../Output/plots/massbal/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'massbal_norm_rcp' + rcp + '.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
