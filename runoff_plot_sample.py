#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:05:29 2019

@author: zoescrewvala
"""

import os
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import pygemfxns_gcmbiasadj as gcmbiasadj

#%% X-Y PLOT
#ds = Dataset(os.getcwd() + '/Output/simulations/CanESM2/R1_CanESM2_rcp26_c1_ba1_1sets_2000_2100.nc')
ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/CanESM2/R1_CanESM2_rcp45_c1_ba1_1sets_1950_2000.nc')
#dens_ice = 917 # in kg/m^3
#mb = ds.loc[:,'mb_mwea']
#area = ds.loc[:,'area']
#mb_uncertainty = ds.loc[:,'mb_mwea_sigma']

# variables for vol over time plot
# variables
time = ds.variables['year'].values[:]
glac_runoff = ds.variables['runoff_glac_monthly'].values[:]
offglac_runoff = ds.variables['offglac_runoff_monthly'].values[:]
total_runoff = glac_runoff + offglac_runoff
total_runoff = total_runoff[:,:,0]
runoff_region = np.sum(total_runoff, axis=0)
runoff_init = np.sum(total_runoff[:,0])
runoff_norm = runoff_region/runoff_init
runoff_region_annual = gcmbiasadj.annual_sum_2darray(total_runoff)
runoff_region_annual = np.sum(runoff_region_annual, axis=0)

# X,Y values
x_values = time  
y_values = runoff_region_annual
#y2_values = ds.loc[...]

# Set up your plot (and/or subplots)
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
             
# Plot
#  zorder controls the order of the plots (higher zorder plots on top)
#  label used to automatically generate legends (legends can be done manually for more control)
ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=2, label='plot1')
#ax[0,0].scatter(x_values, y_values, color='k', zorder=2, s=2)

#ax[0,0].scatter(x_values, y_values[7,:], color='m', zorder=2, s=2)

#ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='plot2')

# Fill between
#  fill between is useful for putting colors between plots (e.g., error bounds)
#ax[0,0].fill_between(x, y_low, y_high, facecolor='k', alpha=0.2, zorder=1)

# Text
#  text can be used to manually add labels or to comment on plot
#  transform=ax.transAxes means the x and y are between 0-1
ax[0,0].text(0.5, 1.03, 'Test glaciers', size=10, horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('Year', size=12)
#ax[0,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), time_values_annual[t1_idx:t2_idx].max())
#ax[0,0].xaxis.set_tick_params(labelsize=12)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('Runoff [m^3]', size=12)
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
#ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
#               frameon=False)
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
figure_fn = 'runoff_plot_sample.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
