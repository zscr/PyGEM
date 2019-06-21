#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:09:42 2019

@author: zoescrewvala
"""
import os
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

#%% X-Y PLOT
ds_mcnabb = pd.read_csv(os.getcwd() + '/Alaska_dV_17jun_preprocessed.csv')
ds_larsen = pd.read_csv(os.getcwd() + '/larsen2015_supplementdata_wRGIIds.csv')

names_m = ds_mcnabb.loc[:,'RGIId']
#names_m = names_m.astype(str)
names_l = ds_larsen.loc[:,'RGIId']
names_l = names_l.astype(str)
mb_m = ds_mcnabb.loc[:,'mb_mwea']
mb_l = ds_larsen.loc[:,'mb_mwea']
# use area from mcnabb, they should both be the same...
area = ds_mcnabb.loc[:,'area']

i = 0
j = 0
k = 0

# mb_m_comp = []
# mb_l_comp = []
# area_comp = []

# array of indices: first row mcnabb mb, second row larson mb
mb_compare = np.empty((0,2), dtype=str)

for i in range(len(names_m)):
    j = 0
    for j in range(len(names_l)):
        if names_m[i] == names_l[j]:
            np.append(mb_compare, [i, i, j], axis=0)
        j = j + 1
    i = i + 1

# X,Y values
x_values = area.loc[mb_compare[:,0]]
y_values = mb_m.loc[mb_compare[:,0]]
y2_values = mb_l.loc[mb_compare[:,1]]

# Set up your plot (and/or subplots)
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
             
# Plot
#  zorder controls the order of the plots (higher zorder plots on top)
#  label used to automatically generate legends (legends can be done manually for more control)
#ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=2, label='plot1')
ax[0,0].scatter(x_values, y_values, color='b', zorder=2, s=2, label = 'McNabb Data')
#ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='plot2')
ax[0,0].scatter(x_values, y2_values, color='r', zorder=2, s=2, label = 'Larsen Data')

# Fill between
#  fill between is useful for putting colors between plots (e.g., error bounds)
#ax[0,0].fill_between(x, y_low, y_high, facecolor='k', alpha=0.2, zorder=1)

# Text
#  text can be used to manually add labels or to comment on plot
#  transform=ax.transAxes means the x and y are between 0-1
ax[0,0].text(0.5, 0.99, 'Glacier mass balance vs area', size=10, horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('area [m^2]', size=12)
#ax[0,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), time_values_annual[t1_idx:t2_idx].max())
#ax[0,0].xaxis.set_tick_params(labelsize=12)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('mass balance [m w.e. yr^-1]', size=12)
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
figure_fp = os.getcwd() + '/../Output/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'compare_mb_data.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)