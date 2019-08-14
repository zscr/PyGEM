#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:36:04 2019

@author: zoescrewvala
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import pygemfxns_gcmbiasadj as pygemfxns

#%%
sim_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
rcp = '85'
RCP = '8.5'

fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
y_values = np.linspace(-2,2)
ax[0,0].plot(y_values, y_values, color='b', linewidth=1, zorder=2, label='plot2')
y3_values = np.linspace(0,0)
ax[0,0].plot(y_values, y3_values, color='k', alpha=0.5, linewidth=1, zorder=1)

for i in range(len(sim_list)):   
    ds = pd.read_csv(os.getcwd() + '/../Climate_data/Zemp_etal_DataTables2a-t_results_regions_global/Zemp_etal_results_region_1_ALA.csv'
                     , skiprows=26)
    #time = ds.loc[:,'Year']
    mb = ds.loc[:, ' INT_mwe']
    mb = mb[10:]
    
    ds2 = xr.open_dataset(os.getcwd() + '/../Output/simulations/' + sim_list[i] + '/R1_' + sim_list[i] + '_rcp'
                          + rcp + '_c1_ba1_1sets_1960_2017.nc')
    #time = ds2.variables['year'].values[:]
    area = ds2.variables['area_glac_annual'].values[:,:-1,0]
    
    mb_mod = ds2.variables['massbaltotal_glac_monthly'].values[:,:,0]
    mb_mod_annual = ((pygemfxns.annual_sum_2darray(mb_mod))/1000)*area
    mb_mod_annual_regional = ((np.sum(mb_mod_annual, axis=0))/(np.sum(area, axis=0)))*1000
    
    # plot
    x_values = mb
    y_values = mb_mod_annual_regional[:-1]
    ax[0,0].scatter(x_values, y_values, linewidth=1, zorder=2, label='plot1', s=2)

# figure styling
ax[0,0].set_xlim(-2,2)
ax[0,0].text(0.5, 0.99, 'Comparison to Zemp et al.: RCP ' + RCP, size=10, horizontalalignment='center', verticalalignment='top', transform=ax[0,0].transAxes)
axes[0,0].set_xlabel('Mass balance from Zemp et al. [m w.e. yr^-1]', size=10)
axes[0,0].set_ylabel('Mass balance from PyGEM [m w.e. yr^-1]', size=10)


# save figure
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/plots/validation/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'validation_rcp' + rcp + '.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
plt.clf()
