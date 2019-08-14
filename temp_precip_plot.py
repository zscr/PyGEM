#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:34:08 2019

@author: zoescrewvala
"""

# external libraries
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# local libraries
import pygem_input as input
import class_climate
import pygemfxns_modelsetup as modelsetup
import pygemfxns_gcmbiasadj as pygemfxns

#%%
#gcm_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#            'MPI-ESM-LR', 'NorESM1-M']
gcm_list = ['CSIRO-Mk3-6-0', 'CNRM-CM5', 'GISS-E2-R', 'GFDL-ESM2M', 'CCSM4', 'MPI-ESM-LR', 'NorESM1-M', 'CanESM2', 'GFDL-CM3', 'IPSL-CM5A-LR']

#gcm_list = ['CanESM2']

rcp_list = ['rcp26', 'rcp45', 'rcp85']
RCP_list = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']

fig, ax = plt.subplots(2, 3, squeeze=False, sharex='col', sharey='row', 
             gridspec_kw = {'wspace':0.1, 'hspace':0.05})

for j in range(len(rcp_list)):
    for i in range(len(gcm_list)):
        gcm = class_climate.GCM(name=gcm_list[i], rcp_scenario=rcp_list[j])
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                      rgi_glac_number=['14443'])
        dates_table = modelsetup.datesmodelrun(startyear=1960, endyear=2100, spinupyears=0)
        time = np.linspace(1960, 2100, 141, dtype=int)
        
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
        
        gcm_temp_annual = pygemfxns.annual_avg_2darray(gcm_temp)
        gcm_prec_annual = pygemfxns.annual_sum_2darray(gcm_prec)
        
        x_values = time
        y_values = gcm_temp_annual[0]
        y2_values = gcm_prec_annual[0]
        
        ax[0,j].plot(x_values, y_values, linewidth=1, zorder=2, label=gcm_list[i])
        ax[1,j].plot(x_values, y2_values, linewidth=1, zorder=2, label=gcm_list[i])
        ax[0,j].xaxis.set_major_locator(plt.MultipleLocator(40))
        ax[1,j].xaxis.set_major_locator(plt.MultipleLocator(5))
        ax[0,j].xaxis.set_minor_locator(plt.MultipleLocator(40))
        ax[1,j].xaxis.set_minor_locator(plt.MultipleLocator(5))
        ax[0,j].tick_params(axis='both', which='major', labelsize=16, direction='inout')
        ax[0,j].text(0.5, 0.99, RCP_list[j], size=20, fontweight='extra bold', horizontalalignment='center', verticalalignment='top', 
                 transform=ax[0,0].transAxes)

    
    ax[0,0].set_ylim(3,14.5)
    ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(1))
    ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    ax[1,0].set_ylim(0,2)
    ax[1,0].yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax[1,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    ax[0,0].text(0.5, 0.99, 'Temperature [˚C]', size=20, fontweight='bold', horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)
    ax[1,0].text(0.5, 0.99, 'Precipitation', size=20, fontweight='bold', horizontalalignment='center', verticalalignment='top', 
             transform=ax[1,0].transAxes)
    
#    ax[1,0].legend(loc=(0.01, 0.5), fontsize=10, labelspacing=0.05, handlelength=1, handletextpad=0.25, borderpad=0, 
#               frameon=False)

# save figure
fig.set_size_inches(12, 8)
figure_fp = os.getcwd() + '/../Output/plots/temp_precip/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'temp_precip_allrcps.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)

plt.clf()
