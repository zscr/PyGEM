#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:36:48 2019

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
def avg_parameters_20yr(component, area):
    """
    sum up annually and convert to km^3, average every 20 yrs, then convert to m w.e. and average every 20 yrs again
    """
    component_annual = ((pygemfxns.annual_sum_2darray(component))/1000)*area
    component_annual_regional_1 = np.sum(component_annual[:,1:], axis=0) # in km^3 aka gigatons
    component_avg_gt_20yr = (np.add.reduceat(component_annual_regional_1, np.arange(0, len(component_annual_regional_1), 20)))/20
    component_annual_regional_2 = ((component_annual_regional_1)/(np.sum(area[:,1:], axis=0)))*1000 # km^3 (gigatons) --> m w.e.
    component_avg_mwea_20yr = (np.add.reduceat(component_annual_regional_2, np.arange(0, len(component_annual_regional_2), 20)))/20
    
    return component_avg_gt_20yr, component_avg_mwea_20yr

#%%
label_list = ['All glaciers', 'Tidewater', 'Non-Tidewater']
sim_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
            'MPI-ESM-LR', 'NorESM1-M']
#sim_list = ['CanESM2']
#rcp_list = ['26', '45', '85']
RCP_list = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
color_list = ['b', 'm', 'r', 'k']
rcp_list = ['26','45','85']

# tidewater glacier indexes from larsen
all_glac_idxs = np.linspace(0,115,116)
tidewater_idxs = [13, 26, 36, 46, 52, 54, 55, 70, 71, 72, 73, 87, 90, 92, 93, 96, 109, 113]
non_tidewater_idxs = np.delete(all_glac_idxs, tidewater_idxs)

acc_avg_gt, acc_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
acc_tide_avg_gt, acc_tide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
acc_nontide_avg_gt, acc_nontide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)

ref_avg_gt, ref_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
ref_tide_avg_gt, ref_tide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
ref_nontide_avg_gt, ref_nontide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)

mlt_avg_gt, mlt_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
mlt_tide_avg_gt, mlt_tide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
mlt_nontide_avg_gt, mlt_nontide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)

abl_avg_gt, abl_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
abl_tide_avg_gt, abl_tide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)
abl_nontide_avg_gt, abl_nontide_avg_mwea = np.zeros((10,6), dtype=float), np.zeros((10,6), dtype=float)

for j in range(len(rcp_list)):
    fig, axes = plt.subplots(2, 3, squeeze=False, sharex='col', sharey='row', 
                 gridspec_kw = {'wspace':0.1, 'hspace':0.05})
    for i in range(len(sim_list)):
        ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/' + sim_list[i] + '/R1_' + sim_list[i] + '_rcp' + rcp_list[j] + '_c1_ba1_1sets_1980_2100.nc')
        time = ds.variables['year'].values[:]
        area = ds.variables['area_glac_annual'].values[:,:-1,0]
        acc = ds.variables['acc_glac_monthly'].values[:,:,0]
        ref = ds.variables['refreeze_glac_monthly'].values[:,:,0]
        mlt = ds.variables['melt_glac_monthly'].values[:,:,0]
        abl = ds.variables['frontalablation_glac_monthly'].values[:,:,0]
        
        acc_tide = acc[tidewater_idxs, :]
        acc_nontide = acc[non_tidewater_idxs.astype(int), :]
        
        ref_tide = ref[tidewater_idxs, :]
        ref_nontide = ref[non_tidewater_idxs.astype(int), :]
        
        mlt_tide = mlt[tidewater_idxs, :]
        mlt_nontide = mlt[non_tidewater_idxs.astype(int), :]
        
        abl_tide = abl[tidewater_idxs, :]
        abl_nontide = abl[non_tidewater_idxs.astype(int), :]
        
        area_tide = area[tidewater_idxs, :]
        area_nontide = area[non_tidewater_idxs.astype(int), :]
        
        acc_avg_gt[i], acc_avg_mwea[i] = avg_parameters_20yr(acc, area)
        acc_tide_avg_gt[i], acc_tide_avg_mwea[i] = avg_parameters_20yr(acc_tide, area_tide)
        acc_nontide_avg_gt[i], acc_nontide_avg_mwea[i] = avg_parameters_20yr(acc_nontide, area_nontide)
        
        ref_avg_gt[i], ref_avg_mwea[i] = avg_parameters_20yr(ref, area)
        ref_tide_avg_gt[i], ref_tide_avg_mwea[i] = avg_parameters_20yr(ref_tide, area_tide)
        ref_nontide_avg_gt[i], ref_nontide_avg_mwea[i] = avg_parameters_20yr(ref_nontide, area_nontide)
        
        mlt_avg_gt[i], mlt_avg_mwea[i] = avg_parameters_20yr(mlt, area)
        mlt_tide_avg_gt[i], mlt_tide_avg_mwea[i] = avg_parameters_20yr(mlt_tide, area_tide)
        mlt_nontide_avg_gt[i], mlt_nontide_avg_mwea[i] = avg_parameters_20yr(mlt_nontide, area_nontide)
        
        abl_avg_gt[i], abl_avg_mwea[i] = avg_parameters_20yr(abl, area)
        abl_tide_avg_gt[i], abl_tide_avg_mwea[i] = avg_parameters_20yr(abl_tide, area_tide)
        abl_nontide_avg_gt[i], abl_nontide_avg_mwea[i] = avg_parameters_20yr(abl_nontide, area_nontide)
        ds.close()
    
    # averaging parameters
    acc_avg_gt_1d, acc_avg_mwea_1d = np.average(acc_avg_gt, axis=0), np.average(acc_avg_mwea, axis=0)
    acc_tide_avg_gt_1d, acc_tide_avg_mwea_1d = np.average(acc_tide_avg_gt, axis=0), np.average(acc_tide_avg_mwea, axis=0)
    acc_nontide_avg_gt_1d, acc_nontide_avg_mwea_1d = np.average(acc_nontide_avg_gt, axis=0), np.average(acc_nontide_avg_mwea, axis=0)
    
    ref_avg_gt_1d, ref_avg_mwea_1d = np.average(ref_avg_gt, axis=0), np.average(ref_avg_mwea, axis=0)
    ref_tide_avg_gt_1d, ref_tide_avg_mwea_1d = np.average(ref_tide_avg_gt, axis=0), np.average(ref_tide_avg_mwea, axis=0)
    ref_nontide_avg_gt_1d, ref_nontide_avg_mwea_1d = np.average(ref_nontide_avg_gt, axis=0), np.average(ref_nontide_avg_mwea, axis=0)
    
    mlt_avg_gt_1d, mlt_avg_mwea_1d = np.average(mlt_avg_gt, axis=0), np.average(mlt_avg_mwea, axis=0)
    mlt_tide_avg_gt_1d, mlt_tide_avg_mwea_1d = np.average(mlt_tide_avg_gt, axis=0), np.average(mlt_tide_avg_mwea, axis=0)
    mlt_nontide_avg_gt_1d, mlt_nontide_avg_mwea_1d = np.average(mlt_nontide_avg_gt, axis=0), np.average(mlt_nontide_avg_mwea, axis=0)
    
    abl_avg_gt_1d, abl_avg_mwea_1d = np.average(abl_avg_gt, axis=0), np.average(abl_avg_mwea, axis=0)
    abl_tide_avg_gt_1d, abl_tide_avg_mwea_1d = np.average(abl_tide_avg_gt, axis=0), np.average(abl_tide_avg_mwea, axis=0)
    abl_nontide_avg_gt_1d, abl_nontide_avg_mwea_1d = np.average(abl_nontide_avg_gt, axis=0), np.average(abl_nontide_avg_mwea, axis=0)
    
    # plotting
    
    # gigatons
    x_values = time[0:120:20]
    y_values = acc_avg_gt_1d
    y2_values = ref_avg_gt_1d
    y3_values = np.negative(mlt_avg_gt_1d)
    y4_values = np.negative(abl_avg_gt_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[0,0].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[0,0].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[0,0].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[0,0].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[0,0].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    axes[0,0].set_title(label=label_list[0], size=10, horizontalalignment='center', verticalalignment='baseline')

    y_values = acc_tide_avg_gt_1d
    y2_values = ref_tide_avg_gt_1d
    y3_values = np.negative(mlt_tide_avg_gt_1d)
    y4_values = np.negative(abl_tide_avg_gt_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[0,1].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[0,1].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[0,1].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[0,1].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[0,1].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    axes[0,1].set_title(label=label_list[1], size=10, horizontalalignment='center', verticalalignment='baseline')
    
    y_values = acc_nontide_avg_gt_1d
    y2_values = ref_nontide_avg_gt_1d
    y3_values = np.negative(mlt_nontide_avg_gt_1d)
    y4_values = np.negative(abl_nontide_avg_gt_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[0,2].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[0,2].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[0,2].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[0,2].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[0,2].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    axes[0,2].set_title(label=label_list[2], size=10, horizontalalignment='center', verticalalignment='baseline')
    
    y_values = acc_avg_mwea_1d
    y2_values = ref_avg_mwea_1d
    y3_values = np.negative(mlt_avg_mwea_1d)
    y4_values = np.negative(abl_avg_mwea_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[1,0].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[1,0].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[1,0].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[1,0].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[1,0].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    
    y_values = acc_tide_avg_mwea_1d
    y2_values = ref_tide_avg_mwea_1d
    y3_values = np.negative(mlt_tide_avg_mwea_1d)
    y4_values = np.negative(abl_tide_avg_mwea_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[1,1].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[1,1].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[1,1].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[1,1].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[1,1].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    
    y_values = acc_nontide_avg_mwea_1d
    y2_values = ref_nontide_avg_mwea_1d
    y3_values = np.negative(mlt_nontide_avg_mwea_1d)
    y4_values = np.negative(abl_nontide_avg_mwea_1d)
    net = y_values + y2_values + y3_values +y4_values
    
    axes[1,2].bar(x_values, y_values, color='b', width=10, zorder=2)
    axes[1,2].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
    axes[1,2].bar(x_values, y3_values, color='r', width=10, zorder=2)
    axes[1,2].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
    axes[1,2].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
    
    # figure styling
    
    # X-axis
    for i in range(3):
        axes[1,i].set_xlabel('Year', size=12)
        axes[1,i].xaxis.set_major_locator(plt.MultipleLocator(40))
        axes[1,i].xaxis.set_minor_locator(plt.MultipleLocator(10))
        axes[1,i].tick_params(axis='x', rotation=70)
    #ax[0,0].set_xlim(0, 1.1)
    #ax[0,0].xaxis.set_tick_params(labelsize=12)
    #ax[0,0].set_xticklabels(['2015','2050','2100'])       
     
    # Y-label
    axes[0,0].set_ylabel('Mass balance [Gt yr^-1]', size=10)
    axes[1,0].set_ylabel('Mass balance [m w.e. yr^-1]', size=10)
    axes[0,0].set_ylim(-110, 45)
    axes[1,0].set_ylim(-4.5, 1.75)
    
    # tick marks for m w.e.    
    axes[1,0].yaxis.set_major_locator(plt.MultipleLocator(2))
    axes[1,0].yaxis.set_minor_locator(plt.MultipleLocator(0.5))    
    
    # tick marks for Gt
    axes[0,0].yaxis.set_major_locator(plt.MultipleLocator(40))
    axes[0,0].yaxis.set_minor_locator(plt.MultipleLocator(10)) 
        
    # save figure
    fig.set_size_inches(5, 8)
    figure_fp = os.getcwd() + '/../Output/plots/gcm_compare_parameters/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    figure_fn = 'massbalparams_partitioned_plot_' + rcp_list[j] + '.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)







