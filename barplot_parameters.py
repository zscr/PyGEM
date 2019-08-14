#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:56:33 2019

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

#%% X-Y PLOT
sim_list = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
            'MPI-ESM-LR', 'NorESM1-M']
rcp_list = ['26', '45', '85']
RCP_list = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
color_list = ['b', 'm', 'r', 'k']


for j in range(len(rcp_list)):  
    fig, axes = plt.subplots(2, 10, squeeze=False, sharex='col', sharey='row', 
                 gridspec_kw = {'wspace':0.1, 'hspace':0.05})
    for i  in range(len(sim_list)):
        ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/' + sim_list[i] + '/R1_' + sim_list[i] + '_rcp' + 
                             rcp_list[j] + '_c1_ba1_1sets_1980_2100.nc')
        time = ds.variables['year'].values[:]
        area = ds.variables['area_glac_annual'].values[:,:-1,0]
        # positive mass balance, summed annually
        acc = ds.variables['acc_glac_monthly'].values[:,:,0]
#        acc_annual = ((pygemfxns.annual_sum_2darray(acc))/1000)*area # convert m w.e. to km w.e.
        ref = ds.variables['refreeze_glac_monthly'].values[:,:,0]
#        refreeze_annual = ((pygemfxns.annual_sum_2darray(refreeze))/1000)*area # convert m w.e. to km w.e.
        mlt = ds.variables['melt_glac_monthly'].values[:,:,0]
#        melt_annual = ((pygemfxns.annual_sum_2darray(melt))/1000)*area # convert m w.e. to km w.e.
        abl = ds.variables['frontalablation_glac_monthly'].values[:,:,0]
#        ablation_annual = ((pygemfxns.annual_sum_2darray(melt))/1000)*area # convert m w.e. to km w.e.
#        
        acc_regional_avg_gt_20yr, acc_regional_avg_mwea_20yr = avg_parameters_20yr(acc, area)
        ref_regional_avg_gt_20yr, ref_regional_avg_mwea_20yr = avg_parameters_20yr(ref, area)
        mlt_regional_avg_gt_20yr, mlt_regional_avg_mwea_20yr = avg_parameters_20yr(mlt, area)
        abl_regional_avg_gt_20yr, abl_regional_avg_mwea_20yr = avg_parameters_20yr(abl, area)


#        # in gigatons (aka km^3)
#        refreeze_regional_annual = np.sum(refreeze_annual, axis=0)
#        acc_regional_annual = np.sum(acc_annual, axis=0)
#        melt_regional_annual = np.sum(melt_annual, axis=0)
#        ablation_regional_annual = np.sum(ablation_annual, axis=0)
#        
#        # averaged every 20 years
#        acc_regional_20yr = np.add.reduceat(acc_regional_annual[:-1], np.arange(0, len(acc_regional_annual)-1, 20))
#        acc_regional_20yr = (acc_regional_20yr)/20
#        refreeze_regional_20yr = np.add.reduceat(refreeze_regional_annual[:-1], np.arange(0, len(refreeze_regional_annual)-1, 20))
#        refreeze_regional_20yr = (refreeze_regional_20yr)/20
#        melt_regional_20yr = np.add.reduceat(melt_regional_annual[:-1], np.arange(0, len(melt_regional_annual)-1, 20))
#        melt_regional_20yr = (melt_regional_20yr)/20
#        ablation_regional_20yr = np.add.reduceat(ablation_regional_annual[:-1], np.arange(0, len(ablation_regional_annual)-1, 20))
#        ablation_regional_20yr = (ablation_regional_20yr)/20

        # X and Y values
        x_values = time[0:120:20]
        y_values = acc_regional_avg_gt_20yr
        y2_values = ref_regional_avg_gt_20yr
        y3_values = np.negative(mlt_regional_avg_gt_20yr)
        y4_values = np.negative(abl_regional_avg_gt_20yr)
        net = y_values + y2_values + y3_values + y4_values
        
#        # prep for averaging gcms
#        acc_gcm_avg_gt[i] = acc_regional_20yr
#        ref_gcm_avg_gt[i] = refreeze_regional_20yr
#        mlt_gcm_avg_gt[i] = melt_regional_20yr
#        abl_gcm_avg_gt[i] = ablation_regional_20yr
        
        # set up plot
        axes[0,i].bar(x_values, y_values, color='b', width=10, zorder=2)
        axes[0,i].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
        axes[0,i].bar(x_values, y3_values, color='r', width=10, zorder=2)
        axes[0,i].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
        axes[0,i].scatter(x_values, net, color='w', marker='_', s=16, zorder=5)
        axes[0,i].set_title(label=sim_list[i], size=10, horizontalalignment='center', verticalalignment='baseline')

                                                        ## m w.e. plots ##
#        # back to m w.e. after dividing by area
#        acc_regional_annual = ((np.sum(acc_annual, axis=0))/(np.sum(area, axis=0)))*1000
#        refreeze_regional_annual = ((np.sum(refreeze_annual, axis=0))/(np.sum(area, axis=0)))*1000
#        melt_regional_annual = ((np.sum(melt_annual, axis=0))/(np.sum(area, axis=0)))*1000
#        ablation_regional_annual = ((np.sum(ablation_annual, axis=0))/(np.sum(area, axis=0)))*1000
#        
#        # averaged every 20 years
#        acc_regional_20yr = np.add.reduceat(acc_regional_annual[:-1], np.arange(0, len(acc_regional_annual)-1, 20))
#        acc_regional_20yr = (acc_regional_20yr)/20
#        refreeze_regional_20yr = np.add.reduceat(refreeze_regional_annual[:-1], np.arange(0, len(refreeze_regional_annual)-1, 20))
#        refreeze_regional_20yr = (refreeze_regional_20yr)/20
#        melt_regional_20yr = np.add.reduceat(melt_regional_annual[:-1], np.arange(0, len(melt_regional_annual)-1, 20))
#        melt_regional_20yr = (melt_regional_20yr)/20
#        ablation_regional_20yr = np.add.reduceat(ablation_regional_annual[:-1], np.arange(0, len(ablation_regional_annual)-1, 20))
#        ablation_regional_20yr = (ablation_regional_20yr)/20

        # X and Y values
        x_values = time[0:120:20]
        y_values = acc_regional_avg_mwea_20yr
        y2_values = ref_regional_avg_mwea_20yr
        y3_values = np.negative(mlt_regional_avg_mwea_20yr)
        y4_values = np.negative(abl_regional_avg_mwea_20yr)
        net = y_values + y2_values + y3_values + y4_values

        
#        # prep for averaging gcms
#        acc_gcm_avg_mwea[i] = acc_regional_20yr
#        ref_gcm_avg_mwea[i] = refreeze_regional_20yr
#        mlt_gcm_avg_mwea[i] = melt_regional_20yr
#        abl_gcm_avg_mwea[i] = ablation_regional_20yr
        
        # set up plot
        axes[1,i].bar(x_values, y_values, color='b', width=10, zorder=2)
        axes[1,i].bar(x_values, y2_values, bottom=y_values, color='m', width=10, zorder=3)
        axes[1,i].bar(x_values, y3_values, color='r', width=10, zorder=2)
        axes[1,i].bar(x_values, y4_values, bottom=y3_values, color='k', width=10, zorder=3)
        axes[1,i].scatter(x_values, net, color='w', marker='_', s=16, zorder=4)

#    # averaging gcms
#    y_values = np.average(acc_gcm_avg_mwea, axis=0)
#    y2_values = np.average(ref_gcm_avg_mwea, axis=0)
#    y3_values = np.average(mlt_gcm_avg_mwea, axis=0)
#    y4_values = np.average(abl_gcm_avg_mwea, axis=0)
            
#        axes[1,i].set_title(label=sim_list[i], size=10, horizontalalignment='center', verticalalignment='baseline')

    ds.close()

    # figure styling
#    plt.subplots_adjust(bottom=0.1, right=0.4, top=0.4)

    # X-label
    for i in range(10):
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
    
        
    # Legend
#    leg_lines = []
#    labels = ['Refreeze', 'Accumulation', 'Melt', 'Frontal Ablation']
#    label_colors = ['m', 'b', 'r', 'k']
#    for nlabel, label in enumerate(labels):
#        line = Line2D([0,0.5],[0,0.5], color=label_colors[nlabel], linewidth=2)
#        leg_lines.append(line)
#    axes[1,0].set_zorder(3)
#    axes[1,0].legend(leg_lines, labels, loc=(0.05,0.05), fontsize=12, 
#      labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    
    axes[1,0].text(1975, -4.2, RCP_list[j], fontsize=16, fontweight='extra bold')

    # save figure
    fig.set_size_inches(15, 6)
    figure_fp = os.getcwd() + '/../Output/plots/gcm_compare_parameters/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    figure_fn = 'massbalparams_plot_' + rcp_list[j] + '.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    # clear figure window after saving
    plt.clf()

