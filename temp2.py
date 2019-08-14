#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:30:21 2019

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
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})

ds = xr.open_dataset(os.getcwd() + '/../Output/simulations/CanESM2/R1_CanESM2_rcp26_c1_ba1_1sets_1980_2100.nc')
time = ds.variables['year'].values[:]
acc = ds.variables['acc_glac_monthly'].values[:,:,0]
acc_regional = np.sum(acc, axis=0)
acc_regional_annual = pygemfxns.annual_sum_2darray(acc_regional)
refreeze = ds.variables['refreeze_glac_monthly'].values[:,:,0]
refreeze_regional = np.sum(refreeze, axis=0)
refreeze_regional_annual = pygemfxns.annual_sum_2darray(refreeze_regional)
ax[0,0].bar(time, acc_regional, color='k', linewidth=1, zorder=2)
ax[0,0].bar(time, refreeze_regional, bottom=acc_regional, color='b', linewidth=1, zorder=2)
ds.close()