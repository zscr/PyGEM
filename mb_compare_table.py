#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:26:51 2019

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
names_l = ds_larsen.loc[:,'RGIId']
mb_m = ds_mcnabb.loc[:,'mb_mwea']
mb_l = ds_larsen.loc[:,'mb_mwea']

i = 0
j = 0

while i < len(names_m):
    while j < len(names_l):
        if names_m[i] != names_l[j]:
            names_m.pop(i)
            names_l.pop(j)
            mb_m.pop(i)
            mb_l.pop(j)
        else:
            i = i + 1
            j = j + 1

compare = [names_m, mb_m, mb_l]
compare