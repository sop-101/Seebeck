# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:23:22 2021

@author: soph_
"""

import pandas as pd

df = pd.read_csv (r'C:\Users\soph_\Documents\phs\Ferrocene\1\261121\009.csv')

#voltages = {
#        'v0': 34,
#        'v4': 28,
#        'v1': 27,
#        'v2': 24,
#        'v3': 20,
#        }



mean_std = []

#for volt in voltages:
#   mean = df[voltages[volt]].mean()
#   std = df[voltages[volt]].std()
#   col =  voltages[volt], mean, std
#   mean_std.append(col)
#
#linear_val = pd.DataFrame(mean_std, columns = ['volts', 'mean', 'std'])