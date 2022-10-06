# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:11:01 2021

@author: soph_
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

os.chdir(r"\Users\soph_\Documents\phs\Ferrocene\2\261121")

read_file = pd.read_csv (r'026input1.xyz', delimiter='\t')
read_file.to_csv (r'026input1.csv', index=None)

read_file = pd.read_csv (r'026input2.xyz', delimiter='\t')
read_file.to_csv (r'026input2.csv', index=None)


ainput1 = pd.read_csv('026input1.csv', header=None)
ainput2 = pd.read_csv('026input2.csv', header=None)

input1 = np.array(ainput1)
voltage = input1.T[2]

input2 = np.array(ainput2)
seebeck = input2.T[2]


data = np.column_stack((voltage,seebeck))

df = pd.DataFrame(data, columns = ['Input', 'Output'])
df['Output'] = (df['Output'])* 1e6/200
df.Input = df.Input.astype(float).round(2)
df['Input'] = ((df['Input'])*1000-500)/10
df.Input = df.Input.astype(float).round(0)

voltages = {
        'v0': 21,
        'v1': 25,
        'v2': 27,
        'v3': 28,
        'v4': 34,
#        'v5': 29,
#        'v6': 34,
        }


df_filter = df.groupby('Input').filter(lambda x: (x['Input'] == voltages['v0']).all() | (x['Input'] == voltages['v1']).all() | (x['Input'] == voltages['v2']).all() | (x['Input'] == voltages['v3']).all() | (x['Input'] == voltages['v4']).all())# |(x['Input'] == voltages['v5']).all()|(x['Input'] == voltages['v6']).all())
df2_large = df_filter.pivot(columns = 'Input', values='Output')

df2 = df2_large.apply(lambda x: pd.Series(x[x != 0].dropna().values))

#df2_large = df.pivot(columns = 'Input', values='Output')
#sys.exit()
pd.DataFrame(df2).to_csv("026.csv")
#df2.count() counts the elements in each column


mean_std = []

for volt in voltages:
   mean = df2[voltages[volt]].mean()
   std = df2[voltages[volt]].std()
   col =  voltages[volt], mean, std
   mean_std.append(col)

linear_val = pd.DataFrame(mean_std, columns = ['volts', 'mean', 'std'])