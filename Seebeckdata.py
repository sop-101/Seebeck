# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:27:56 2020

@author: soph_
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

os.chdir(r"\Users\soph_\Documents\phs\Anthra-Au\141220_Anthracene_Zn\12345678\141220_Anthracene_Zn\Anthra1")

read_file = pd.read_csv (r'011input1.xyz', delimiter='\t')
read_file.to_csv (r'011input1.csv', index=None)

read_file = pd.read_csv (r'011input2.xyz', delimiter='\t')
read_file.to_csv (r'011input2.csv', index=None)


ainput1 = pd.read_csv('011input1.csv', header=None)
ainput2 = pd.read_csv('011input2.csv', header=None)
#binput1 = pd.read_csv('006input1.csv', header=None)
#binput2 = pd.read_csv('006input2.csv', header=None)

input1 = np.array(ainput1)
#b1 = np.array(binput1)
#input1 = np.concatenate((a1, b1))
voltage = input1.T[2]


input2 = np.array(ainput2)
#b2 = np.array(binput2)
#input2 = np.concatenate((a2, b2))
seebeck = input2.T[2]

data = np.column_stack((voltage,seebeck))

df = pd.DataFrame(data, columns = ['Input', 'Output'])
df['Output'] = (df['Output'])* 1e6/200
df.Input = df.Input.astype(float).round(1)

voltages = {
        'v0': -1.5,
        'v1': 0.0,
        'v2': 0.6,
        'v3': 1.5,
        }

   
df_filter = df.groupby('Input').filter(lambda x: (x['Input'] == voltages['v0']).all() | (x['Input'] == voltages['v1']).all() | (x['Input'] == voltages['v2']).all() | (x['Input'] == voltages['v3']).all())
#
df2_large = df_filter.pivot(columns = 'Input', values='Output')
#df2 = df2_large.apply(lambda x: pd.Series(x.dropna().values))
df2 = df2_large.apply(lambda x: pd.Series(x[x != 0].dropna().values))

#df3 = df2[(df['Output'] != 0)]

pd.DataFrame(df2).to_csv("011.csv")

mean_std = []

for volt in voltages:
   mean = df2[voltages[volt]].mean()
   std = df2[voltages[volt]].std()
   col =  voltages[volt], mean, std
   mean_std.append(col)

linear_val = pd.DataFrame(mean_std, columns = ['volts', 'mean', 'std'])

sys.exit()


fig, ax = plt.subplots(figsize=(4,4))
#sns.scatterplot(x='volts', y='mean', ci='sd', ax=ax, data=linear_val)

#for x in 


sns.distplot(df2[-1.0].dropna(), kde=False, bins=5, label='-1.0', vertical=True, ax=ax)
sns.distplot(df2[0.0].dropna(), kde=False, bins=5, label='0.0', vertical=True, ax=ax)
sns.distplot(df2[1.1].dropna(), kde=False, bins=5, label='1.1', vertical=True, ax=ax)
sns.distplot(df2[1.6].dropna(), kde=False, bins=5, label='1.6', vertical=True, ax=ax)
plt.xticks(df2.columns)
plt.show()





#plotting hist---------------
number_of_bins = 20
hist_range = (np.min(df2), np.max(df2))
binned_df2 = [
        np.histogram(d, range=hist_range, bins=number_of_bins)[0]
        for d in df2
        ]

binned_maximums = np.max(binned_df2, axis=0)
x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)

ax.set_xticks(x_locations)
ax.set_xticklabels(df2.columns)

ax.set_ylabel("Data values")
ax.set_xlabel("Data sets")

plt.show()
