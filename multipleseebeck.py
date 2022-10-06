# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 13:17:22 2022

@author: soph_
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# Print the current working directory
#print("Current working directory: {0}".format(os.getcwd()))


# Change the current working directory
os.chdir(r"\Users\soph_\Documents\phs\Ferrocene\2\180122_2")

#State the path for the next line
path = os.getcwd()


# Print the current working directory
#print("Current working directory: {0}".format(os.getcwd()))

# Lists everything in this specific dir
#paths = os.listdir('seebeck/')



# Get a xyz files from specific directory
files = glob.glob(path + '/*.xyz')
#csv_files = glob.glob(path + '/*.csv')

##For loop to read all xyz files
#for file in files:
#    read_xyz = pd.read_csv(file, delimiter=',')
#    read_xyz.to_csv(file + '.csv', index=None)
    
#csv_files = glob.glob(path + '/*.csv')


#Merge all input1 (voltage) files together

voltage = []    
for x in files:
    if 'ainput1' in x:
        df_volt = pd.read_csv(x, index_col=None, header=None, sep=',|\t', engine='python')
        voltage.append(df_volt)
        
input1 = pd.concat(voltage, axis=0, ignore_index=True)

#pd.DataFrame(input1).to_csv("ainput1.csv")

seebeck = []    
for y in files:
    if 'ainput2' in y:
        df2_see = pd.read_csv(y, index_col=None, header=None, sep=',|\t', engine='python')
        seebeck.append(df2_see)
        
input2 = pd.concat(seebeck, axis=0, ignore_index=True)

#pd.DataFrame(input2).to_csv("ainput2.csv")


#-----------

#taking the z column of xyz files
input1_arr = np.array(input1)
volts = input1_arr.T[2]

input2_arr = np.array(input2)
nanos = input2_arr.T[2]

#stacking two z columns together from input1 and input2
data = np.column_stack((volts,nanos))

df = pd.DataFrame(data, columns = ['Input', 'Output'])
df['Output'] = (df['Output'])* 1e6/200
df.Input = df.Input.astype(float).round(2)
df['Input'] = ((df['Input'])*1000-500)/10
df.Input = df.Input.astype(float).round(0)

#sys.exit()
#df['Input'].unique() shows the different values of input1

#create a dictionary? define unique values etc etc

voltages = {
        'v0': 18,
        'v1': 17,
        'v2': 21,
        'v3': 22,
        'v4': 25,
        'v5': 26,
        'v6': 32,
        }


df_filter = df.groupby('Input').filter(lambda x: (x['Input'] == voltages['v0']).all() | (x['Input'] == voltages['v1']).all() | (x['Input'] == voltages['v2']).all() | (x['Input'] == voltages['v3']).all() | (x['Input'] == voltages['v4']).all() |(x['Input'] == voltages['v5']).all()|(x['Input'] == voltages['v6']).all())
df2_large = df_filter.pivot(columns = 'Input', values='Output')

df2 = df2_large.apply(lambda x: pd.Series(x[x != 0].dropna().values))


mean_std = []

for volt in voltages:
   mean = df2[voltages[volt]].mean()
   std = df2[voltages[volt]].std()
   col =  voltages[volt], mean, std
   mean_std.append(col)

linear_val = pd.DataFrame(mean_std, columns = ['volts', 'mean', 'std'])

pd.DataFrame(df2).to_csv("a_data.csv")

