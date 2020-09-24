# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:16:12 2020

@author: Sanjir
"""

import pandas as pd
import os
import re
import numpy as np


df= pd.read_csv(r"C:\Users\Sanjir\Desktop\DataIku\Minerva\kaggle\dataset.csv")

X = np.float64(df.iloc[:,0:-1])
print(X)

Y = np.float64(df.iloc[:,9:10])
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.35, random_state=0)

df_train,df_test = train_test_split(df, test_size=0.35, random_state=0)

df_train.to_csv("training_dataset.csv",index=False)
df_test.to_csv("testing_dataset.csv",index=False)