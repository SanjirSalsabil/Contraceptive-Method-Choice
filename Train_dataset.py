# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:41:16 2020

@author: Sanjir
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


attributes = ["Wife's age", "Wife's education", "Husband's education", 
              "Number of children", "Wife's religion", "Wife is working",
              "Husband's occupation", "SLI", "Media exposure", "CMC"]

df= pd.read_csv(r"C:\Users\Sanjir\Desktop\DataIku\Minerva\kaggle\training_dataset.csv", names=attributes)

#Describe_data
df.head()
df.info()
df.columns
df.shape
df.describe()

# data vizualizations
plt.hist(df["Wife's age"])
plt.hist(df["Wife is working"])
plt.hist(df["SLI"])
plt.hist(df["Husband's education"])
plt.hist(df["Wife's education"])

#indexing_&_Selecting_Data
X = np.float64(df.iloc[:,0:-1])
Y = np.float64(df.iloc[:,9:10])

#drop nan
df=df.dropna()

# feature scaling

X_sc = StandardScaler()
X[:,0:1] = X_sc.fit_transform(X[:,0:1])
X[:,3:4] = X_sc.fit_transform(X[:,3:4])


X_train, , Y_train = (X, Y, train_size=0.35, random_state=0)


#Build_model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, Y_train)

model.predict(X_train)

model.score(X_train, Y_train)

from sklearn.externals import joblib
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

#Build_another_Model
from sklearn.svm import SVC
model_svm = SVC(kernel = "rbf", gamma = 1.5)
model_svm.fit(X_train, Y_train)

# prediction
Y_pred = model_svm.predict(X_train)
print(Y_pred)

# evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(Y_train, Y_pred)
acc = accuracy_score(Y_train, Y_pred)
F1 = f1_score(Y_train, Y_pred, average="micro") 

from sklearn.metrics import classification_report
print(classification_report(Y_train, Y_pred))

from sklearn.model_selection import cross_val_score
k_fold_acc = cross_val_score(model_svm, X_train, Y_train, cv=10)
K_fold_mean = k_fold_acc.mean()

from sklearn.externals import joblib
# save the model to disk
filename = 'SVM_model.sav'
joblib.dump(model, filename)
