# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:26:41 2020

@author: Sanjir
"""
import numpy as np
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler

attributes = ["Wife's age", "Wife's education", "Husband's education", 
              "Number of children", "Wife's religion", "Wife is working",
              "Husband's occupation", "SLI", "Media exposure", "CMC"]

df= pd.read_csv(r"C:\Users\Sanjir\Desktop\DataIku\Minerva\kaggle\testing_dataset.csv", names=attributes)

#Describe_data
df.head()
df.info()
df.columns
df.shape
df.describe()


#indexing_&_Selecting_Data

X = np.float64(df.iloc[:,0:-1])
Y = np.float64(df.iloc[:,9:10])

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
print(X)
print(Y)

#drop nan
df=df.dropna()

# feature scaling
sc = StandardScaler()
test_x = sc.fit_transform(df)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)
print(X_test.shape)

print(df.dtypes)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(Y)
print(y)

X_sc = StandardScaler()
X[:,0:1] = X_sc.fit_transform(X[:,0:1])
X[:,3:4] = X_sc.fit_transform(X[:,3:4])

#Test_model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_test, Y_test)

model.predict(X_test)

model.score(X_test, Y_test)


#Build_another_Model
from sklearn.svm import SVC
model_svm = SVC(kernel = "rbf", gamma = 1.5)
model_svm.fit(X_test, Y_test)

# prediction
Y_pred = model_svm.predict(X_test)
print(Y_pred)

# evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_test, Y_pred, average="micro") 

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

from sklearn.model_selection import cross_val_score
k_fold_acc = cross_val_score(model_svm, X_test, Y_test, cv=10)
K_fold_mean = k_fold_acc.mean()


