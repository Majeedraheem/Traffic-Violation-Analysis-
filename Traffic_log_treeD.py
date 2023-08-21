#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:12:29 2019

@author: mac
"""

import os
import pandas as pd
import numpy  as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold
os.getcwd()
os.chdir(r'C:\Users\kmamgain\Desktop\majeed') 
Uncleaned_tvdf=pd.read_csv('Traffic_Violation_Log.csv')

#Null values grahp
tv=Uncleaned_tvdf.dropna()
#take Sample from 1 million rows Randomly
 #dataset has Zero null values
tv.isnull().sum().sum()
tv.isnull().any()
#Drop columns unncessary :
tvdf=tv.drop(['Geolocation','Latitude','Longitude','Agency','VehicleType','Work Zone','Driver City','HAZMAT','Fatal','Work Zone','Article','Year','Date Of Stop','Time Of Stop','Accident','Longitude','Location','Description','SubAgency'],axis=1)

#Rename Columns Removing Space:
df=tvdf.rename(columns={'Violation Type':'ViolationType','Driver City':'DriverCity','Arrest Type':'ArrestType','Contributed To Accident':'ContributedToAccident','Commercial License':'CommercialLicense','Commercial Vehicle':'CommercialVehicle','PersonalInjury':'PersonalInjury'})

tv.isnull().any()
dftv=df.sample(frac=0.010)
dftv.info()
###save the data for 8671 rows in new csv file
dftv.to_csv('Traffic_Violation_Log.csv', index=False)
#Check null values 
tvdf.isnull().sum().sum()
tvdf['ViolationType'].values().count()
tvdf['ArrestType'].value_counts()

dftv['ArrestType'].value_counts().plot(kind='bar',title='Violation Distribution')

dftv.groupby((['Gender','ArrestType'])).count()
dftv.groupby(['Gender','ArrestType']).size().unstack().plot(kind='bar',stacked=True)

#sample in data set is 8691 
print('The number of samples into the test data is {}.'.format(dftv.shape[0]))
print(' Voilaters grouped by port of embarkation (A = Marked Patrol, S = License Plate Recognition, Q = Marked Laser, M - Marked (Off-Duty),B - Unmarked Patrol, I = Marked Moving Radar (Moving), E - Marked Stationary Radar G - Marked Moving Radar (Stationary),O - Foot Patrol,L - Motorcycle):')
print(tvdf['ArretsType'].value_counts())
dftv.ArrestType.value_counts().plot(kind='bar')
sns.countplot(x='ArrestType', data=dftv, palette='Set2')
plt.show()



#modeling: 
#droping dependent variable x data without actual respone value
# dropping response ArrestType column
x=dftv.drop(['ArrestType'],axis=1)
y=dftv.ArrestType

#Scaling 
sc = StandardScaler()
dftv_scaled = pd.DataFrame(sc.fit_transform(x), columns=x.columns)

#Create train set with 70%, test set with 30%
x_train, x_test, y_train, y_test = train_test_split(dftv_scaled, y, test_size = 0.3, random_state = 0)

#import classifier
model = LogisticRegression()

#train model
model.fit(x_train, y_train)


#predict on test set
y_pred = model.predict(x_test)

accuracy_score(y_test, y_pred)

conm = confusion_matrix (y_test, y_pred)

TN = conm[0][0]
FP = conm[0][1]
FN = conm[1][0]
TP = conm[1][1]

print(classification_report(y_test,y_pred))
#should we use none, micro, macro or weighted for average for below multiclass
sensitivity1 = recall_score(y_test, y_pred, average='micro')
sensitivity2 = recall_score(y_test, y_pred, average='macro')

#sensitivity = TP/(TP+FN)
precision1 = precision_score(y_test, y_pred, average='micro')
precision2 = precision_score(y_test, y_pred, average='macro')

#precision = TP/(TP+FP)

specificity = TN/(TN+FP)

FPR = FP/(TN+FP)

print(sensitivity1)
print(sensitivity2)
print(precision1)
print(precision2)
print(specificity)
print(FPR)

probabilities = model.predict_proba(x_test)
y_pred_prob = model.predict_proba(x_test)[:,1]
from sklearn.preprocessing import binarize
new_y_pred = binarize(y_pred_prob.reshape(442,1),0.7)

confusion_matrix(y_test, new_y_pred)


# proba both p or 1-p

#K-FOLD

from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
cross_val_score(clf,dftv_scaled,y,cv=4).mean()









'''def summerize_data(df):
    for column in dftv.columns:
        print(column)
        if dftv.dtypes[column] == np.object: # Categorical data
            print (dftv[column].value_counts())
        else:
            print (dftv[column].describe()) 
            
        print ('\n')
    
summerize_data(dftv)'''












#cehck Corrleation Between Variables 
plt.figure(figsize=(40,40))
sns.heatmap(s.corr(),linewidths=0.5,linecolor='black',square=True,cmap='summer')
s.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)#indicate no relationship


#Apply Chisqure for corrleation amoung features 
s=s.apply(lambda x : pd.factorize(x)[0])+1
pd.DataFrame([chisquare(s[x].values,f_exp=s.values.T,axis=1)[0] for x in s])

#how many values our target have 
df['ViolationType'].unique
df.isnull().sum().sum()
df.columns

#####Tree Clasification :

""" TREE CLASSIFICATION"""

X_tree=tvdf.copy()
y_tree=X_tree['ArrestType']
X_tree.drop(['ArrestType'], axis=1, inplace= True)

X_tree.columns

X_tree1 = pd.get_dummies(X_tree,columns=['SubAgency', 'Description', 'Location', 'Belts', 'Personal Injury',
       'Property Damage', 'CommercialLicense', 'CommercialVehicle', 'Alcohol',
       'State', 'Make', 'Model', 'Color', 'ViolationType', 'Charge',
       'ContributedToAccident', 'Race', 'Gender', 'Driver State', 'DL State'],drop_first=True)

    
DataFramey_tree.value.count()    
    
from sklearn.model_selection import train_test_split
Xb_train, Xb_test, yb_train, yb_test = train_test_split (X_tree1, y_tree , test_size = 1/3, 
                                                         random_state=0)


# step 4 import classifier

from sklearn.tree import DecisionTreeClassifier
model_traffic = DecisionTreeClassifier()


# step 5 train the model

model_traffic.fit(Xb_train, yb_train)


# Step 6 predict on test set
yb_pred = model_traffic.predict(Xb_test)



# Step 7 calculate the accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(yb_test, yb_pred)


# Step 8 look into the confusion matrix / contingency table

cm_traffic = confusion_matrix(yb_test, yb_pred)


accuracy_score(yb_test, yb_pred)
confusion_matrix(yb_test, yb_pred)
print(classification_report(yb_test,yb_pred))















