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
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
os.getcwd()
os.chdir('/Users/mac/Desktop/LogsticModel') 
tvdf=pd.read_csv('Traffic_Violation_Log.csv')

tvdf.isnull().sum().sum()
tvdf.isnull().any()
tvdf.info()
###save the data for 8671 rows in new csv file

tvdf['ArrestType'].value_counts()

tvdf['ArrestType'].value_counts().plot(kind='bar',title='Violation Distribution')

tvdf.groupby((['Gender','ArrestType'])).count()
tvdf.groupby(['Gender','ArrestType']).size().unstack().plot(kind='bar',stacked=True)

#sample in data set is 8691 
def Agency_simplifier(val):
    if val == '1st district, Rockville' or val == '2nd district, Bethesda': 
        val = 'North'
        return val
    elif val == '3rd district, Silver Spring' or val == '4th district, Wheaton': 
        val = 'South'
        return val    
    elif val == '6th district, Gaithersburg / Montgomery Village' or val == '5th district, Germantown':
        val = 'West'
        return val
    else:
        return val

tvdf['SubAgency']=tvdf.SubAgency.apply(Agency_simplifier)

tvdf['SubAgency'].unique




#Scaling 

#Create train set with 70%, test set with 30%

#correlation between Features 



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

 #####Corrr
corr=X_tree1.corr()
f, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(corr,vmax=1, vmin=0.5, square=True)
'''

#saleprice correlation matrix(zoomed heatmap style)
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(18, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

'''
#saleprice correlation matrix(zoomed heatmap style)
'''k = 10 #number of variables for heatmap
cols = corr.nlargest(k, X_tree1(k,'ViolationType')['ViolationType'].index
cm = np.corrcoef(X_tree1[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(18, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)   

'''
plt.figure(figsize=(40,40))
sns.heatmap(X_tree1.corr(),linewidths=0.5,linecolor='black',square=True,cmap='summer')
#greating tests set and train sets 70% 30%
Xb_train, Xb_test, yb_train, yb_test = train_test_split (X_tree1, y_tree , test_size = 0.3, 
                                                         random_state=0)


# step 4 import classifier

from sklearn.tree import DecisionTreeClassifier
model_traffic = DecisionTreeClassifier()

# step 5 train the model



Xb_train.columns

dtree=model_traffic.fit(Xb_train, yb_train)









type(dtree)

'''#viualization for Dtree 
dot_data = StringIO()
export_graphviz(dtree,out_file=dot_data,filled=True,rounded=True,special_characters=True, feature_names=True,)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
'''

# Step 6 predict on test set
yb_pred = model_traffic.predict(Xb_test)


# Step 7 calculate the accuracy score



accuracy_score(yb_test, yb_pred)


# Step 8 look into the confusion matrix / contingency table

cm_traffic = confusion_matrix(yb_test, yb_pred)
TN = cm_traffic[0][0]
FP = cm_traffic[0][1]
FN = cm_traffic[1][0]
TP = cm_traffic[1][1]


accuracy_score(yb_test, yb_pred)
confusion_matrix(yb_test, yb_pred)
print(classification_report(yb_test,yb_pred))


#############
#should we use none, micro, macro or weighted for average for below multiclass
sensitivity1 = recall_score(yb_test, yb_pred, average='micro')
sensitivity2 = recall_score(yb_test, yb_pred, average='macro')

#sensitivity = TP/(TP+FN)
precision1 = precision_score(yb_test, yb_pred, average='micro')
precision2 = precision_score(yb_test, yb_pred, average='macro')

#precision = TP/(TP+FP)

specificity = TN/(TN+FP)

FPR = FP/(TN+FP)

print(sensitivity1)
print(sensitivity2)
print(precision1)
print(precision2)
print(specificity)
print(FPR)

print(Xb_train.columns)

#how to the varibale effect the model how k fold is taking random sample and spliting acrross 
model = DecisionTreeClassifier()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X_tree1, y_tree)):
 model.fit(X_tree1.iloc[train], y_tree.iloc[train])
 score = model.score(X_tree1.iloc[test], y_tree.iloc[test])
 scores.append(score)
print(scores)


'''To many Coulmns  
feature_colums=['SubAgency', 'Description', 'Location', 'Belts', 'Personal Injury',
       'Property Damage', 'CommercialLicense', 'CommercialVehicle', 'Alcohol',
       'State', 'Make', 'Model', 'Color', 'Charge',
       'ContributedToAccident', 'Race', 'Gender', 'Driver State', 'DL State']
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  filled=True, rounded=True, special_characters=True, feature_names = feature_colums,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Violation .png')
Image(graph.create_png())





