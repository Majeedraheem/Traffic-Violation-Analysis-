#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:46:04 2019

@author: mac
"""
import os
import datetime as dt
import pandas as pd
import numpy  as np
import seaborn as sns 
import matplotlib.pyplot as plt

##Data Cleaning#######
#drops the following columns. axis=1 denotes that these are columns not rows And also make sure the cloumns have no missing values
#####Cleaning Data modifing DataSet Coulmns name Removing space greating new dataset calleed Cleaned Data
#REMOVE Unknow Gender from dataset 

os.getcwd()
os.chdir('/Users/mac/Desktop/PyThon /')
Traffic_Violations_DF=pd.read_csv("Traffic_Violations.csv",nrows=10000, na_values='NA')
Traffic_Violations_DF.to_csv('Traffic_Violation_10000.csv', index=False)
Traffic_Violations_DF=pd.read_csv("Traffic_Violation_10000.csv", na_values='NA')
Traffic_Violations_DF=Traffic_Violations_DF.drop(['Geolocation','Latitude','Longitude','Agency','VehicleType','Work Zone','Driver City','HAZMAT','Fatal','Work Zone','Article'],axis=1)
Traffic_Violations_DF=Traffic_Violations_DF.dropna()
Cleaned_Traffic_Violations_DF=Traffic_Violations_DF[(Traffic_Violations_DF.Gender !='U')]
Cleaned_Traffic_Violations_DF=Traffic_Violations_DF[(Traffic_Violations_DF['Gender'].notnull())&(Traffic_Violations_DF['Violation Type'].notnull())&(Traffic_Violations_DF['Alcohol'].notnull())&(Traffic_Violations_DF['Description'].notnull())&(Traffic_Violations_DF['Time Of Stop'].notnull()&(Traffic_Violations_DF['Year'].notnull())&(Traffic_Violations_DF['Charge'].notnull())&(Traffic_Violations_DF['Driver City'].notnull())&(Traffic_Violations_DF['DL State'].notnull())&(Traffic_Violations_DF['Arrest Type'].notnull()))]
Cleaned_Traffic_Violations_DF_Columns=Cleaned_Traffic_Violations_DF.rename(columns={'Violation Type':'ViolationType','Driver City':'DriverCity','Arrest Type':'ArrestType'})
Cleaned_Traffic_Violations_DF.info()
Cleaned_Traffic_Violations_DF.shape
Cleaned_Traffic_Violations_DF.to_csv('Cleaned_Traffic_Violations_DF_10000.csv', index=False)
Cleaned_Traffic_Violations_DF=pd.read_csv("Cleaned_Traffic_Violations_DF_10000.csv", na_values='NA')
Cleaned_Traffic_Violations_DF.info()
Cleaned_Traffic_Violations_DF.dropna()
Cleaned_Traffic_Violations_DF.shape

#######################################################################################################################
#Question 1: How many male and femal our dataset have ? bar chart
Cleaned_Traffic_Violations_DF.Gender.value_counts().plot(kind='bar')
Cleaned_Traffic_Violations_DF_Gender=Cleaned_Traffic_Violations_DF(['Gender']).values_counts()
Cleaned_Traffic_Violations_DF_Gender=Cleaned_Traffic_Violations_DF.groupby((['Gender'])).count()
Cleaned_Traffic_Violations_DF_Gender.shape
M_Count=Cleaned_Traffic_Violations_DF[Cleaned_Traffic_Violations_DF.Gender=='M']
F_Count=Cleaned_Traffic_Violations_DF[Cleaned_Traffic_Violations_DF.Gender=='F']
####################################################################################################################### 
###Question 2:Which date with most violation after ? Table or barchart
tv=Cleaned_Traffic_Violations_DF_Columns
tv.info()
tv['stopdate']=pd.to_datetime(tv['Date Of Stop']).dt.date.astype('datetime64[ns]')
tv['stoptime']=pd.to_datetime(tv['Time Of Stop']).dt.date.astype('datetime64[ns]')
tv.info()
tv.drop(['Date Of Stop','Time Of Stop'],axis=1,inplace=True)
tv.info()
tv['WeekDay']=tv['stopdate'].apply(lambda x:x.weekday())
replace_map={'WeekDay':{0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}}
tv.replace(replace_map,inplace=True)
tv.WeekDay.unique()
tv.WeekDay.value_counts().plot(kind='bar')
###################################################################################################################
###Question 3: which month with most violation happend ? pie chart or bar
tv['Month']=tv['stopdate'].dt.month_name()
tv.Month.value_counts().plot(kind='bar')
#QUESTION 3 Is there realtion between the carType and Voilation ? WHAT IS THE MOST CARTYPE COMMITED THE VOILATION?
Cleaned_Traffic_Violations_DF_Corr_CarType_Voilation=Cleaned_Traffic_Violations_DF_Columns.loc[:, ['VehicleType','Make','Model']]
Cleaned_Traffic_Violations_DF_Corr_CarType_Voilation.shape
#####################################################################################################################
###Question 4: What violation Type precentage  in whole dataset? piechart
tv['ViolationType'].value_counts().plot(kind='pie',autopct='%1.1f%%',title='Violation Distribution')
#####################################################################################################
#Question 5  what race has most violation commited and what type of voilation ? barchar or pie
Cleaned_Traffic_Violations_DF_Race_Ttype=Cleaned_Traffic_Violations_DF_Columns.groupby((['Race','ViolationType'])).count()
Cleaned_Traffic_Violations_DF_Columns.groupby(['Race','ViolationType']).size().unstack().plot(kind='bar',stacked=True)
######################################################################################################################
#Question No 6: How many Female , male commited Traffic Violation GROUPED BY Gender and ViolationType?
Cleaned_Traffic_Violations_DF_Gender_Ttype=Cleaned_Traffic_Violations_DF.groupby((['Gender','Violation Type'])).count()
Cleaned_Traffic_Violations_DF.groupby(['Gender','Violation Type']).size().unstack().plot(kind='bar',stacked=True)
######################################################################################################################
# 7 what make of car  with most Voilation ?
"""Cleaned_Traffic_Violations_DF_Make=Cleaned_Traffic_Violations_DF.groupby(['Make'])['Violation Type'].count().reset_index(name='NumbersOfViolation')#tv.Make.value_counts().plot(kind='bar')
TopViolation=Cleaned_Traffic_Violations_DF_Make.sort_values('NumbersOfViolation',ascending=False)
Number_Violation=TopViolation.NumbersOfViolation.head().reset_index(drop=True)
Make_Index=TopViolation.Make.head().reset_index(drop=True)
Make_Index=list(Make_Index)
pos = list(np.arange(len(Make_Index)))
plt.xticks(pos, Make_Index)#use name insted of having numbers under the bar replace the numbers by names
plt.xlabel('Make', fontsize=16)
plt.ylabel('Number Of violation', fontsize=16)
plt.title('Barchart - Violation By Make',fontsize=20)
plt.bar(pos,Number_Violation,color='blue',edgecolor='black')
plt.show()
#######################################################################################################3########
#tables 7 b
tv.Make.unique()
infinity_citation=tv.loc[(tv.Make=='INFINITY') & (tv.ViolationType=='Citation')]
infinity_citation.info()
x= tv[(tv.Make=='INFINITY') & (tv.ViolationType=='Citation')].count()
infinity_citation.Belts.value_counts()
"""
##############################################################################################################
#Question 7 how many people have involved in Accindent and what type of violation its  ? 
Cleaned_Traffic_Violations_DF_Accident_Ttype=Cleaned_Traffic_Violations_DF.groupby((['Accident','Violation Type'])).count()
Cleaned_Traffic_Violations_DF_Accident_Ttype.columns
Cleaned_Traffic_Violations_DF.groupby(['Accident','Violation Type']).size().unstack().plot(kind='bar',stacked=True)
#########################################################################################################################################
#Question 8 what is the time where the most  voilation happened DayTime AfterNoon NightTime?
Cleaned_Traffic_Violations_DF['ViolationHour']=pd.DatetimeIndex(Cleaned_Traffic_Violations_DF['Time Of Stop']).hour
Cleaned_Traffic_Violations_DF['ViolationHour'].value_counts().plot(kind='pie',autopct='%1.1f%%', title='Violation Hourly Distribution')
##########################################################################################################################################
#9 How many arrest happened per year and which type of arrest

Cleaned_Traffic_Violations_DF_ArrestType=Cleaned_Traffic_Violations_DF.groupby((['Arrest Type','Year'])).count()
Cleaned_Traffic_Violations_DF_ArrestType=Cleaned_Traffic_Violations_DF.groupby[['Arrest Type']['Year']].value_counts().plot(kind='pie',autopct='%1.1f%%',title='Arrest By year ')
###########################################################################################

#10
"""df=Cleaned_Traffic_Violations_DF
df.loc[df['Make'].str.contains('HON'), 'Make'] = 'HONDA'
df.loc[df['Make'].str.contains('TOYT'), 'Make'] = 'TOYOTA'
df.loc[df['Make'].str.contains('CHEV'), 'Make'] = 'CHEVROLET'
df.loc[df['Make'].str.contains('CADI'), 'Make'] = 'CADILLAC'
df.loc[df['Make'].str.contains('ACUR'), 'Make'] = 'ACURA'
Cleaned_Traffic_Violations_DF_Make_C=df
-----------------------------------------------------------------------
CountCharges=Cleaned_Traffic_Violations_DF.groupby(['Make'])['Charge'].count().reset_index(name='FrequencyOfCharges')
TopCharges=CountCharges.sort_values('FrequencyOfCharges',ascending=False).head().reset_index(drop=True)
TopCharges.counts().polt(kind='bar')
TopChargesMake=list(TopCharges['Make'])"""

