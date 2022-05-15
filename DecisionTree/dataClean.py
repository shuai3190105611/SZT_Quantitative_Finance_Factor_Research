# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:38:30 2022

@author: ZTSHUAI
"""
import pandas as pd
import numpy as py
df=pd.read_csv("bank-additional-full.csv")
columns=df.columns.tolist()[0].split(";")
dataList=df.iloc[:,0].tolist()
dataSet=pd.DataFrame(df,columns=columns)
for i in range(len(dataList)):
    dataSet.loc[str(i)]=dataList[i].split(";")
    if i%100==0:
        print(dataList[i].split(";"))


dataSet.to_csv("dataSetCleaned_new.csv")