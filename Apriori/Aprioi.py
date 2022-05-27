# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:39:23 2022

@author: ZTSHUAI
"""

import pandas as pd
import numpy as np
import datetime
import joblib
from copy import deepcopy

def MySplit(dataSet,K,ratio):
    '''
    Function:
    To generate the data for training and testing.
    Input:
    (DataFrame)dataSet: the raw data set.
    (int)K: the number of turns
    (float)ratio: the ratio of splitting
    '''
    returnSet=[]
    for i in range(K):
        train_start=(int(dataSet.shape[0]*i*1.0/K))%dataSet.shape[0]
        train_end=(int(ratio*dataSet.shape[0])+int(dataSet.shape[0]*i*1.0/K))%dataSet.shape[0]
        print(train_start,train_end)
        if train_start>train_end:
            dfPre=dataSet.iloc[train_end:train_start]
            dfTrain=pd.concat([dataSet.iloc[0:train_end],dataSet.iloc[train_start:]])
        else:
            dfTrain=dataSet.iloc[train_start:train_end]
            dfPre=pd.concat([dataSet.iloc[0:train_start],dataSet.iloc[train_end:]])

        returnSet.append((dfTrain,dfPre))
    return returnSet

class Apriori:
    '''
    Input:
    (DataFrame)self.dataSet: the dataSet to mine
    (float)threshold: the minimum the support of transactions(or tuples) should be
    (int)k: the number of features in the transactions.
    (list)retainList: to retain the satified minimum terms
    (list)support: record the score of the support
    Note:
    we need to encode the values of features for convenience.
    '''
    
    def __init__(
            self
            ,dataSet
            ):
        self.dataSet=dataSet
        self.support=[]
        
    def countNum(self):
        for col in self.dataSet.columns:
            print("###############")
            print(col)
            countList=deepcopy(self.dataSet[col])
            for j in countList:
                print("feature: ",j)
                print("number: ",self.dataSet[self.dataSet[col]==j][col].count())
    def check(self,colSingle,checkList):
        '''
        Function:
        To check if the column's name is in the list
        Output:
        1: is in. 0: not in.
        '''
        flag=0
        for item in checkList:
            if colSingle==item:
                flag=1
                break
        return flag
        
    def retainImptRe(self,single):
        '''
        Function:
        return the important terms.
        Input:
        (list)single: the feasible features can be added in.list of lists.
        note:
        pay attention to skip the same iterms to avoid the conditions like: (x1,x1)
        '''
        temp=deepcopy(self.retainList)
        for item in self.retainList:
            #search for the labels
            #use a list to contain
            checkList=[]
            for i in item:
                posItem_=i.find("_")
                colItem=i[0:posItem_]
                checkList.append(colItem)
                
            for i in range(len(single)):
                #print("i = ",i)
                for j in range(len(single[i])):
                    posSingle_=single[i][j].find("_")
                    colSingle=single[i][j][0:posSingle_]
                    if self.check(colSingle,checkList)==1:
                        continue
                    tempItem=deepcopy(item)
                    tempItem.append(single[i][j])
                    probNew=self.getSup(tempItem)
                    probOld=self.getSup(item)
                    #print("item = ",item)
                    #print("probNew = ",probNew,"probOld = ",probOld)
                    if probNew>=self.threshold * probOld:
                        temp.append(tempItem)
                        self.support.append(probNew)
                    #print(len(self.support),len(temp))
        return temp   
    def retainImptAb(self,single):
        '''
        Function:
        return the important terms.
        Input:
        (list)single: the feasible features can be added in.list of lists.
        note:
        pay attention to skip the same iterms to avoid the conditions like: (x1,x1)
        '''
        temp=[]
        for item in self.retainList:
            checkList=[]
            for i in item:
                posItem_=i.find("_")
                colItem=i[0:posItem_]
                checkList.append(colItem)
                
            for i in range(len(single)):
                for j in range(len(single[i])):
                    posSingle_=single[i][j].find("_")
                    colSingle=single[i][j][0:posSingle_]
                    if self.check(colSingle,checkList)==1:
                        continue
                    tempItem=item
                    tempItem.append(single[i][j])
                    prob=self.getSup(tempItem)
                    if prob>=self.threshold * self.total:
                        temp.append(tempItem)
                        self.support.append(prob)
        return temp   
    def getSup(self,tempImpt):
        '''
        Function:
        To calculate the support.
        Input:
        (list)tempImpt: the item append single[i][j]
        '''
        indexList=self.dataSet.index.tolist()
        for i in tempImpt:
            pos_=i.find("_")
            col=i[0:(pos_)]
            indexList=list(set(indexList) & set(self.dataSet[self.dataSet[col]==i][col].index.tolist()))

        prob=1.0 * len(indexList)/self.total
        #print(prob,len(tempImpt),len(self.dataSet[self.dataSet[col]==i][col].index.tolist()))
        return prob
    def findMax(
            self
            ,threshold
            ,k
            ,option="re"
            ):
        '''
        Function:
        Find the max support k-transactions
        Input:
        (float)threshold: the minimum the support of transactions(or tuples) should be
        (int)k: the number of features in the transactions.
        (string)option: "re": relative support,"ab": absolute support
        '''
        self.threshold=threshold
        self.k=k
        row=self.dataSet.shape[0]
        self.total=row
        self.retainList=[]
        if k>len(self.dataSet.columns.tolist()):
            print("Error! The k is bigger than the number of features!")
        else:
            '''
            And we need the single list to denote which single items can be added in.
            list of lists.
            '''
            single=[]
            for col in self.dataSet.columns.tolist():
                countList=deepcopy(self.dataSet[col].drop_duplicates().tolist())
                single.append(countList)
            #print(single)
            for layer in range(self.k):
                
                temp=[]
                if layer==0:
                    ##  the logic is totally different from that belows in the else block ##
                    for i,col in enumerate(self.dataSet.columns):
                        temp1=[]
                        for j in range(len(single[i])):
                            if self.dataSet[self.dataSet[col]==single[i][j]][col].count()>self.threshold*self.total:
                                temp1.append(single[i][j])
                                temp2=[]
                                temp2.append(single[i][j])
                                if len(temp2)>0:
                                    self.retainList.append(temp2)
                                    self.support.append(1.0*self.dataSet[self.dataSet[col]==single[i][j]][col].count()/self.total)
                        if len(temp1)>0:
                            temp.append(temp1)
                            
                    single=deepcopy(temp)
                    #print(len(self.support),len(self.retainList))
                else:
                    
                     '''
                     The items in retainList are in the form of list:[x,y].
                     We search if we can add a element and still satisfy the threshold.
                     '''
                     if option=="ab":
                         temp=self.retainImptAb(single)
                     else:
                         temp=self.retainImptRe(single)
                     self.retainList=deepcopy(temp)
                     #print(len(self.support),len(self.retainList))
                     
                #print("retainList = ")
                #print(self.retainList)
            
            df=pd.DataFrame(self.retainList)
            support=pd.DataFrame(self.support)
            df=pd.concat([df,support],axis=1)
            df.to_csv("result_K"+str(self.k)+"_row_"+str(self.total)+"_threshold_"+str(self.threshold)+".csv")
            

######################  data  #############################


df=pd.read_csv("dataSetCleaned.csv")
df=df.set_index(df.columns.tolist()[0])
df=df.dropna()
######################  demo  #############################

######   replace the feature to int numbers   #############
def replaceToCode(df):
    '''
    Function:
    To replace the feature into int numbers.
    And we need to remove the "_"
    None.

    '''
    tempCol=df.columns.tolist()
    for i in tempCol:
        temp=i
        temp=temp.replace("_","")
        df.rename(columns={i:temp},inplace=True)
        #print(temp,i)
    for col in df.columns:
        countList=df[col].drop_duplicates().tolist()
        count=len(countList)
        '''
        print("countList = ",countList)
        print("count = ",count)
        '''
        temp=deepcopy(df[col])
        for i in range(count):
            '''
            print("count i = ",countList[i])
            print(df[df[col]==countList[i]][col])
            print("i = ",i)
            '''
            tempIndex=df[df[col]==countList[i]][col].index.tolist()
            tempStr=str(col)+'_'+str(countList[i])
            temp.loc[tempIndex]=tempStr
        df[col]=temp
    
    
    print(df)
    return df
######################  test framework  ###################
df=replaceToCode(df)
dftemp=deepcopy(df)
testTime=[]
kList=[]
totalList=[]
threshold=0.7
for i in range(4):
    for j in range(4):
        df=dftemp.loc[dftemp.index.tolist()[0:(j+1)*10000]]
        start = datetime.datetime.now()
        AP=Apriori(df)
        AP.findMax(threshold,i+1)
        end = datetime.datetime.now()
        testTime.append((end-start).microseconds)
        kList.append(i)
        totalList.append((j+1)*10000)
testTime=pd.DataFrame(testTime)
kList=pd.DataFrame(kList)
totalList=pd.DataFrame(totalList)
testTime=pd.concat([testTime,kList,totalList],axis=1)
testTime.columns=['testTime','k','totalData']
testTime.to_csv("testTime"+str(threshold)+".csv")
#AP.countNum()








