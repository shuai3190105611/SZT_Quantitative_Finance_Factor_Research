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
from sklearn.tree import DecisionTreeClassifier
class MyDTree:
    '''
    The desicion tree, implemented myself.
    Add some user-friendly functions.
    Params:
    (DataFrame)dataSet: the used data
    (featureList)featureList: the features of training
    (TreeNode)self.Root: the root of the tree
    (string)label: the name of the label
    (int)max_depth: the maximum deepth
    (int)min_element: the minimum number of the element of a leaf
    (int)max_child: the maximum number of the child nodes
    '''
        
    def __init__( self
                 ,settings
                 ):
        '''
        Function: 
        Initialize. We input the settings only in this part.
        Input:
        (dictionary)settings: to set the value of the self params.
        which is used to be the label of the classification task.
        '''
        self.dataSet=None
        self.featureList=[]
        self.label=None
        self.setParam(settings)
    def setParam(self,settings):
        if 'max_depth' in list(settings.keys()):
            self.max_depth=settings['max_depth']
        if 'min_element' in list(settings.keys()):
            self.min_element=settings['min_element']
        if 'max_child' in list(settings.keys()):
            self.max_child=settings['max_child']
        
    def fit(self,dataSet,label):
        '''
        Function: 
        Initialize. We input the dataSet only in the initialization.
        Input:
        (DataFrame)dataSet: data  set being used.
        (string)label: the label is the name of the selected column
        which is used to be the label of the classification task.
        '''
        self.dataSet=dataSet
        self.featureList=self.dataSet.columns.tolist()
        self.featureList.remove(label)
        self.label=label
        runtime=self.train()
        return runtime
    def printShape(self):
        print("The head of the data")
        print(self.dataSet.head(5))
        row,col=self.dataSet.shape
        print("The row: ",row," the col: ",col)

    def train(self):
        start = datetime.datetime.now()
        self.Root=TreeNode( self.dataSet
                           ,deepcopy(self.dataSet.index.tolist())
                           ,self.featureList
                           ,self.label
                           ,0
                           ,self.max_depth
                           ,self.min_element
                           ,self.max_child)
        end = datetime.datetime.now()
        print('totally time is ', end - start)
        print("OKKKKKKKKKKKKKKKKKKKKK! Training completed!")
        return end-start
    def predict(self,dataTest,factor):
        '''
        Function:
        Predict the value of feature: "default".
        We search from root to leaf. Then use the modal number (class) of the leaf.
        '''
        indexList=dataTest.index.tolist()
        rst=pd.DataFrame(columns=[factor],index=indexList)
        for i in indexList:
            rst.loc[i]=self.Root.search(dataTest.loc[i])
        return rst
class TreeNode:
    '''
    Params:
    The others are same as the MyDTree.
    (float)curEntropy: current entropy
    (string)splitFactor: name of the factor used to split
    (list)childList: the list to contain the child nodes
    '''
    def __init__(self
                 ,dataSet
                 ,indexList
                 ,featureList
                 ,label
                 ,level
                 ,max_depth=10
                 ,min_element=10
                 ,max_child=10):
        
        #########################################################
        ##### 重要的事情用中文说：数据传引用，不要拷贝 ############
        ##########     我们用index来解决即可       ###############
        #########################################################
        
        ################### initialize the data #################
        self.dataSet=dataSet
        self.indexList=indexList
        self.max_depth=max_depth
        self.min_element=min_element
        self.max_child=max_child
        self.level=level
        self.featureList=featureList
        self.label=label
        self.childList=[]
        ########## calculate the entropy and select  ############
        self.splitFactor=self.getFactor()
        ################  split and check  ######################
        if self.splitFactor is None:
            print("LEVEL: ",self.level,"not suitable factor!")
        else:
            childFeature=deepcopy(self.featureList)
            childFeature.remove(self.splitFactor)
            nodeList=self.dataSet.loc[self.indexList,self.splitFactor].drop_duplicates().tolist()
            #print("The list of the nodes are: ", nodeList)
            
            ################# set the childnode  ####################
            '''
            in this part, we need to set the index and features on the
            passed data, and remove the factor of the featurlist, then 
            set the level.
            and we should calculate the entropy for the children. 
            '''
            if self.level<self.max_depth:
                for i in nodeList:
                    
                    indexChildList=self.dataSet[self.dataSet[self.splitFactor]==i].index.tolist()
                    indexChildList=list(set(indexChildList) & set(self.indexList))
                    ##########     recursiveeeeeeeeeee   god bless!    ########
                    newChild=TreeNode(  self.dataSet
                                       ,indexChildList
                                       ,childFeature
                                       ,self.label
                                       ,self.level+1
                                       ,self.max_depth
                                       ,self.min_element
                                       ,self.max_child)
                    self.childList.append(newChild)
        
    def getEntropy(self,factor):
        '''
        Function:
        To calculate a factor's entropy.
        And we use valid to test if there's factor satisfy our
        max\min requirements.
        Input:
        (string)factor: name of the factor we focus on.
        Output:
        (int)valid: 0: not satisfied. 1: satisfied. 
        '''
        sumEntropy=0
        temp=self.dataSet.loc[self.indexList].groupby([factor])[self.label].count()
        valid=1
        #print(temp)
        if len(temp)>self.max_child:
            valid=0
        else:
            for i in temp:
                if i<self.min_element:
                    valid=0
                    sumEntropy+=np.log(float(i)/self.dataSet.loc[self.indexList].shape[0])*float(i)/self.dataSet.loc[self.indexList].shape[0]
                    sumEntropy=-sumEntropy
                #print("sumEntropy = ",sumEntropy)
        return valid,sumEntropy
    def getFactor(self):
        minEntropy=self.dataSet.loc[self.indexList].shape[1]
        factor=None
        for i in self.featureList:
           valid,entro=self.getEntropy(i)
           if valid==0:
               continue
           if entro<minEntropy:
               minEntropy=entro
               factor=i
        print("LEVEL: ",self.level," we get the ",factor, " to split!")
        return factor
    def search(self,dataTuple):
        '''
        Function:
        Search for the leaf recursively. And find the corresponding class.
        Input:
        (DataFrame)dataTuple
        '''
        
        if len(self.childList)==0:
            #################         leaf node        ####################
            ####### calculate which class has the maximum number  #########
            temp=self.dataSet.loc[self.indexList]
            temp=temp.groupby([self.label])[self.dataSet.columns.tolist()[0]].count()
            #print(temp)
            max_index=0
            max_count=0
            for i in temp.index:
                #print("i=" ,i)
                if temp[i]>max_count:
                    max_count=temp[i]
                    max_index=i
            return max_index
            #print("max_index = ", max_index," max_count = ",max_count)
        else:
            flag=dataTuple[self.splitFactor]
            for i in range(len(self.childList)):
                if self.childList[i].dataSet.loc[self.childList[i].indexList[0],self.splitFactor]==flag:
                    return self.childList[i].search(dataTuple)
                    
                
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

######################  data  #############################


df=pd.read_csv("dataSetCleaned.csv")
df=df.set_index(df.columns.tolist()[0])
df=df.dropna()
######################  demo  #############################
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
        temp.loc[tempIndex]=i
    df[col]=temp

settings={'max_depth':10,'min_element':10,'max_child':10}
Demo=DecisionTreeClassifier(criterion="entropy",max_depth=10,random_state=30) #init
count=0
######################  test framework  ###################


SKResult=pd.DataFrame(columns=['Total_Num','True_Num','True_ratio','Run_time'])
for (dfTrain,dfPre) in MySplit(df,10,0.8):
    Path='SKDTree.pkl'+str(count)
    
    
    dfTrainY=dfTrain['"default"']
    dfTrainX=dfTrain
    del dfTrainX['"default"']
    
    start = datetime.datetime.now()
    Demo.fit(dfTrainX.astype('int'),dfTrainY.astype('int')) #training
    end = datetime.datetime.now()
    runTime=end-start
    joblib.dump(Demo, Path) #saving
    #Demo=joblib.load(Path)
    
    
    dfPreY=dfPre['"default"']
    dfPreX=deepcopy(dfPre)
    del dfPreX['"default"']
    
    
    rst=Demo.predict(dfPreX.astype('int'))
    dfPre['rst']=rst
    TrueNum=dfPre[dfPre['rst']==dfPre['"default"']]['rst'].count()
    TotalNum=dfPre['rst'].count()
    print("Total Num:",TotalNum)
    print("True Num:",TrueNum)
    print("True ratio = ",float(TrueNum)/TotalNum)
    print("Run time = ",runTime)
    SKResult.loc[str(count)]=[TotalNum,TrueNum,float(TrueNum)/TotalNum,runTime]
    count+=1
SKResult.to_csv("SKResult.csv")
      




'''
######################  demo  #############################

settings={'max_depth':10,'min_element':10,'max_child':10}
Demo=MyDTree(settings) #init
count=0
######################  test framework  ###################


MyResult=pd.DataFrame(columns=['Total_Num','True_Num','True_ratio','Run_time'])
for (dfTrain,dfPre) in MySplit(df,10,0.8):
    Path='MyDTree.pkl'+str(count)
    runTime=Demo.fit(dfTrain,'"default"') #training
    joblib.dump(Demo, Path) #saving
    #Demo=joblib.load(Path)
    rst=Demo.predict(dfPre,'"default"')
    dfPre['rst']=rst
    TrueNum=dfPre[dfPre['rst']==dfPre['"default"']]['rst'].count()
    TotalNum=dfPre['rst'].count()
    print("Total Num:",TotalNum)
    print("True Num:",TrueNum)
    print("True ratio = ",float(TrueNum)/TotalNum)
    print("Run time = ",runTime)
    MyResult.loc[str(count)]=[TotalNum,TrueNum,float(TrueNum)/TotalNum,runTime]
    count+=1
MyResult.to_csv("MyResult.csv")
'''      






