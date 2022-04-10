import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
class RetainImptDA:
    '''
    
    
    self.variables:
    (list)self.DataSet: the input list of datasets of assets(DataFrame)
    (list)self.Return: the input list of Return of assets(DataFrame)
    (list)self.Price: the input list of Price of assets(DataFrame)
    (list)selfImpt_Sroce: score of the importance. based on t-test result or
    volume * t-test result.
    (array)self.RiskExpose: the array of risk exposure
    Note:
    This class is mainly to retain the important factors,
    based on several different methods.
    Return the factors' position (in order)in the raw DataSet.
    DA:based on :different assets have diffenert function.
    But we can't use IC method.
    we input the DataSet to be processed.
    we output the retained list. And the RiskExpose(dataframe)
    In this class, the number of variables is smaller,
    compared to the other classes we use.
    Please firstly use the get_impt(self,weight)
    '''
    def __init__(self,DataSet,Return,Price):
        '''
        Input: 
        (list)DataSet: the input list of datasets of assets(DataFrame)
        (list)Return: the input list of Return of assets(DataFrame)
        (list)Price: the input list of Price of assets(DataFrame)
        '''
        row,col=DataSet[0].shape
        self.DataSet=copy.deepcopy(DataSet)
        self.Return=copy.deepcopy(Return)
        self.Price=copy.deepcopy(Price)
        self.RiskExpose=np.zeros((len(DataSet),col))
        self.T_Result=np.zeros((len(DataSet),col))
        self.Impt_Score=np.zeros(col)
    def get_reg_impt_multi(self):
        '''
        Function:Multi-factor regression to get risk-expsure
        Note:
        calculate the assets' RiskExposure
        It's the step to get the matrix of RiskExposure.
        We mainly want to get a matrix about the RiskExposure of each factor,
        for each asset.
        For each asset,we use their own data to do this.
        The number of tuples of different assets would be different.
        Do the T-test based on the LinearRegression.
        It's a normal thinking.
        We use the data of RiskExposure matrix to do this.
        Muiti-factor regression
        '''
        for k in range(len(self.DataSet)):

            X=copy.deepcopy(self.DataSet[k])
            X = sm.add_constant(X) 
            Y=copy.deepcopy(self.Return[k])
            model = sm.OLS(Y,X).fit()
            W=model.params
            bais=model.bse
            row,col=self.DataSet[k].shape
            for j in range(col):
                self.RiskExpose[k][j]=W[j+1]
                self.T_Result[k][j]=float(W[j+1])/float(bais[j+1])
    def get_reg_impt_singel(self):
        '''
        Function:Single-factor regression to get risk-expsure
        '''
        row,col=self.DataSet[0].shape
        for k in range(len(self.DataSet)):
            for j in range(col):
                X=copy.deepcopy(self.DataSet[k][self.DataSet.columns.to_list()[j]])
                X = sm.add_constant(X) 
                Y=copy.deepcopy(self.Return[k])
                model = sm.OLS(Y,X).fit()
                W=model.params
                bais=model.bse
                self.RiskExpose[k][j]=W[1]
                self.T_Result[k][j]=float(W[1])/float(bais[1])
    def get_impt_t(self,weight=False,IsMulti=1):    
        '''
        Function: get the t value
        Input:
        (bool)wight:
        True: consider the weight,mainly for volume.
        False: don't consider the weight.
        (bool)IsMulti:
        1: Multi regression
        0: single regression
        '''
        if IsMulti==1:
            self.get_reg_impt_multi()
        else:
            self.get_reg_impt_single()
        if weight==True:
            volume_list=[]
            for i in range(len(self.DataSet)):
                volume_sum=self.Price[i]['volume'].sum()
                volume_list.append(volume_sum)
            row,col=self.DataSet[0].shape
            for i in range(col):
                score=0
                for k in range(len(self.DataSet)):
                    score+=volume_list[k]*self.T_Result[k][i]
                self.Impt_Score[i]=score
        else :
            row,col=self.DataSet[0].shape
            for i in range(col):
                score=0
                for k in range(len(self.DataSet)):
                    score+=self.T_Result[k][i]
                self.Impt_Score[i]=score
    def get_impt_ttest(self):
        '''
        Function: get the t-test results
        output:
        (list)rst:a list of names
        '''
        
        column=self.DataSet[0].columns.to_list()
        score_list=copy.deepcopy(self.Impt_Score)
        score_list=score_list.tolist()
        print(score_list)
        score_list_copy=copy.deepcopy(score_list)
        pos=np.arange(0,len(score_list),1)
        pos=pos.tolist()
        for i in range(len(score_list)-1):
            for j in range(0,i):
                if score_list[j]<score_list[j+1]:
                    temp=score_list[j]
                    score_list[j]=score_list[j+1]
                    score_list[j+1]=temp
                    temp=pos[j]
                    pos[j]=pos[j+1]
                    pos[j+1]=temp
        rst=[]
        for i in range(len(self.Impt_Score)):
            rst.append(column[pos[i]])
        return rst
    def get_riskexpose(self):
        '''
        Function:Return a total DataFrame of RiskExpose.
        '''
        self.RiskExpose=copy.deepcopy(pd.DataFrame(self.RiskExpose,columns=self.DataSet[0].columns.to_list()))
        return copy.deepcopy(self.RiskExpose)
class RetainImptDT:
    '''
    self variables:
    (list)self.DataSet: the input list of datasets of assets(DataFrame)
    (list)self.Return: the input list of Return of assets(DataFrame)
    (list)self.Price: the input list of Price of assets(DataFrame)
    (list)self.Impt_Sroce: score of the importance. based on t-test result
    
    
    
    Note:
    !!!!!!!!!  In this part  !!!!!!!!!
    !!!!!!!!we do two regression!!!!!!
    One to get the impt score based on
    regression on T.
    And then we regression on asset,
    !!!!!!!to get the RiskExpose.!!!!!!
    DA:based on :different time-stamp have diffenert function.
    In other word, we may use cross-section regression.
    This class is mainly to retain the important factors,
    based on several different methods.
    In this class, the number of variables is smaller,
    compared to the other classes we use.
    Please firstly use the get_impt(self,weight)
    '''
    def __init__(self,DataSet,Return,Price):
        '''
        Input: 
        (list)DataSet: the input list of datasets of assets(DataFrame)
        (list)Return: the input list of Return of assets(DataFrame)
        (list)Price: the input list of Price of assets(DataFrame)
        Note:
        Due to the methods of sampling, the numbers of tuples of different
        DataSet[i] may be diffent, so we get the max number of the tuples.
        Then for each tuple's corresponding time, we get the data and do OLS
        or other methods.
        Time format: 20xx-xx-xx
        '''
        Row=0
        '''
        Get the time-stamps
        '''
        self.TimeList=pd.DataFrame(columns=['time'])
        for k in range(len(DataSet)):
            DataSetNew=pd.DataFrame(DataSet[k].index.to_list,columns=['time'],index=DataSet[k].index.to_list)
            self.TimeList=pd.concat([self.TimeList,DataSetNew])
            self.TimeList=self.TimeList.drop_duplicates()
            self.TimeList=self.TimeList.sort_index()
        Row,col=self.TimeList.shape
        row,col=DataSet[0].shape
        self.DataSet=copy.deepcopy(DataSet)
        self.Return=copy.deepcopy(Return)
        self.Price=Price
        self.RiskExpose=np.zeros((Row,col))
        self.RiskExposeA=np.zeros((len(DataSet),col))
        self.T_Result=np.zeros((Row,col))
        self.Impt_Score=np.zeros(col)
        self.IC_Score=np.zeros(col)
    def get_X(self,T):
        '''
        Function: get X data
        Input: 
        (timedate)T:the timestamp
        '''
        Result=pd.DataFrame(columns=self.DataSet[0].columns.to_list())
        for k in range(len(self.DataSet)):
            if self.TimeList[T] in self.DataSet[k].index.to_list():
                Result=pd.concat([Result,self.DataSet[k].loc[self.TimeList[T]]])
        return Result
    def get_Y(self,T):
        '''
        Function: get Y data
        Input: 
        (timedate)T:the timestamp
        '''
        Result=pd.DataFrame(columns=self.Return[0].columns.to_list())
        for k in range(len(self.Return)):
            if self.TimeList[T] in self.Return[k].index.to_list():
                Result=pd.concat([Result,self.Return[k].loc[self.TimeList[T]]])
        return Result
    def get_reg_impt_multi(self):
        
        row,col=self.TimeList.shape
        for k in range(row):
            X=self.get_X(k)
            Y=self.get_Y(k)
            model = sm.OLS(Y,X).fit()
            W=model.params
            bais=model.bse
            row,col=self.DataSet[k].shape
            for j in range(col):
                self.RiskExpose[k][j]=W[j+1]
                self.T_Result[k][j]=float(W[j+1])/float(bais[j+1])
    def get_reg_impt_singel(self):
        '''
        Function: Single-factor regression to get the result
        '''
        row,col=self.DataSet.shape
        for k in range(len(self.DataSet)):
            for j in range(col):
                X=copy.deepcopy(self.DataSet[k][self.DataSet.columns.to_list()[j]])
                X = sm.add_constant(X) 
                Y=copy.deepcopy(self.Return[k])
                model = sm.OLS(Y,X).fit()
                W=model.params
                bais=model.bse
                self.RiskExpose[k][j]=W[1]
                self.T_Result[k][j]=float(W[1])/float(bais[1])
    def get_impt_t(self):    
        row,col=self.DataSet[0].shape
        for i in range(col):
            score=0
            for k in range(len(self.TimeList)):
                score+=self.T_Result[k][i]
            self.Impt_Score[i]=score
    def get_riskexpose(self):
        self.RiskExposeA=copy.deepcopy(pd.DataFrame(self.RiskExposeA,columns=self.DataSet[0].columns.to_list()))
        return copy.deepcopy(self.RiskExposeA)
    def get_impt_ttest(self):
        '''
        Function: get the t-test results
        '''
        '''
        output:
        (list)rst:a list of names
        '''
        
        column=self.DataSet[0].columns.to_list()
        score_list=copy.deepcopy(self.Impt_Score)
        score_list=score_list.tolist()
        score_list_copy=copy.deepcopy(score_list)
        pos=np.arange(0,len(score_list),1)
        pos=pos.tolist()
        for i in range(len(score_list)-1):
            for j in range(0,i):
                if score_list[j]<score_list[j+1]:
                    temp=score_list[j]
                    score_list[j]=score_list[j+1]
                    score_list[j+1]=temp
                    temp=pos[j]
                    pos[j]=pos[j+1]
                    pos[j+1]=temp
        rst=[]
        for i in range(len(self.Impt_Score)):
            rst.append(column[pos[i]])
        return rst
    def get_reg_impt_multi(self):
        '''
        Function:Multi-factor regression to get risk-expsure
        Note:
        calculate the assets' RiskExposure
        It's the step to get the matrix of RiskExposure.
        We mainly want to get a matrix about the RiskExposure of each factor,
        for each asset.
        For each asset,we use their own data to do this.
        The number of tuples of different assets would be different.
        Do the T-test based on the LinearRegression.
        It's a normal thinking.
        We use the data of RiskExposure matrix to do this.
        Muiti-factor regression
        '''
        for k in range(len(self.DataSet)):

            X=copy.deepcopy(self.DataSet[k])
            X = sm.add_constant(X) 
            Y=copy.deepcopy(self.Return[k])
            model = sm.OLS(Y,X).fit()
            W=model.params
            bais=model.bse
            row,col=self.DataSet[k].shape
            for j in range(col):
                self.RiskExposeA[k][j]=W[j+1]