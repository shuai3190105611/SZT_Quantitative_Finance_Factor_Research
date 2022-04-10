# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:07:35 2022

@author: ZTSHUAI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
import DataClean as dc
class TestFrameWork:
    '''
    self variables:
    (list)self.FactorList: list of Factor's name
    (list)self.AssetList: list of Asset's code
    (list)self.DataSet: list of each asset's corresponding data
    (DataFrame)self.Weight: the weight of the asset
    (DataFrame)self.RiskExposeDA: the matrix of the risk exposure of each factor.The size don't need to be 
    modified.We get the column names and RiskExpose,then we index the data more conveniently.
    (list)self.Return: list of Assets' return
    (list)self.ReturnList:the name of the return
    (string)self.Name: the name of the portfolio
    (list)self.Price: list of DataFrame for:open,close,volume,fee(not used).The order should be this.
    '''
    ##############################  Part 1: initialize ################################
    def __init__(
        self,
        Factor_List,
        Asset_List,
        Return_List,
        Weight,
        RiskExposeDA,
        Name,        
        Price_List=['open','close','volume']
                ):
        '''
        Input:
        (list)Factor_List: list of Factor's name
        (list)Asset_List: list of Asset's code
        (DataFrame)Weight: the weight of the asset
        (DataFrame)RiskExposeDA: the matrix of the risk exposure of each factor.The size don't need to be
        modified.We get the column names and RiskExpose,then we index the data more conveniently.
        (list)Return_List:the name of the return
        Note:
        The factor in DataSet may be same as someone in Price.
        '''
        self.FactorList=copy.deepcopy(Factor_List)
        self.AssetList=copy.deepcopy(Asset_List)
        self.Name=Name
        self.DataSet=[]
        self.Weight=copy.deepcopy(Weight)
        self.RiskExposeDA=pd.DataFrame(columns=self.FactorList)
        self.Return=[]
        self.ReturnList=copy.deepcopy(Return_List)
        self.PriceList=copy.deepcopy(Price_List)
        self.Price=[]
        self.pred_return=pd.DataFrame(columns=['pred_return'])
    def get_data_initial(
        self,
        Start_Time,
        End_Time
                        ):
        '''
        Function:
        Use the information from Asset and Factor,
        and the Start_Time and the End_Time to
        get the data from rqdata database.
        '''
        '''
        Input:
        Start_Time: input start time
        End_Time: input end time
        '''
        import rqdatac
        from rqdatac import get_price,init,get_factor
        init(,)#your account
        
        for i in range(len(self.AssetList)):
            
            df=get_factor(self.AssetList[i],self.FactorList,start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            self.DataSet.append(df)
        for i in range(len(self.AssetList)):
            df=get_factor(self.AssetList[i],self.ReturnList,start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            print('df=',df)
            self.Return.append(df)
        for i in range(len(self.AssetList)):

            df=get_price(self.AssetList[i],start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            df=df[self.PriceList]
            self.Price.append(df)
       
    def get_baseline_initial(
        self,
        BaseLine,
        
                        ):
        '''
        Function: to fetch the return data of baseline.
        Input:
        (list)BaseLine: the code of the baseline asset.
        '''
        Start_Time=self.Price[0].index.to_list()[0]
        End_Time=self.Price[0].index.to_list()[-1]
        import rqdatac
        from rqdatac import get_price,init,get_factor
        init('17757482910','wangzhe6012,')
        df=get_factor(BaseLine,self.ReturnList,start_date=Start_Time,end_date=End_Time)
        df=df.reset_index()
        del df['order_book_id']
        df=df.set_index('date')
        return copy.deepcopy(df)
    ########################### Part 2 : data fetch ###############################
    '''
    Function: get the target data
    Input:
    data
    (int)index: to index the position
    '''
    def get_pred_return(self):
        return copy.deepcopy(self.pred_return)
    ########################### Part 3 : data set   ###############################
    '''
    Function: set the target data
    Input:
    data
    (int)index: to index the position
    '''
    
    def set_weight(self,weight):
        self.Weight=copy.deepcopy(weight)
    def set_dataset(self,dataset):
        self.DataSet=copy.deepcopy(dataset)
    def set_dataset_single(self,data,index):
        self.DataSet[index]=copy.deepcopy(data)
    def set_return(self,Return):
        self.Return=copy.deepcopy(Return)
    def set_return_single(self,Return,index):
        self.Return[index]=copy.deepcopy(Return)
    def set_price(self,Price):
        self.Price=copy.deepcopy(Price)
    def set_price_single(self,Price,index):
        self.Price[index]=copy.deepcopy(Price)
    ######################### Part 4 : data describe #############################
    def describe(self):
        '''
        Function:print the information about the data
        '''
        print('--------------------DESCRIBE----------------------')
        print('This is the information of the portfolio ',self.Name)
        print('--------------------------------------------------')
        print('Weight:')
        print(self.Weight)
        print('--------------------------------------------------')
        print('Risk Exposure:')
        print(self.RiskExposeDA)
        print('--------------------------------------------------')
        print('DataSet:')
        print(self.DataSet)
        print('--------------------------------------------------')
        print('Return:')
        print(self.Return)
        print('--------------------------------------------------')
        print('AssetList:')
        print(self.AssetList)
        print('--------------------------------------------------')
        print('FactorList:')
        print(self.FactorList)
        print('--------------------------------------------------')
        print('Price:')
        print(self.Price)
    def describe_shape(self):
        '''
        function: print the information about the data
        '''
        print('-----------------DESCRIBE shape-------------------')
        print('This is the information of the portfolio ',self.Name)
        print('--------------------------------------------------')
        print('Weight:')
        print(len(self.Weight))
        print('--------------------------------------------------')
        print('Risk Exposure:')
        print(self.RiskExposeDA.shape)
        print('--------------------------------------------------')
        print('DataSet:')
        print(len(self.DataSet))
        print('--------------------------------------------------')
        print('Return:')
        print(len(self.Return))
        print('--------------------------------------------------')
        print('AssetList:')
        print(len(self.AssetList))
        print('--------------------------------------------------')
        print('FactorList:')
        print(len(self.FactorList))
        print('--------------------------------------------------')
        print('Price:')
        print(len(self.Price))

        
        for k in range(len(self.DataSet)):
            print('--------------------------------------------------')
            print('DataSet shape,',k)
            print(self.DataSet[k].shape)
            print('--------------------------------------------------')
            print('Return shape,',k)
            print(self.Return[k].shape)
            print('--------------------------------------------------')
            print('Price shape,',k)
            print(self.Price[k].shape)
        print('--------------------------------------------------')
        print('pred_return:')
        print(len(self.pred_return))
    def Plot_Poly_Double(self,Data1,Data2,Path):
        fig, ax = plt.subplots(figsize = (12,5))
        X=np.arange(0,len(Data1),1)
        ax.plot(X,Data1,
                color = 'blue',
                alpha = 0.4,
                label = "data1"
               )
        ax.plot(X,Data2,
                color = 'blue',
                alpha = 0.4,
                label = "data2"
               )
        ax.set_xlabel("Time")
        ax.set_ylabel("Return")
        ax.set_title("The returns ")
        ax.legend()
        ax.grid()
    def plot_cmp(self,BaseLine):
        '''
        Function: plot the baseline's data and portfolio's data.
        Input: 
        (string)BaseLine: the code of the baseline asset.
        '''
        cmp=self.get_baseline_initial(BaseLine)
        if len(self.pred_return)==0:
            print('Please get the pred_return first!')
        else:
            self.Plot_Poly_Double(self.pred_return,cmp,Path='.\Return.png')
    ########################## Part 5 : main methods #############################
    def cal_pf_return(self):
        '''
        Function: calculate the portfolio's predicted return based on time-interval.
        Output: 
        (DataFrame)pred_return: A dataframe about the return.
        Note: use the raw sequence, in this way, the time interval is equal.
        '''
        rst=[]
        row,col=self.Return[0].shape
        index_list=self.Return[0].index.to_list()
        for i in range(row):
            sum_rst=0
            
            for j in range(len(self.Return)):
                sum_rst=sum_rst+self.Return[j].loc[index_list[i]].values[0]*self.Weight[j]
            rst.append(sum_rst)
        print('rst:',rst)
        pred_return=pd.DataFrame(rst,columns=['pred_return'],index=index_list)
        self.pred_return=copy.deepcopy(pred_return)
    def evaluate(self,BaseLine):
        count=0
        sum_pre_return=0
        cmp=self.get_baseline_initial(BaseLine)
        index_list=self.Return[0].index.to_list()
        print(cmp)
        print(self.pred_return)
        for i in range(len(self.pred_return)):
            if self.pred_return.loc[index_list[i]].values[0]>cmp.loc[index_list[i]].values[0]:
                count+=1
                sum_pre_return=self.pred_return.loc[index_list[i]].values[0]-cmp.loc[index_list[i]].values[0]
        Start_Time=self.Price[0].index.to_list()[0]
        End_Time=self.Price[0].index.to_list()[-1]
        print('--------------------------------------------------')
        print('start:',Start_Time,'end: ',End_Time)
        print('accumulated extra return:',sum_pre_return)
        print('The number of days that we win: ',count)
        print('The ratio: ',float(count)/len(self.pred_return))
        