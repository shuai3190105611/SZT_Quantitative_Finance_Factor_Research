# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:53:31 2022

@author: ZTSHUAI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
class Optimizer:
    ########################  Part 1 : initialize   #######################
    '''
    self variables
    (list)self.Retain: list of dataframes of returns
    (DataFrame)self.RiskExposeDa: the dataframe of risk exposure
    (DataFrame)self.Weight: the dataframe of weight
    '''
    def __init__(
            self,
            Return,
            RiskExposeDA
            ):
        '''
        Input:
        (list)Return: a list of returns of all assets. 
        Note:
        The class is to generate the weight,
        So it's no need to input a weight.
        
        The Returns are raw sequences, we need to do
        some pre-precesses on it.
        '''
        self.Return=copy.deepcopy(Return)
        self.RiskExposeDA=copy.deepcopy(RiskExposeDA)
        self.Weight=[]
    ########################  Part 2: data fetch ##########################
    def get_weight(self):
        return copy.deepcopy(self.Weight)
    ########################  Part 3: data set   ##########################    
    def set_Return(self,Return):   
        self.Return=copy.deepcopy(Return)
    def set_RiskExpose(self,RiskExposeDA):
        self.RiskExposeDA=RiskExposeDA
    #######################  Part 4: description   ########################
    def describe(self):
        print('--------------------DESCRIBE----------------------')
        print('--------------------------------------------------')
        print('Weight:')
        print(self.Weight)
    ######################  Part 5: basic operation #######################
    def cal_arith_avg(self):
        '''
        Function: calculate the average return of the assets.
        Output:
        (list)avg_return
        '''
        avg_return=[]
        for i in range(len(self.Return)):
            sum_avg=0
            index_list=self.Return[i].index.to_list()
            for j in range(len(self.Return[i])):
                sum_avg+=self.Return[i].loc[index_list[j]].values[0]
            sum_avg=float(sum_avg)/len(self.Return[i])
            avg_return.append(sum_avg)
        return avg_return
    def cal_geo_avg(self):
        '''
        Function: calculate the average return of the assets.
        Output:
        (list)avg_return
        '''
        avg_return=[]
        for i in range(len(self.Return)):
            sum_avg=1
            index_list=self.Return[i].index.to_list()
            for j in range(len(self.Return[i])):
                sum_avg=pow(sum_avg,(j-1)/(j))*pow(self.Return[i].loc[index_list[j]].values[0],1.0/(j+1))
            avg_return.append(sum_avg)
        return avg_return
    ######################  Part 6: main operation #######################
    def cal_weight_1d(self,Max_Total,Admit_Short=True,method=1):
        '''
        Function: calculate the optimal weight,
        based on the linear programming.
        Input:
        (bool)AdmitShort: 1 if short sellinf is permitted; 0 otherwise.
        (int)method: 1: arith-avg to get the average
        0: geo_avg to get the average
        Output:
        (DataFrame)Weight: the weight of the asset.
        '''
        if method==1:
            pro_return=self.cal_arith_avg()
        if method==0:
            pro_return=self.cal_geo_avg()
        if Admit_Short==True:
            #################################
            ##### linear programming  #######
            #################################
            
            ###### transfer the data ######## 
            A_ub=[]
            B_ub=[1]
            c=[]
            A_eq=[]
            B_eq=[]
            row,col=self.RiskExposeDA.shape
            print('row=',row,'col=',col)
            for j in range(col):
               
               temp=[]
               for i in range(len(self.Return)):
                   temp.append(self.RiskExposeDA.iloc[i,j])
               A_eq.append(temp)
               B_eq.append(0)
            temp=[]
            for i in range(len(self.Return)):
                c.append(-pro_return[j])
                temp.append(1)
            A_ub.append(temp)
            res = linprog(c, A_ub=A_ub,b_ub=B_ub,A_eq=A_eq, b_eq=B_eq)
            self.Weight=copy.deepcopy(res.x.tolist())
        else :
            #################################
            ##### linear programming  #######
            #################################
            
            ###### transfer the data ######## 
            ###### need a positive bound ####
            A_ub=[]
            B_ub=[1]
            c=[]
            Bound=[]
            A_eq=[]
            B_eq=[]
            row,col=self.RiskExposeDA.shape
            print('row=',row,'col=',col)
            for j in range(col):
               
               temp=[]
               for i in range(len(self.Return)):
                   temp.append(self.RiskExposeDA.iloc[i,j])
               A_eq.append(temp)
               B_eq.append(0)
            temp=[]
            for i in range(len(self.Return)):
                c.append(-pro_return[j])
                temp.append(1)
                Bound.append((0,None))
            A_ub.append(temp)
            res = linprog(c, A_ub=A_ub,b_ub=B_ub,A_eq=A_eq, b_eq=B_eq)
            self.Weight=copy.deepcopy(res.x.tolist())
            

    