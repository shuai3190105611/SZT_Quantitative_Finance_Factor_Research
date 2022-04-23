# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:07:07 2022

@author: ZTSHUAI
"""

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy

class visualizer:
    def __init__(
        self,
        dataset,
        name="Visualizer 0"
                ):
        '''
        Input:
        (string)name: name of the graph
        (dataframe)dataset: the dataset
        '''
        self.name=name
        self.dataset=deepcopy(dataset)
        self.config_indexs=['upper_bound','lower_bound','max_point_num','interval']
        mat=np.ones((len(self.config_indexs),len(self.dataset.columns)))
        self.config=pd.DataFrame(mat,columns=self.dataset.columns,index=self.config_indexs)
        #set the default value
        row,col=self.dataset.shape
        for i in self.dataset.columns:
            self.config.loc['max_point_num',i]=row
            self.config.loc['interval',i]=1
            if self.dataset[i].dtype != "int64" and self.dataset[i].dtype != "float64":
                continue
            upper=self.dataset[i].max()
            lower=self.dataset[i].min()
            self.config.loc['upper_bound',i]=upper
            self.config.loc['lower_bound',i]=lower
            
    def get_dataset(self):
        return deepcopy(self.dataset)
    def get_shape(self):
        row,col=self.dataset.shape
        return row,col
    def set_dataset(self,dataset):
        self.dataset=deepcopy(dataset)
    def print_data(self):
        print("Name: ",self.name)
        print("Dataset:")
        print(self.dataset)
    def print_shape(self):
        print("Name: ",self.name)
        print("Dataset:")
        print(self.dataset.shape)
    def describe_data(self):
        self.dataset.describe()
    def config_plot(self,Factor,upper_bound,lower_bound,max_point_num,interval):
        '''
        Function: set the configuration.
        Input: 
        (float)upper_bound: size of the
        (string)Factor: which factor to configurate
        (float)upper_bound: value of upper bound
        (float)lower_bound: value of lower bound
        (int)max_point_num: maximum number of points
        (int)interval: interval for sampling
        '''
        self.config.loc['upper_bound',Factor]=upper_bound
        self.config.loc['lower_bound',Factor]=lower_bound
        self.config.loc['max_point_num',Factor]=max_point_num
        self.config.loc['interval',Factor]=interval
        
    def select_label(self):
        '''
        Function: select the data
        Output: the labels of the selected data
        '''
        labels=[]
        count=0
        i=0
        index_list=self.dataset.index.to_list()
        while i < len(index_list) and count < self.config.loc['max_point_num'].min():
            #run through the rows
            flag=1
            for j in self.dataset.columns:
                #for each factors test if it's satisfied
                if self.dataset[j].dtype != "int64" and self.dataset[j].dtype != "float64":
                    continue
                if self.dataset.loc[index_list[i],j]>self.config.loc['upper_bound',j] or self.dataset.loc[index_list[i],j]<self.config.loc['lower_bound',j]:
                    flag=0
                    break
            
            #use the max of the interval
            if flag==1:
                count+=1
                labels.append(index_list[i])
            i+=int(self.config.loc['interval'].max())
        return labels
    def coherance(self,Factor1,Factor2,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (string)Factor1: name of the first factor
        (string)Factor2: name of the second factor
        Output:
        pictures
        '''
        # Use two factors in the dataset
        #check the correctness
        if Factor1 not in self.dataset.columns or Factor2 not in self.dataset.columns or label not in self.dataset.columns:
            print("Error! Factors not valid!")
        else:
            #set the label
            selected_label=self.select_label()
            sub_dataset=self.dataset.loc[selected_label]
            #set the limitations of the variables
            row,col=self.get_shape()
            max_y=max(sub_dataset[Factor1].max(),sub_dataset[Factor2].max())
            min_y=min(sub_dataset[Factor1].min(),sub_dataset[Factor2].min())
            s1=np.array(sub_dataset[Factor1])
            s2=np.array(sub_dataset[Factor2])
            
            #t=np.array(sub_dataset[label])
            # transform to the time type
            t = np.array([datetime.datetime.strptime(d,"%Y-%m-%d").date()
                     for d in sub_dataset[label]])
            
            plt.style.use('seaborn-darkgrid')
            fig, axs = plt.subplots(2, 1)
            
            # we sub 1 to avoid the case: the last tuple is (2020-xx-xx,xx,xx) and the value is 0
            
            #remenber to add label="xxx"
            
            
            axs[0].plot(t[:len(t)-1], s1[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "green",alpha = 0.5
                       ,label=Factor1
                       )
            axs[0].plot(t[:len(t)-1], s2[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "red",alpha = 0.5
                       ,label=Factor2
                       )
            plt.figure(figsize=(12, 5))
            axs[0].legend()
            axs[0].set_xlim(t[0], t[int(min(self.config.loc['max_point_num',Factor1],self.config.loc['max_point_num',Factor2]))-2])
            axs[0].set_ylim(min_y, max_y)
            
            #set the labels
            axs[0].set_xlabel(label)
            axs[0].set_ylabel(Factor1+' and '+Factor2)
            axs[0].grid(True)
            
            cxy, f = axs[1].cohere(s1, s2, min(int(len(selected_label)/2-1),64), 1. / 0.1
                                  ,marker = "+",markersize = 4,color= "blue",alpha = 0.5
                                  ,label="coherance"
                                  )
            axs[1].set_ylabel('coherence')
            
            fig.tight_layout()
            axs[1].legend()
            fig.savefig('.\\'+Factor1+' '+Factor2+'.jpg')
            
            plt.show()
    def heatmap(self,matrix,Name):
        # we need a n*n matrix
        '''
        Function: to plot the heat map based on matrix(DataFrame exactly)
        Input:
        (DataFrame)matrix: the matrix we want to plot
        (string)Name: the name of the heat map
        Output:
        pictures
        '''
        plt.figure(dpi=120)
        sns.heatmap(data=matrix,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签               
             )
        plt.title('Heat map for '+ Name)
        plt.savefig('.\\'+Name+' heat map'+'.jpg')
        plt.show()
        plt.clf()
    def cor_total_plot(self):
        #calculate a matrix and then plot
        #first we calculate the cor
        '''
        Function: call the function: heatmap to draw a heat map
        '''
        
        '''
        row,col=self.dataset.shape
        matrix=np.ones((col,col))
        for i in range(len(self.dataset.columns)):
            for j in range(len(self.dataset.columns)):
                #calculate the cor
                matrix[i,j]=self.dataset[self.dataset.columns[i]].corr(self.dataset[self.dataset.columns[j]])
        '''
        #pandas's function can help us exclude the non-numeric factors' obstruction
        matrix=self.dataset.corr()
        print(matrix)
        self.heatmap(matrix,'coefficient')
    def single(self,Factor,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (string)Factor: name of the factor
        Output:
        pictures
        '''
        if Factor not in self.dataset.columns or label not in self.dataset.columns:
            print("Error! Factors not valid!")
        else:
            selected_label=self.select_label()
            sub_dataset=self.dataset.loc[selected_label]
            #set the limitations of the variables
            row,col=self.get_shape()
            max_y=sub_dataset[Factor].max()
            min_y=sub_dataset[Factor].min()
            s1=np.array(sub_dataset[Factor])
           
            # transform to the time type
            t = np.array([datetime.datetime.strptime(d,"%Y-%m-%d").date()
                     for d in sub_dataset[label]])
            
            plt.style.use('seaborn-darkgrid')
            plt.plot(t[:len(t)-1], s1[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "blue",alpha = 0.5
                       ,label="Open"
                       )
            #plt.set_xlim(t[0], t[int(self.config.loc['max_point_num',Factor])-2])
            #plt.set_ylim(min_y, max_y)
            plt.title("Single")
            plt.ylabel(Factor)
            plt.xlabel("time")
            plt.legend()
            plt.savefig('.\\single: '+Factor+'.jpg')
            plt.show()
    def check_in(self,FactorList):
        flag=1
        for i in FactorList:
            if i not in self.dataset.columns:
                flag=0
                break
        return flag
    def multi(self,FactorList,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (list)Factor: list of names of the factors
        Output:
        pictures
        '''
        flag=self.check_in(FactorList)
        if flag==0:
            print("Error! Factors not valid!")
        else:
            #set the label
            selected_label=self.select_label()
            sub_dataset=self.dataset.loc[selected_label]
            #set the limitations of the variables
            row,col=self.get_shape()
            #use list to store the factors
            max_bound=0
            min_bound=0
            count=0
            s_list=[]
            for i in FactorList:
                if count==0:
                    max_bound=sub_dataset[i].max()
                    min_bound=sub_dataset[i].min()
                else:
                    if max_bound<sub_dataset[i].max():
                        max_bound=sub_dataset[i].max()
                    if min_bound>sub_dataset[i].min():
                        min_bound=sub_dataset[i].min()
                s_list.append(np.array(sub_dataset[i]))
                count+=1
            
            # transform to the time type
            t = np.array([datetime.datetime.strptime(d,"%Y-%m-%d").date()
                     for d in sub_dataset[label]])
            
            plt.style.use('seaborn-darkgrid')
            for i in range(len(FactorList)):
                plt.plot(t[:len(t)-1], s_list[i][:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= (0.5, 0.9*i/len(FactorList), 0.5),alpha = 0.5
                       ,label=FactorList[i]
                       )
                
            plt.title("Multi")
            plt.ylabel("Factors")
            plt.xlabel("time")
            plt.legend()
            plt.savefig('.\\Multi.jpg')     
            plt.show()
    def show_stat(self,FactorList):
        '''
        Function:
        To show the statistic features of the factors
        Input:
        (list)FactorList:the list of factors
        Output: the pictures
        '''
        flag=self.check_in(FactorList)
        if flag==0:
            print("Error! Factors not valid!")
        else:
            #set the label
            selected_label=self.select_label()
            sub_dataset=self.dataset.loc[selected_label]
            row,col=self.get_shape()
            #generate the new df about the
            col_list=['max','min','avg','factor']
            stat_df=pd.DataFrame(columns=col_list)
            for i in FactorList:
                stat_df.loc[i]=[ self.dataset[i].max()
                                ,self.dataset[i].min()
                                ,self.dataset[i].mean()
                                ,i
                               ]
            
            #use the melt operations
            print("Description:")
            print(stat_df.describe())
            stat_df=stat_df.melt(id_vars=['factor'])
            #print(stat_df)
            plt.style.use('seaborn-darkgrid')
            #value: the value of the stats of factors, variable: the names of the melted ,eg.max min avg
            #factor: the name of the factors
            #ci to set the error,sd:default:95%
            #set to note the style of the color
            sns.barplot(x="factor",y="value",data=stat_df,hue="variable",ci="sd",capsize=.2,palette="Set3")
            
            
            
            
            plt.title("stats")
            plt.ylabel("values")
            plt.xlabel("Factors")
            plt.legend()
            plt.savefig('.\\show_stat.jpg')     
            plt.show()     
            plt.clf()
    def subline(self,Factor1,Factor2,label):
        '''
        Function: show the marginal of two factors and draw the graph
        Input:
        (string)Factor1: name of the first factor
        (string)Factor2: name of the second factor
        (string)label: the x label of the data
        Output:
        pictures
        '''
        # Use two factors in the dataset
        #check the correctness
        if Factor1 not in self.dataset.columns or Factor2 not in self.dataset.columns or label not in self.dataset.columns:
            print("Error! Factors not valid!")
        else:
            #set the label
            selected_label=self.select_label()
            sub_dataset=self.dataset.loc[selected_label]
            #set the limitations of the variables
            row,col=self.get_shape()
            max_y=max(sub_dataset[Factor1].max(),sub_dataset[Factor2].max())
            min_y=min(sub_dataset[Factor1].min(),sub_dataset[Factor2].min())
            s1=np.array(sub_dataset[Factor1])
            s2=np.array(sub_dataset[Factor2])
            s_sub=(s1-s2)
            print(s_sub)
            #t=np.array(sub_dataset[label])
            # transform to the time type
            t = np.array([datetime.datetime.strptime(d,"%Y-%m-%d").date()
                     for d in sub_dataset[label]])
            
            plt.style.use('seaborn-darkgrid')
            fig, axs = plt.subplots(2, 1)
            
            # we sub 1 to avoid the case: the last tuple is (2020-xx-xx,xx,xx) and the value is 0
            
            #remenber to add label="xxx"
            
            
            axs[0].plot(t[:len(t)-1], s1[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "green",alpha = 0.5
                       ,label=Factor1
                       )
            axs[0].plot(t[:len(t)-1], s2[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "red",alpha = 0.5
                       ,label=Factor2
                       )
            axs[0].plot(t[:len(t)-1], s_sub[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "blue",alpha = 0.5
                       ,label='margin'
                       )
            plt.figure(figsize=(12, 5))
            axs[0].legend()
            #set_xlim is not needed
            axs[0].set_xlim(t[0], t[int(min(self.config.loc['max_point_num',Factor1],self.config.loc['max_point_num',Factor2]))-2])
            axs[0].set_ylim(-max_y, max_y)
            
            #set the labels
            axs[0].set_xlabel(label)
            axs[0].set_ylabel(Factor1+' and '+Factor2)
            axs[0].grid(True)
            #for the margin
            axs[1].plot(t[:len(t)-1], s_sub[:len(t)-1]
                       #,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12#these are settings
                       ,marker = "+",markersize = 4,color= "blue",alpha = 0.5
                       ,label='margin'
                       )
            plt.figure(figsize=(12, 5))
            axs[1].legend()
            axs[1].set_xlim(t[0], t[int(min(self.config.loc['max_point_num',Factor1],self.config.loc['max_point_num',Factor2]))-2])
            axs[1].set_ylim(-max_y, max_y)
            
            #set the labels
            axs[1].set_xlabel(label)
            axs[1].set_ylabel('margin')
            axs[1].grid(True)
            fig.savefig('.\\margin of '+Factor1+' '+Factor2+'.jpg')
            
            plt.show()
#Demo mainly process data and some operations on it
df=pd.read_csv("ADANIPORTS.csv")
Demo_v=visualizer(df,"Demo 0")
Demo_v.print_data()
Demo_v.config_plot('Open',upper_bound=1000,lower_bound=500,max_point_num=50,interval=10)
Demo_v.coherance('Open','High','Date')
Demo_v.single('Open','Date')
FactorList=['Open','Close','High','Low']
Demo_v.multi(FactorList,'Date')
Demo_v.cor_total_plot()
Demo_v.show_stat(FactorList)
Demo_v.subline('Open','High','Date')