import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
import random
class SingleDataClean:
    
    
    '''
    self variables:
    (DataFrame)self.Data: the DataSet,a dataset of a single data
    Note:
    This class is mainly to do data cleaning work.
    Most of the widely used methods are contained.
    The class is adaptive,so the API mainly use the DataFrame,
    and we only update the data.
    We will consider the consistency like the shape of matrix
    in other places, eg.whenever we use some methods to retain
    important factors, we modify FactorList in MyPortfolio at
    the same time.
    All methods are private,but the methods are adaptble.
    
    A bit of redundancy, but safe.
    The most important: returned list of index
    NOTE: DON'T reset the index!
    '''
    ##############################  Part 1 initialize  ############################
    def __init__(self,Data):
        self.Data=copy.deepcopy(Data)
        self.Data.dropna()
    ############################    Part 2: data fetch   ##########################
    def get_Data(self):
        return self.Data
    #########################    Part 3: data describe   ##########################
    def describe(self):
        print('------------------------')
        print('Cleaner: data:')
        print(self.Data)
    ######################  Part 4 standarlize and de-extremum  ###################
    def get_median_df(self,index):
        '''
        Function: return the median of a list.
        Input: 
        (int) index:the position
        Output: 
        (float)List[int(len(df)/2)]:the median 
        '''
        df=copy.deepcopy(self.Data.iloc[:,index])
        List=[]
        for i in range(len(df)):
            List.append(df.iloc[i])
        List.sort()
        return List[int(len(df)/2)]
    def get_median(self,series):
        '''
        Function: return the median of a list.
        Input: 
        (int) index:the position
        Output: 
        (float)List[int(len(df)/2)]:the median 
        '''
        df=copy.deepcopy(series)
        List=[]
        for i in range(len(df)):
            List.append(df.iloc[i])
        List.sort()
        return List[int(len(df)/2)]
    def median_de_extremum(self,n,Is_Std=1):
        '''
        Function: get the targeted self-column de-extremumed
        Input:
        (int)n: the bound that determine the tolerrance of the extremum
        Is_Std: 1: do the standarlize, do nothing otherwise
        we execute the processes:
        1.use the median method to eliminate the extremum
        2.standarlize the data
        3.we process in the order of the columns
        '''
        row,col=self.Data.shape
        column=self.Data.columns.to_list()
        for i in range (col):
            xm=self.get_median_df(i)
            xmi=self.Data.apply(lambda x:abs(x[column[i]]-xm),axis=1)
            D_MAD=self.get_median(xmi)
            for j in range(len(self.Data[column[i]])):
                if xm+n*D_MAD<self.Data.iloc[j,i]:
                    self.Data.iloc[j,i]=xm+n*D_MAD
                if xm-n*D_MAD>self.Data.iloc[j,i]:
                    self.Data.iloc[j,i]=xm-n*D_MAD
            
        if Is_Std==1:
            self.stdl()
    def stdl(self):
        '''
        Note:
        This part is to standarlize the whole DataFrame.
        '''
        row,col=self.Data.shape
        column=self.Data.columns.to_list()
        for i in range(col):
            sigma=self.Data[column[i]].std()
            mu=self.Data[column[i]].mean()
            self.Data[column[i]]=self.Data.apply(lambda x: (x[column[i]]-mu)/sigma,axis=1)
    #####################################   Part 5 data_sampling   ######################################
    def resample_volume_share(self,threshold,Volume):
        '''
        Function: resample data based on their volume
        Input：
        (DataFrame)Volume: the volume of the asset
        (float)threshold: the threshold , when the accumulated
        volume of several rows' volume excedd it, the accumulated
        volume will be reset to 0
        Output:
        (list)self.Data.index.to_list(): the retained datas' indexes.
        '''
        '''
        Note:
        We resample the data mainly by the share.
        If the share bigger than threshold, we add
        up them and use the updated time stamp.
        threshold: the minimum volume of sampling
        (DataFrame/series)Volume: the volume
        (DataFrame/series)Return: the Return
        The index should be all same.
        For other parameters:
        Most of our factors won't be changed in a short period time.
        And most of them are not the flow.
        So only the volume should be add up and others will be updated to the lateset one.
        We don't need to consider different assets' number of tuples,
        only the Factors should be the same.
        Return the retaining indexes' set
        Operations are base on index.More safer.
        We modify the data.And return the index.
        '''
        volume=copy.deepcopy(Volume)
        row,col=self.Data.shape
        volume_sum=0
        start_index=0
        index_list=self.Data.index.to_list()
        for i in range(row):
            volume_sum+=volume[index_list(i)]
            if volume_sum>threshold:
                '''
                Then delete the tuples.
                Drop through index.
                '''
                self.Data=self.Data.drop(index_list[start_index:i])
                volume[index_list(i)]=volume_sum
                volume_sum=0
                start_index=i+1
        '''
        Then drop the tuples that are unsatisfied.
        '''
        if start_index<row:
            self.Data=self.Data.iloc[:start_index]
        return self.Data.index.to_list()
    def resample_volume_money(self,threshold,Volume):
        '''
        Function: resample data based on their volume
        Input：
        (DataFrame)Volume: the volume of the asset
        (float)threshold: the threshold , when the accumulated
        volume of several rows' volume excedd it, the accumulated
        volume will be reset to 0
        Output:
        (list)self.Data.index.to_list(): the retained datas' indexes.
        '''
        '''
        Note:
        We resample the data mainly by the money.
        
        The function is the same to the Resample_volume_share.
        But for consistency,we list this function indivially.
        
        If the share bigger than threshold, we add
        up them and use the updated time stamp.
        threshold: the minimum volume of sampling
        (DataFrame/series)Volume: the volume
        (DataFrame/series)Return: the Return
        The index should be all same.
        For other parameters:
        Most of our factors won't be changed in a short period time,and most of them are
        not the flow.
        The volume should has the same index like Return and DataSet[i]
        So only the volume should be add up and others will be updated to the lateset one.
        We don't need to consider different assets' number of tuples,
        only the Factors should be the same.
        '''
        volume=copy.deepcopy(Volume)
        row,col=self.Data.shape
        volume_sum=0
        start_index=0
        index_list=self.Data.index.to_list()
        for i in range(row):
            volume_sum+=volume[index_list(i)]
            if volume_sum>threshold:
                '''
                Then delete the tuples.
                Drop through index.
                '''
                self.Data=self.Data.drop(index_list[start_index:i])
                volume_sum=0
                start_index=i+1
        '''
        Then drop the tuples that are unsatisfied.
        '''
        if start_index<row:
            self.Data=self.Data.drop(index_list[start_index:])
        return self.Data.index.to_list()
    def non_balance_resample(self,threshold,delta,Open):
        '''
        Function: resample data based on their open
        Input：
        (DataFrame)Open: the open price of the asset
        (float)threshold: the threshold , when the signal
        variables of several rows excedd it, the signal
        variable will be reset to 0
        (float)delta: the threshold , when the difference 
        of open[i] and open[i+1] is bigger than it, the 
        signal variable will + 1
        
        Output:
        (list)self.Data.index.to_list(): the retained datas' indexes.
        '''
        
        '''
        Note:
        Based on considering the balance of the data.
        eg.when the series is fluctating on a base level, we sample less;
        and when the series is going up constantly, it represents more information,
        so we will sample more frequently.
        '''
    
        theta=0
        row,col=self.Data.shape
        start_index=0
        index_list=self.Data.index.to_list()
        for i in range(1,row):
            if abs(Open[index_list[i]]-Open[index_list[i-1]])<delta:
                theta-=1
            else :
                theta=theta+1
            if theta>threshold:
                '''
                sampling
                '''
                self.Data=self.Data.drop(index_list[start_index:i])
                start_index=i+1
                theta=0
        if start_index<row:
            self.Data=self.Data.drop(index_list[start_index:])
        return self.Data.index.to_list()
    def non_balance_resample_volume(self,threshold,delta,volume,Open):
        
        '''
        Function: resample data based on their volume and open
        Input：
        (DataFrame)Open: the open price of the asset
        (float)threshold: the threshold , when the signal
        variables of several rows excedd it, the signal
        variable will be reset to 0
        (float)delta: the threshold , when the difference 
        of volume[i]*open[i] and volume[i+1]*open[i+1]
        is bigger than it, the signal variable will + 1
        
        Output:
        (list)self.Data.index.to_list(): the retained datas' indexes.
        '''
        
        '''
        Note:
        Based on considering the balance of the data.
        eg.when the series is fluctating on a base level, we sample less;
        and when the series is going up constantly, it represents more information,
        so we will sample more frequently.
        '''
        theta=0
        row,col=self.Data.shape
        start_index=0
        index_list=self.Data.index.to_list()
        for i in range(1,row):
            if abs(volume[index_list[i]]*Open[index_list[i]]-volume[index_list[i]]*Open[index_list[i-1]])<delta:
                theta-=1
            else :
                theta=theta+1
            if theta>threshold:
                '''
                sampling
                '''
                self.Data=self.Data.drop(index_list[start_index:i])
                start_index=i+1
                theta=0
        if start_index<row:
            self.Data=self.Data.drop(index_list[start_index:])
        return self.Data.index.to_list()
def sum_volume(index_list,volume):
    '''
    Function: based on the index_list to select the rows,
    and add up the volume of a period.
    Input:
    (list)index_list:indexes of the raw data
    (DataFrame)volume: the volume of the assets
    Output:
    (DataFrame)volume: the processed volume
    Note:
    When set the volume,must use it!!
    This part is mainly to sum up the volume during the period 
    between two adjoining indexes in the index_list. 
    '''
    POS=0
    INDEX=volume.index.to_list()
    volume_sum=0
    
    for i in range(len(INDEX)):
        if POS==len(index_list):
            break
        if INDEX[i] !=index_list[POS]:
            volume_sum+=volume[INDEX[i]]
            volume=volume.drop(INDEX[i])
        else :
            volume.loc[INDEX[i]]+=volume_sum
            volume_sum=0
            POS+=1
    return volume
            
######################################################################
#################   Features identifying public  #####################
def cusum_sample_data(Data,index,h):
    '''
    Function: sample the abnormal points that 
    exceed the bounds.
    Input
    (DataFrame)Data: the data to be processed
    (int)index: which col to sample
    (float)h: the bounds
    '''
    '''
    Note:
    (DataFrame)Data: the Data needed to be processed
    (int)index: which column to observe
    (float)h:threshold,h should be positive
    
    Return two DataFrame: positive/run-up events and negative/run-down events.
    That's because we want the data passed through the API be DataFrame or index.
    '''
    SumH=0
    SumL=0
    index_list_high=[]
    index_list_low=[]
    row,col=Data.shape
    column=Data.columns.to_list()
    for i in range(1,row):
        SumH+=Data.iloc[i,index]-Data.iloc[i-1,index]
        if SumH<0:
            SumH=0
        SumL+=Data.iloc[i,index]-Data.iloc[i-1,index]
        if SumL>0:
            SumL=0
        if SumH>h:
            index_list_high.append(i)
            SumH=0
        if -SumL>h:
            SumL=0
            index_list_low.append(i)
    
    index_list_high=pd.DataFrame(index_list_high,columns=column,index=Data.index.values.to_list())
    index_list_low=pd.DataFrame(index_list_low,columns=column,index=Data.index.values.to_list())
    return index_list_high,index_list_low
def label_generate_3parclose(Data,HLevel,LLevel,TLength,start_time,end_time):
    '''
    Note:
    (DataFrame)Data:the data to be identified
    (float)HLevel:high price threshold
    (float)LLevel:low price threshold
    (int)start_time:observe start time
    (int)end_time:observe end time
    output:
    A DataFrame of label.
    The element is like [1,1,1]
    Corresponding to high,low,wide
    If the value bigger than the high-line, then 1;
    If the value smaller than the low-line, then 1;
    If the value exceed the time-end, then 1;
    '''
    row,col=Data.shape
    Index_list=Data.index.to_list()
    label=[]
    index_list=[]
    '''
    Record the abnormal conditions and their corresponding indexes.
    '''
    MyTupleH=[]
    MyTupleL=[]
    MyTupleT=[]
    for i in range(row):
        flag=0
        if Data.iloc[i]>HLevel:
            MyTupleH.append(1)
            flag=1
        else:
            MyTupleH.append(0)
        if Data.iloc[i]<LLevel:
            MyTupleL.append(1)
            flag=1
        else:
            MyTupleH.append(0)
        if i>int(end_time-start_time):
            MyTupleT.append(1)
            flag=1
        else:
            MyTupleT.append(0)
        if flag==1:
            index_list.append(Data.iloc[i].index.values)
    result=pd.DataFrame([MyTupleH,MyTupleL,MyTupleT],columns=['MyTupleH','MyTupleL','MyTupleT'],index=Index_list)
    return result

def transaction_sample(Data,Label):
    '''
    Function: overcome the problem of non-IID
    Input: 
    (DataFrame)Data:raw data, for its indexes
    (DataFrame)Label: the labels that are sampled 
    in other places,index should be timeseries;
    each column should be the labels.
    Output: 
    the result DataFrame with the index of raw data.
    The labels: like that are generated in the function label_generate_3parclose()
    are the signs of the abnormal signals.
    So,even though every tuple of our pre-processed raw sequentials
    are valid in every position, the signs or labels may not.
    eg. in the label_generate_3parclose() we consider three labels,
    not all of the labels are 1 or valid.
    And we do the samplings, iterating from head to toe,we take a label from the labels and update the unique-value
    eg. in T, the tuple of label is (0,0,1,1).Then we don't know which one of the two labels we should give them. 
    So we choose based on probability,.
    
    By the way, above are my confusions during reading the Advances in Financial Machine Learning
   
    pre-process the Label.For safety.
    Get the indexes.
    '''
    Data_Concat=Data.join(Label,how='left')
    Data_Concat.fillna(0)
    Uni_Label=Data_Concat[Label.columns.to_list()]
    Uni_Mat=copy.deepcopy(Uni_Label)
    '''
    Calculate.
    '''
    valid_count=[]
    '''
    initialize
    '''
    row,col=Uni_Label.shape
    for i in range(col):
        count=0
        for j in range(row):
            dem=0
            for k in range(col):
                dem+=Uni_Label.iloc[j,k]
            if dem!=0:
                Uni_Mat.iloc[j,i]=float(Uni_Label.iloc[j,i])/dem
                count+=1
        valid_count.append(count)
    '''
    Sampling and updating
    First we copy Uni_Label
    Then we ranfomly get a j;
    Then we modify Uni_Label,and then we modify Uni_Mat.
    '''
    Result=copy.deepcopy(Uni_Label)
    column=Uni_Mat.columns.to_list()
    for i in range(row):
        '''
        randomly get a j
        '''
        sum_prob=0
        for j in range(col):
            sum_prob+=float(Uni_Mat[column[j]].sum())/valid_count[j]
        flag=random.random()
        sum_norm=0
        index_flag=0
        for j in range(col):
            sum_norm+=float(Uni_Mat[column[j]].sum())/valid_count[j]
            if float(sum_norm)/sum_prob>flag:
                index_flag=j
                break
        for j in range(col):
            '''
            Update the Uni_Mat.
            '''
            for k in range(row):
                if Uni_Mat[column[index_flag]]!=0:
                    Uni_Mat[column[j]]=float(Uni_Mat[column[j]])/2
        for j in range(col):
            '''
            Update the Result.
            '''
            Result.iloc[i,j]=0
            if j==index_flag:
                Result.iloc[i,j]=1
    return Result