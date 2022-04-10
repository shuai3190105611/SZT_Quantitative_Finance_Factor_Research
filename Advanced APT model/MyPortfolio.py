import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression
import DataClean as dc
class Portfolio:
    '''
    self variables:
    (list)self.FactorList: list of Factor's name
    (list)self.AssetList: list of Asset's code
    (list)self.DataSet: list of each asset's corresponding data
    (DataFrame)self.Weight: the weight of the asset
    (DataFrame)self.RiskExpose: the matrix of the risk exposure of each factor.The size don't need to be 
    modified.We get the column names and RiskExpose,then we index the data more conveniently.
    (list)self.Return: list of Assets' return
    (string)self.ReturnList:the name of the return
    (string)self.Name: the name of the portfolio
    (list)self.Price: list of DataFrame for:open,close,volume,fee(not used).The order should be this.
    (list)self.ImptList:the name of important factors.
    (list)self.IndexList:the list of names of each assets' index.
    The factor in DataSet may be same as someone in Price.
    '''
    
    
    '''
    Note:
    1.This class is the MAIN data structure of our project,
    which will be used in many other places.
    The data will be loaded from the financial databases,
    and you can modify the database and the APIs in the
    function: get_data_initial().
    '''
    
    '''
    2.Each MyPortfolio instance represents a asset portfolio,
    and the information including weights and data should be
    stored here.
    '''
    
    '''
    3.About Naming:
    AaaBbb: the name of the class
    Aaa_Bbb: the parameters passed between function
    AAABBB: macro definition
    aaa_bbb: for function
    '''
    
    '''
    4.Coding standards:
    a.use deepcopy to ensure the data of the MyPortfolio couldn't
    be modified in the wrong places.
    b.check the length and the dimension before use
    c.decrese the redundant code
    '''
    ##################  Part 1: Initialize   ##################
    def __init__(
        self,
        Factor_List,
        Asset_List,
        Return_List,
        Name,
        Price_List=['open','close','volume']
                ):
        
        
        '''
        Input:
        (list)Factor_List: list of Factor's name
        (list)Asset_List: list of Asset's code
        (list)DataSet: list of each asset's corresponding data
        (list)Return: list of Assets' return
        (string)ReturnList:the name of the return
        (string)Name: the name of the portfolio
        (list)Price: list of DataFrame for:open,close,volume,fee(not used).The order should be this.
        The factor in DataSet may be same as someone in Price.
        '''
        
        '''
        Note:
        Mainly to initialize
        the data fetching process will be done later
        in the data_fetch() function
        The returned index and ImptList should be used to select,or the selecting process won't be eventually done.
        '''
        self.FactorList=copy.deepcopy(Factor_List)
        self.AssetList=copy.deepcopy(Asset_List)
        self.Name=Name
        self.DataSet=[]
        self.Weight=[]
        self.RiskExposeDA=pd.DataFrame(columns=self.FactorList)
        self.Return=[]
        self.ReturnList=copy.deepcopy(Return_List)
        self.PriceList=copy.deepcopy(Price_List)
        self.Price=[]
        self.ImptList=[]
        self.IndexList=[]
    def get_data_initial(
        self,
        Start_Time,
        End_Time
                        ):
        '''
        Use the information from Asset and Factor,
        and the Start_Time and the End_Time to
        get the data from rqdata database.
        '''
        
        '''
        data:
        Start_Time: input start time
        End_Time: input end time
        '''
        import rqdatac
        from rqdatac import get_price,init,get_factor
        init(,)#your account
        for i in range(self.num_asset()):
            
            df=get_factor(self.AssetList[i],self.FactorList,start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            self.DataSet.append(df)
        for i in range(self.num_asset()):
            df=get_factor(self.AssetList[i],self.ReturnList,start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            self.Return.append(df)
        for i in range(self.num_asset()):

            df=get_price(self.AssetList[i],start_date=Start_Time,end_date=End_Time)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            df=df[self.PriceList]
            self.Price.append(df)
        for i in range(self.num_asset()):
            temp=[]
            self.IndexList.append(temp)
        for i in range(self.num_factor()):
            temp=[]
            self.ImptList.append(temp)
        print('Fetching success! The number of factor is:', self.num_factor)
    ##################  Part 2: data fetch   ##################
    '''
    Function: get the target data
    Input:
    data
    (int)index: to index the position
    '''
    def num_asset(self):
        return len(self.AssetList)
    def num_factor(self):
        return len(self.FactorList)
    def get_weight(self):
        return copy.deepcopy(self.Weight)
    def get_dataset(self):
        return copy.deepcopy(self.DataSet)
    def get_dataset_single(self, index):
        return copy.deepcopy(self.DataSet[index])
    def get_asset_list(self):
        return copy.deepcopy(self.AssetList)
    def get_factor_list(self):
        return copy.deepcopy(self.FactorList)
    def get_return(self):
        return copy.deepcopy(self.Return)
    def get_return_single(self,index):
        return copy.deepcopy(self.Return[index])
    def get_price(self):
        return copy.deepcopy(self.Price)
    def get_price_single(self,index):
        return copy.deepcopy(self.Price[index])
    def get_riskexposeDA(self):
        return copy.deepcopy(self.RiskExposeDA)
    def get_indexlist(self,k):
        return copy.deepcopy(self.IndexList[k])
    def get_imptlist(self):
        return copy.deepcopy(self.ImptList)
    def get_name(self):
        return copy.deepcopy(self.Name)
    ########################### Part 3 : data set   ###############################
    '''
    Function: set the target data
    Input:
    data
    (int)index: to index the position
    '''
    def set_name(self,Name):
        self.Name=copy.deepcopy(Name)
    def set_indexlist(self,k,index_list):
        self.IndexList[k]=copy.deepcopy(index_list)
    def set_imptlist(self,impt_list):
        self.ImptList=copy.deepcopy(impt_list)
    def set_weight(self,Weight):
        self.Weight=copy.deepcopy(Weight)
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
    def set_riskexposeDA(self,RiskExposeDA):
        self.RiskExposeDA=copy.deepcopy(RiskExposeDA)  
    ######################### Part 4 : data describe #############################
    def describe(self):
        '''
        print the information about the data
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
        print('--------------------------------------------------')
        print('ImptList:')
        print(self.ImptList)
        print('--------------------------------------------------')
        print('IndexList')
        print(self.IndexList)
    def describe_shape(self):
        '''
        print the information about the data
        '''
        print('--------------------DESCRIBE----------------------')
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
        print('--------------------------------------------------')
        print('ImptList:')
        print(len(self.ImptList))
        print('--------------------------------------------------')
        print('IndexList')
        print(len(self.IndexList))
        
        
        
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
    ######################    PART 5  Basic Operations   ########################
    
    def Plot_Bar(self,Data,Path):
        '''
        Function: plot a bar grah
        Input:
        data: the data to be ploted
        path: the path to store the picture
        '''
        x=np.linspace(0,len(Data),len(Data))
        colors = []
        for _ in range(int(len(Data) / 2)):
            colors.append([_ / int(len(Data) / 2), 0.5, 0.5])
        colors = colors + colors[::-1]
        x_tick = list(map(lambda num: "" if num % 10 != 0 else num, x))
        plt.figure(figsize=(300, 100), dpi=10)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 300, }
        plt.xlabel('Assets', font2)
        plt.ylabel('Weight', font2)
        plt.grid()
        plt.bar(range(len(Data)), Data, color=colors, width=0.5)
        plt.xticks(range(len(x_tick)), x_tick, size=200)
        plt.yticks(size=200)
        fig = plt.gcf()
        plt.show()
        fig.savefig(Path)
    def Plot_Weight(self,Path='.\Weight.png'):
        self.Plot_Bar(self.Weight,Path)
    def Plot_Return_All(self,index,Path='.\Total_Return.png'):
        '''
        Function: plot the return of all the assets
        Input:
        index: choose the time of the data
        (we should check the value)
        '''
        if index>=len(self.Return[0]):
            print('ERROR: index exceed capacity. In Plot_Return_All function. Class: MyPortfolio.')
            index=len(self.Return[0])-1
        data=[]
        for i in range(self.num_asset()):
            data.append(self.Return[i][index])
        self.Plot_Bar(data,Path)
    def Plot_Poly(self,Data,Name,Path):
        '''
        Function: plot a poly grah
        Input:
        data: the data to be ploted
        path: the path to store the picture
        '''
        fig, ax = plt.subplots(figsize = (12,5))
        X=np.arange(0,len(Data),1)
        ax.plot(X,Data,
                color = 'blue',
                alpha = 0.4,
                label = Name
               )
        ax.set_xlabel("Time")
        ax.set_ylabel("Return")
        ax.set_title("The Option ")
        ax.legend()
        ax.grid()
    def Plot_Return(self,index,Path='.\Return.png'):
        '''
        Input:
        index: the index of the asset in the DataSet
        '''
        Name='Portfolio:'+self.Name+' Asset:'+index+'Return'
        self.Plot_Poly(self.Return[index],Name,Path)
    def Plot_Factor(self,factor,index,Path='.\Factor.png'):
        '''
        Input:
        index: the index of the asset in the DataSet
        Factor: the factor to plot
        '''
        Name='Portfolio:'+self.Name+' Asset:'+index+'factor'+factor
        self.Plot_Poly(self.DataSet[index][factor],Name,Path)
    def Plot_Open_Price(self,index,Path='.\OpenPrice.png'):
        '''
        Input:
        index: the index of the asset in the DataSet
        '''
        column=self.Price[index].columns.valves
        Name='Portfolio:'+self.Name+' Asset:'+index+'OpenPrice'
        self.Plot_Poly(self.Price[index][column[0]],Name,Path)
    def Plot_Close_Price(self,index,Path='.\ClosePrice.png'):
        '''
        Input:
        index: the index of the asset in the DataSet
        '''
        Name='Portfolio:'+self.Name+' Asset:'+index+'ClosePrice'
        column=self.Price[index].columns.values
        self.Plot_Poly(self.Price[index][column[1]],Name,Path)
    def Plot_Volume(self,index,Path='.\Volume.png'):
        '''
        Input:
        index: the index of the asset in the DataSet
        '''
        Name='Portfolio:'+self.Name+' Asset:'+index+'Volume'
        column=self.Price[index].columns.valves
        self.Plot_Poly(self.Price[index][column[2]],Name,Path)
    def select_index(self,index):
        '''
        Function: select the rows based on index_list.
        Input:
        index: the index of asset
        index_list: retain which index
        '''
        ########################################################################
        ########################################################################
        ###########  This is the most important operation!  ####################
        #########  You must do this after any operation in DataClean  ##########
        ########## On the corresponding dataframe in the DataSet   #############
        ########################################################################
        ########################################################################
        self.Return[index]=self.Return[index].loc[self.IndexList[index]]
        self.DataSet[index]=self.DataSet[index].loc[self.IndexList[index]]
        self.Price[index]['open']=self.Price[index]['open'].loc[self.IndexList[index]]
        self.Price[index]['close']=self.Price[index]['close'].loc[self.IndexList[index]]
        self.Price[index]['volume']=dc.sum_volume(self.IndexList[index],self.Price[index]['volume'])
        self.Price[index]=self.Price[index].dropna()
    def select_columns(self,Num):
        '''
        Function: select the cols based on Impt_list.
        Input:
        Num: the number we want to retain, no more than the number of factors.
        '''
        '''
        Note:
        every DataSet[i] has same factors.
        (list)columns: the list of retained factors
        The RiskExpose better be set at the same time.
        For safety.
        Even if the order of the returned list from RetainImpt(that is, the ImptList),
        the RiskExpose and FactorList and Dataset are modified in the order of ImptList,
        so it's nevermind.
        
        '''
        ########################################################################
        ########################################################################
        ###########  This is the most important operation!  ####################
        #########  You must do this after any operation in RetainImpt  #########
        ########## On the corresponding dataframe in the DataSet   #############
        ########################################################################
        ########################################################################
        if Num>len(self.ImptList):
            Num=len(self.ImptList)
        print('self.ImptList=',self.ImptList)
        for k in range(len(self.DataSet)):
            self.DataSet[k]=self.DataSet[k][self.ImptList[0:Num]]
        self.FactorList=self.ImptList[0:Num]
        self.RiskExposeDA=self.RiskExposeDA[self.ImptList[0:Num]]
    ####################    PART 6  Boardcasting and Testing    ################
    def select_asset_avg(self,Num,method=0):
        '''
        Input:
        Num:the number of the asset to pick up.
        method:using which RiskExpose.
        method:0:DA
        Output:None
        But we modify the DataSet there.
        '''
        '''
        Note:
        We get the most important assets from our given assets.
        Based on the predicted return using RiskExpose.
        Use the simple average methods.
        '''
        
        
        if method==0:
            '''
            This time it's each asset has a function
            For each asset,we get predicted returns for each timestamp,
            and then we calulate the sum/average.
            And we will choose based on the sum/average.
            '''
            pred_return=[]
            for k in range(len(self.DataSet)):
                row,col=self.DataSet[k].shape
                sum_pred=0
                for i in range(row):
                    temp=0
                    for j in range(col):
                        temp+=self.DataSet[k].iloc[i,j]*self.RiskExposeDA.iloc[i,j]
                    sum_pred+=temp
                pred_return.append(sum_pred)
            pos=np.arange(0,len(self.score_list),1)
            pos=pos.tolist()[0]
            for i in range(len(self.score_list)-1):
                for j in range(i):
                    if pred_return[j]<pred_return[j+1]:
                        temp=pred_return[j]
                        pred_return[j]=pred_return[j+1]
                        pred_return[j+1]=temp
                        temp=pos[j]
                        pos[j]=pos[j+1]
                        pos[j+1]=temp
        pos=pos[0:Num]
        pos=pos.sort()
        Data_New=[]
        Return_New=[]
        Price_New=[]
        AssetList_New=[]
        for i in range(len(pos)):
            Data_New.append(self.DataSet[pos[i]])
            Return_New.append(self.Return[pos[i]])
            Price_New.append(self.Price[pos[i]])
            AssetList_New.append(self.AssetList[pos[i]])
        self.DataSet=copy.deepcopy(Data_New)
        self.Return=copy.deepcopy(Return_New)
        self.Price=copy.deepcopy(Price_New)
        self.AssetList=copy.deepcopy(AssetList_New)