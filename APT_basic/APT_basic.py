import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from rqdatac import get_price,init,get_factor
import rqdatac
import statsmodels.api as sm
class MyProtfolio:
    '''
    Readme:
    (list)self.AssetList: a list to store the codes of the component stocks
    (int)self.Num: number of assets
    (list)self.AssetWieght: weight of the asset
    (list)self.Factor: the factor of the apt model
    (list of df)self.DataSet: a set of dataframes
    (list)Return :return ratio of stock
    
    Note:
    1.the database and account could be out of date
    2.for factors, please refer the document:
    financial & accounting:https://www.ricequant.com/doc/rqdata/python/fundamentals-dictionary.html
    This version is based on rqdata
    
    '''
    def __init__(self,NumOfAsset,AssetList,Factor,Return,start_date,end_date):
        '''
        The needed packages are as follow:
        '''
        import pandas as pd
        import numpy as py
        import matplotlib.pyplot as plt
        from rqdatac import get_price,init,get_factor
        import rqdatac
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        self.NumOfAsset=NumOfAsset
        self.Weight=np.zeros(NumOfAsset)
        self.AssetList=AssetList
        self.Factor=Factor
        self.DataSet=[]
        self.Return=[]
        self.RiskExpose=np.zeros((NumOfAsset,len(Factor)))
        '''
        fetch the data
        frequency: daily
        '''
                                 
        init('17757482910','wangzhe6012,')
        for i in range(NumOfAsset):
            print(self.AssetList[i])
            df=get_factor(self.AssetList[i],Factor,start_date=start_date,end_date=end_date)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            
            self.DataSet.append(df)
        print('Return=',Return)
        for i in range(NumOfAsset):
            print(self.AssetList[i])
            df=get_factor(self.AssetList[i],Return,start_date=start_date,end_date=end_date)
            df=df.reset_index()
            del df['order_book_id']
            df=df.set_index('date')
            self.Return.append(df)
        print('Fetching success! The number of factor is:', self.num_factor)
    def num_factor(self):
        return len(self.Factor)
    def num_asset(self):
        return len(self.AssetList)
    def Standard(self):
        '''
        drop the NA
        '''
        for i in range(self.num_asset()):
            self.DataSet[i].dropna()
            for j in range(self.num_factor()):
                Max=self.DataSet[i][self.Factor[j]].max()
                Min=self.DataSet[i][self.Factor[j]].min()
                self.DataSet[i].apply(lambda x:(x[self.Factor[j]]-np.mean(x[self.Factor[j]]))/(Max-Min),axis=1)
    def RetainImptPCA(self,method,NumImpt):
        '''
        It's a method to decrease the dimension, but not used in this version,just for a redundency design
        method=
        1:PCA
        2:PCA but transfer to the raw
        other:coe of the factor and r,need to use the Standard
        NO USE in this version
        for the PCA can be only used in some medium steps
        '''
        if method==1:
            if NumImpt>self.num_factor():
                print('Exceed the number of factors')
                self.Standard()
            else :
                print('Retain ',NumImpt,'factors')
                for i in range(self.num_asset()):
                    self.DataSet[i].dropna()
                    self.DataSet[i].fillna(self.DataSet[i].mean())
                    pca = PCA(n_components=NumImpt)
                    self.DataSet[i]=pca.fit_transform(self.DataSet[i])
        if method==2:
            if NumImpt>num_factor:
                print('Exceed the number of factors')
            else :
                print('Retain ',NumImpt,'factors')
                for i in range(self.num_asset()):
                    self.DataSet[i].dropna()
                    self.DataSet[i].fillna(self.DataSet[i].mean())
                    pca = PCA(n_components=NumImpt)
                    self.DataSet[i]=pca.inverse_transform(self.DataSet[i])
        if method!=1 and method !=2:
            self.Standard()
    def DataZip(self):
        df=self.DataSet[0]
        for i in range(1,self.num_asset()):
            df=pd.concat([df,self.DataSet[i]])
        return df
    def cal_coe(self,df,i,j):
        '''
        This version based on a fundamental method, so we define a function that's in the class of the MyProtfolio
        '''
        sigma1=df[self.Factor[i]].std()
        sigma2=df[self.Factor[j]].std()
        NewDf=df.apply(lambda x:x[self.Factor[i]]*x[self.Factor[j]]/(sigma1*sigma2),axis=1)

        return NewDf.sum()
    def RetainImpt(self,method,NumImpt):
        '''
        It's the most simple method to wipe off the high-relevant factor, but as you see,it's just an elicitation method
        In this version, we use it as an assemble of data-cleaning,factor-identifying,abnormal phenomenon analysing
        method=
        1:delete the similar factors based on the correlation 
        2:based on cluster to identify the similar ones
       
        '''
        
        if method==1:
            if NumImpt>self.num_factor():
                print('Exceed the number of factors')
                self.Standard()
            else :
                '''
                calculate the total's std & coe
                '''
                self.Standard()
                print('Retain ',NumImpt,'factors')
                for i in range(self.num_asset()):
                    self.DataSet[i].dropna()
                    self.DataSet[i].fillna(self.DataSet[i].mean())
                df_cal=self.DataZip()
                '''
                add up coes then sort and drop
                '''
                Total=[]
                for i in range(self.num_factor()):
                    coe=0
                    for j in range(self.num_factor()):
                        if i==j:
                            continue
                        coe+=self.cal_coe(df_cal,i,j)
                    
                    Total.append(coe)
                Total_coe=Total.copy()
                Total_coe.sort()
                print('sort coe:',Total_coe)
                #fetch the index
                for i in range(len(Total_coe)):
                    pos=Total.index(Total_coe[i])
                    Total_coe[i]=pos
                for i in range(len(Total_coe)):
                    Total_coe[i]=Factor[Total_coe[i]]
                #delete the DataSet, based on Total_coe
                for i in range(NumImpt,len(Total_coe)):
                    for j in range(self.num_asset()):
                        del self.DataSet[j][Total_coe[i]]
                #delete the Factor
                for i in range(NumImpt,len(Total_coe)):
                    pos=Factor.index(Total_coe[i])
                    del Factor[pos]
                    
                                 
       
        if method!=1 and method !=2:
            self.Standard()
    def MyDescribe(self):
        print('DataSet=',self.DataSet[0])
        print('RiskExpose=',self.RiskExpose)
    def CalRiskExpose(self):
        self.RiskExpose=np.zeros((self.NumOfAsset,len(Factor)) ) 
                                 
        for i in range(self.num_asset()):
            X=self.DataSet[i]
            indexs=X.index.to_list()
            row,col=self.DataSet[i].shape
            delta=row
            Y=self.Return[i]
            X = sm.add_constant(X) 
            print("Return=",self.Return)
            print("X=",X)
            print("Y=",Y)
            model = sm.OLS(Y,X).fit() 
            W=model.params
            bais=model.bse
            print('sigma=',bais)
            print('W=',W)
            
            for j in range(self.num_factor()):
                self.RiskExpose[i][j]=W[j]

'''
It's a demo
'''
AssetList=['000002.XSHE','000011.XSHE','000012.XSHE','000021.XSHE','000032.XSHE','000042.XSHE',
           '000061.XSHE','000062.XSHE'
 
          ]
'''
Factor come from the rqdata database
'''
Factor=[
        'pe_ratio_lyr','pcf_ratio_total_lyr','cfp_ratio_lyr','pb_ratio_lf','ps_ratio_lyr','market_cap',#from value
        'inc_revenue_lyr','inc_return_on_equity_lyr','inc_book_per_share_lyr'# from growth
]
Return=[
   'return_on_equity_lyr'
]

NumImpt=5
method=1


MyDemo=MyProtfolio(len(AssetList),AssetList,Factor,Return,'20180202','20190202')
MyDemo.RetainImpt(method,NumImpt)
MyDemo.CalRiskExpose()
MyDemo.MyDescribe()

