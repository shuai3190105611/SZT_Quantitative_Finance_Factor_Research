
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from rqdatac import get_price,init,get_factor
import rqdatac
import MyPortfolio as mp
import DataClean as dc
import RetainImpt as ri
import BackTesting as bt
import OptimalWeight as ow
'''
It's a demo
'''
'''
Step 1: The configuration of data.
'''
AssetList=['000001.XSHE','000002.XSHE','000011.XSHE','000012.XSHE','000021.XSHE','000032.XSHE','000042.XSHE',
           '000061.XSHE','000062.XSHE'
 
          ]
'''
Factor come from the rqdata database
'''
Factor=[
        'pe_ratio_lyr','pcf_ratio_total_lyr','cfp_ratio_lyr','pb_ratio_lf','ps_ratio_lyr','market_cap',#from value
        'inc_revenue_lyr','inc_return_on_equity_lyr','inc_book_per_share_lyr'# from growth
]
ReturnList=[
   'return_on_equity_lyr'
]
PriceList=[
        'open','close','volume'
           ]
'''
Example of using
'''
################    step 1 create a MyPortfolio entity: PTest ####################
PTest=mp.Portfolio(Factor_List=Factor,Asset_List=AssetList,Return_List=ReturnList,Name='SZT_Product_1',Price_List=PriceList)
PTest.get_data_initial(Start_Time='2018-02-02',End_Time='2019-02-02')
################  step 2 describe the information of the data  ###################
#PTest.describe()
Start_Time='2018-02-02'
End_Time='2019-02-02'
###################  step 3 get the data from the Portfolio  #####################
'''
We only need to pass the DataSet we want to clean to it.
And we pass a dataframe a time.
'''

for k in range(PTest.num_asset()):
    Temp=PTest.get_dataset_single(k)
    #print('Asset ',k)
    #print('---------------------------------')
    #print(Temp)
    Cleaner=dc.SingleDataClean(Temp)
    Cleaner.median_de_extremum(5,0)#we set the tolerance coe 'n' to 5
    #use the resampling methods
    #the delta shouldn't be too big.
    delta=0.005*PTest.get_price_single(k)['volume'].mean()*PTest.get_price_single(k)['open'].mean()
    #print('delta=',delta)
    threshold=1
    new_index=Cleaner.non_balance_resample_volume(threshold,delta,PTest.get_price_single(k)['volume'],PTest.get_price_single(k)['open'])
    #Cleaner.describe()
    #set the index, the changes really happen.
    PTest.set_indexlist(k, new_index)
    PTest.select_index(k)

#PTest.describe_shape()#show the result
###################   step 4 : Retain important factors      #####################    
'''
Firstly we get the Filter.
And the we get the ordered list of factors.
'''
DataSet=PTest.get_dataset()
Return=PTest.get_return()
Price=PTest.get_price()
Filter=ri.RetainImptDA(DataSet, Return, Price)
Filter.get_impt_t(weight=False)
RiskExpose=Filter.get_riskexpose()
column=Filter.get_impt_ttest()
PTest.set_riskexposeDA(RiskExpose)
print('ordered column:')
print(column)
Num=5
PTest.set_imptlist(column)
PTest.select_columns(Num)
print('----------------select result------------------------')
PTest.describe_shape()#show the result
#######################   step 5 : calculate the weights    ######################
Return=PTest.get_return()
RiskExposeDA=PTest.get_riskexposeDA()
Optimizer=ow.Optimizer(Return, RiskExposeDA)
TotalMoney=100000000
Optimizer.cal_weight_1d(TotalMoney,True,1)
#Optimizer.describe()
Weight=Optimizer.get_weight()
PTest.set_weight(Weight)
############################     step 6 : Testing     ############################
Weight=PTest.get_weight()
RiskExposeDA=PTest.get_riskexposeDA()
Test=bt.TestFrameWork(Factor, AssetList, ReturnList, Weight, RiskExposeDA, 'Test for SZT_Product_1')

print(Start_Time, End_Time)
Test.get_data_initial(Start_Time, End_Time)
Test.cal_pf_return()
Test.describe_shape()
BaseLine='000032.XSHE'
Test.evaluate(BaseLine)
Test.plot_cmp(BaseLine)