# Development Guide

## Catalog

[1.Introduce](#Introduce)

[2.Design Thinking](#Design_Thinking)

[3.Data Flow Diagram](#Data_Flow_Diagram)

[4.Classes and Functions](#Classes_and_Functions)

​	[4.1.Portfolio](#Portfolio)

​	[4.2.DataClean](#SingleDataClean&other_functions_in_DataClean)

​	[4.3.RetainImpt](#RetainImptDA)

​	[4.4.Optimizer](#Optimizer)

​	[4.5.TestFrameWork](#TestFrameWork)

## Introduce

Hi！This is APT_Advanced, wish you can enjoy it!

APT model is one of the most commonly used quantitative investment models. Its main function is to extract the optimal value factor from the massive stock information based on statistics and econometrics and obtain the optimal asset portfolio. This model can meet the needs of most sell-side institutions.  

## Design_Thinking

Our design of the entire system is based on the concept of the processor, and we treat data as a stream passing between processor objects. We perform different functions in different processors and make the necessary Settings every time we call the processor.  

We have five classes: MyPortfolle. py, which holds stock and Portfolio information, dataclelean, which cleans data. The Optimizer class in optimalweight. py is responsible for solving the optimal asset portfolio, and the RetainImptDA class in RetainImpt. And TestFrameWork classes in the backtesting.py file used to test portfolio performance.  

## Data_Flow_Diagram

- For convenience, our data types are only List and DataFrame.  
- We note the data flow and the processors in the diagram.

Our data flow chart is as follows:  

![dfd](.\asset\dfd.png)

## Classes_and_Functions

### Portfolio

The main purpose of the Portfolio class is to act as a portfolio object, at the time of initialization, we to it to the following information: the preliminary screening list of factors, a preliminary screening of the assets of the list, we hope to look at the price of the list of related factors, we hope to see list of earnings factors, the name of the asset Portfolio. 

The following is the information for the private variables of the Portfolio class:  

```python
'''
	self variables:
    (list)self.FactorList: list of Factor's name
    (list)self.AssetList: list of Asset's code
    (list)self.DataSet: list of each asset's corresponding data
    (DataFrame)self.Weight: the weight of the asset
    (DataFrame)self.RiskExpose: the matrix of the return on risk exposure of each factor.The size don't need to be 
    modified.We get the column names and RiskExpose,then we index the data more conveniently.
    (list)self.Return: list of Assets' return
    (string)self.ReturnList:the name of the return
    (string)self.Name: the name of the portfolio
    (list)self.Price: list of DataFrame for:open,close,volume,fee(not used).The order should be this.
    (list)self.ImptList:the name of important factors.
    (list)self.IndexList:the list of names of each assets' index.
    The factor in DataSet may be same as someone in Price.
'''
```

Intro of the functions:

| functions                                                    | Input vars                                                   | effect                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| get_data_initial(<br/>        self,<br/>        Start_Time,<br/>        End_Time<br/>                        ): | Start_Time: input start time<br/> End_Time: input end time   | get the data through the info in the vars' list              |
| num_asset(self)                                              |                                                              | get the length of asset, the others with the form: num_xxx are the same |
| get_weight(self)                                             |                                                              | get the weight of the portfolio, the others with the form: get_xxx are the same |
| get_dataset_single(self,index)                               | (int)index: to index the position                            | get the portfolio's dataset at the specific time, the others with the form: get_xxx_single are the same |
| set_weight(self)                                             |                                                              | set the weight of the portfolio, the others with the form: get_xxx are the same |
| set_dataset_single(self,index)                               | (int)index: to index the position                            | set the portfolio's dataset at the specific time, the others with the form: get_xxx_single are the same |
| describe_shape(self)                                         |                                                              | print the shape of the portfolio's data                      |
| describe(self)                                               |                                                              | print the data of the portfolio                              |
| Plot_Bar(self,Data,Path)                                     | data: the data to be ploted<br/>        path: the path to store the picture | plot bar-graph                                               |
| Plot_Poly(self,Data,Name,Path)                               | data: the data to be ploted<br/>        path: the path to store the picture | plot Poly-graph                                              |
| Plot_Weight(self,Path='.\Weight.png')                        | path: the path to store the picture                          | plot the weight of assets using bar=graph                    |
| Plot_Volume(self,index,Path='.\Volume.png')                  | path: the path to store the picture                          | plot the volume of assets using poly-graph,the others with the form: Plot_xxx are the same |
| select_index(self,index)                                     | index: the index of asset                                    | select the rows based on index_list                          |
| select_columns(self,Num)                                     | Num: the number we want to retain, no more than the number of factors | select the cols based on Impt_list                           |
| select_asset_avg(self,Num,method=0)                          | Num:the number of the asset to pick up.<br/>        method:using which RiskExpose.<br/>        method:0:DA | We get the most important assets from our given assets.      |



### SingleDataClean&other_functions_in_DataClean

The SingleDataClean class performs data cleansing on incoming data, such as data normalization, removal of extreme values, resampling, and so on, and returns processed data along with retained labels. Other functions in the file realize other data cleaning functions. For example, according to the list for storing labels obtained after the filtering operation, the volume data as weight data is summed up according to the corresponding interval in the label list.  

The following is the information for the private variables of the SingleDataClean class:  

```python
'''
	(DataFrame)self.Data: the DataSet,a dataset of a single data
'''
```

Intro of the functions:

(some similar functions are not mentioned)

| functions                                                    | Input vars                                                   | effect                                                       |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| get_median(self,series)                                      | (series) series:the data to be processed                     | return the median of a list.                                 |
| median_de_extremum(self,n,Is_Std=1)                          | (int)n: the bound that determine the tolerrance of the extremum<br/>        (int)Is_Std: 1: do the standarlize, do nothing otherwise | get the targeted self-column de-extremumed                   |
| resample_volume_share(self,threshold,Volume)                 | (DataFrame)Volume: the volume of the asset<br/>        (float)threshold: the threshold , when the accumulated<br/>        volume of several rows' volume excedd it, the accumulated<br/>        volume will be reset to 0 | resample data based on their volume,the others with the form: resample_xxx are the same, resample based on xxx |
| sum_volume(index_list,volume)                                | (list)index_list:indexes of the raw data<br/>    (DataFrame)volume: the volume of the assets | based on the index_list to select the rows, and add up the volume of a period. |
| cusum_sample_data(Data,index,h)                              | (DataFrame)Data: the data to be processed<br/>    (int)index: which col to sample<br/>    (float)h: the bounds | sample the abnormal points that exceed the bounds.           |
| label_generate_3parclose(<br/>      Data,<br/>      HLevel,<br/>      LLevel,<br/>      TLength,<br/>      start_time,<br/>      end_time) | (DataFrame)Data:the data to be identified<br/>    (float)HLevel:high price threshold<br/>    (float)LLevel:low price threshold<br/>    (int)start_time:observe start time<br/>    (int)end_time:observe end time | generate labels, the element is like [1,1,1]                 |
| transaction_sample(Data,Label)                               | (DataFrame)Data:raw data, for its indexes<br/>    (DataFrame)Label: the labels that are sampled <br/>    in other places,index should be timeseries;<br/>    each column should be the labels. | overcome the problem of non-IID                              |

### RetainImptDA

The main function of the RetainImptDA module is to calculate the return of each asset on each factor, calculate the significance of the return of each asset on each factor, and filter the factors accordingly. Specifically, we used the data at each time point of each asset (Factor_1, Factor_2,,,, Factor_k, return) to perform OLS regression to obtain the excess return of each asset in each factor, and conducted T test to obtain the significance of each factor corresponding to each asset. We weighted average the significance of all assets in each factor, calculate the significance of each factor, and sort at the same time, and return the serial number list of the sorted assets.  

The following is the information for the private variables of the RetainImptDA class:  

```python
'''
	(list)self.DataSet: the input list of datasets of assets(DataFrame)
    (list)self.Return: the input list of Return of assets(DataFrame)
    (list)self.Price: the input list of Price of assets(DataFrame)
    (list)self.Impt_Sroce: score of the importance. based on t-test result
    
'''
```

Intro of the functions:

(some similar functions are not mentioned)

| functions                               | Input vars                                                   | effect                                                       |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| get_reg_impt_multi(self)                |                                                              | Multi-factor regression to get risk-expsure                  |
| get_reg_impt_singel(self)               |                                                              | Single-factor regression to get risk-expsure                 |
| get_impt_t(self,weight=False,IsMulti=1) | (bool)wight:<br/>        True: consider the weight,mainly for volume.<br/>        False: don't consider the weight.<br/>        (bool)IsMulti:<br/>        1: Multi regression<br/>        0: single regression | get the important value of each factor, using the weighted-average based on volume |
| get_impt_ttest(self)                    |                                                              | get the t-test result                                        |

### **RetainImptDT**

The RetainImptDT class is similar to RetainImptDA. The difference lies in that the former performs cross-sectional regression and carries out weighted average of the regression results at each time point. Other than that, the mechanics are similar.  

### Optimizer

Optimizer uses mathematical programming to find the optimal portfolio of assets. We enter the return sequence and the return of each asset on each factor's exposure. The model returns a list of weights for each asset.  

The following is the information for the private variables of the Optimizer class:  

```python
'''
	(list)Return: a list of returns of all assets. 
	(DataFrame)RiskExpose: the dataframe of return on riskexpose
'''
```

Intro of the functions:

(some similar functions are not mentioned)

| functions                                                    | Input vars                                                   | effect                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cal_arith_avg(self)                                          |                                                              | calculate the average return of the assets.                  |
| cal_geo_avg(self)                                            |                                                              | calculate the average return of the assets.                  |
| cal_weight_1d(<br/>self,<br/>Max_Total,<br/>Admit_Short=True,<br/>method=1) | (bool)AdmitShort: 1 if short sellinf is permitted; 0 otherwise.<br/>        (int)method: 1: arith-avg to get the average<br/>        0: geo_avg to get the average | calculate the optimal weight, based on the linear programming. |

### TestFrameWork

The TestFrameWork class is used to evaluate the performance of our Portfolio, and the data passed into this module is similar to the Portfolio module.  

The following is the information for the private variables of the TestFrameWork class:  

```python
'''
	(list)Return: a list of returns of all assets. 
	(DataFrame)RiskExpose: the dataframe of return on riskexpose
'''
```

Intro of the functions:

(some similar functions are not mentioned)

| functions               | Input vars                                        | effect                                                       |
| ----------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| plot_cmp(self,BaseLine) | (string)BaseLine: the code of the baseline asset. | plot the baseline's data and portfolio's data.               |
| cal_pf_return(self)     |                                                   | calculate the portfolio's predicted return based on time-interval |
| evaluate(self,BaseLine) |                                                   | evaluate the performence                                     |

