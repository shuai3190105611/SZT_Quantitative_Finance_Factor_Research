# Development Guide

### Introduce

About APT_basic:

It's a light-weight implementation about APT model. The code only include the function of: basic data cleaning, basic factor selecting, and calculating the risk-exposure. And it's good for beginners to learn the APT model, for its simplicity. My advanced APT model is based on the experience and inspiration during developing the APT_basic. Have a look at the model may help you learn other models I've developed.

However,it' s still a self-finished product, the design of the class, the data passing, and the coding style are not so good. So please make sure you are keeping a critical eye on it.

## Algorithms

In this simple version, we mainly use linear regression to get the risk-exposure of each assets.



## Classes and Functions

This version we only implement a class called **MyPortfolio**.

Containing the data of factors and returns, the class is used to **calculate the risk-exposure** of each asset.

### **Data Structure**

| Name            | Introduce                                                   | Eg.                                     |
| --------------- | ----------------------------------------------------------- | --------------------------------------- |
| `self.DataSet`    | a list of DataFrame about the data of factors of each asset | [df1,df2,df3] (multi-dimension for dfs) |
| `self.Return`     | a list of DataFrame about the return of each asset          | [df1,df2,df3] (1-dimension for dfs)     |
| `self.RiskExpose` | a array of risk-expose of each asset                        | array([1 1 1] [2 2 2])                  |



### Functions

| Class                                | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `RetainImpt(self,method,NumImpt)`    | get the important factors base on coefficient of associations, we retain the factor which have the  smaller of it |
| `Standard(self)`                     | standardize the data                                         |
| `CalRiskExpose(self)`                | calculate the risk-exposure based on OLS                     |
| `RetainImptPCA(self,method,NumImpt)` | Use the PCA method to get important factors                  |


