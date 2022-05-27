# `Aprioi Algorithm`

Contributor: `Zitao Shuai`

Developed in: `Data-mining course-2022-summer`

instructor: Prof. `Shijian Li`

## Data Set

We use the data set from `kaggle`,  it's a data set with plenty of features, not only can it being used in the classification task, but also we can do some pattern mining work on this data set. And the link is as follows:

[Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

We mainly use the `Apriori Algorithm` to complete the pattern mining work.

Firstly we do some data cleaning work, because the features are  compressed into a single column. And we split each string into a list of strings, and then transform it to a row of data. And we use the rows to construct a new `dataframe`.

And the code is as follows:

```python
import pandas as pd
import numpy as py
df=pd.read_csv("bank-additional-full.csv")
columns=df.columns.tolist()[0].split(";")
dataList=df.iloc[:,0].tolist()
dataSet=pd.DataFrame(df,columns=columns)
for i in range(len(dataList)):
    dataSet.loc[str(i)]=dataList[i].split(";")
    if i%100==0:
        print(dataList[i].split(";"))


dataSet.to_csv("dataSetCleaned_new.csv")
```

The cleaned data is in this form:

![image-20220510133749416](E:\大三下\数据挖掘\HW2\asset\image-20220510133749416.png)

And the used tables are in `.\HW3\code` in which you can find the source code too.

## Basic knowledge

Before getting start, we'd better review some basic knowledge about the frequent pattern  mining.

#### Definitions

If we consider the question globally, then we may have the `support`:
$$
Support_i=\frac{times\ the\ {X_i}\ occurs}{the\ global\ samples}
$$
Then we have the metric for a transaction:
$$
Support(X_i,X_j)=\frac{times\ the\ ({X_i,X_j})\ occurs}{the\ global\ samples}
$$
And if we consider two variables, think about their joint distribution of probability, then we can get the `confidence`:
$$
confidence(X_i,X_j)=\frac{times\ the\ ({X_i,X_j})\ occur}{times\ the\ {X_i}\ occur}
$$

#### Examples

So we have three main definitions about this work. We will illustrate them with a simple example.

Think about this case about the skill set:

| item              | amount |
| ----------------- | ------ |
| coding            | 11     |
| math              | 9      |
| reading           | 6      |
| writing           | 8      |
| slides            | 5      |
| presentation      | 6      |
| (coding, math)    | 8      |
| (coding,slides)   | 3      |
| (math,slides)     | 3      |
| (math,writing)    | 3      |
| (coding,writing)  | 4      |
| (coding,math,ppt) | 2      |
| Total             | 15     |

So we can get the `support` of the feature `coding` is: `0.733`

And the `support` for the transaction `(coding,math)` is: `0.533`

And  the `confidence` of the `(coding,math)` is: `0.727`

#### Some hidden rules

Firstly, you can see that, a superset's support is smaller than the subset, like the `confidence` of `coding` and the `confidence`  of `math` are both bigger than that of the `(coding,math)`.

Second, we the terms of all combination may be:
$$
2^k
$$
when k is the number of items, so it's hard to compute. What we need is to do some pruning work on it. So that's where the `Apriori Algorithm` comes from.

### Getting start to implement the `Apriori Algorithm`

So we need to solve the problem reversely?

No not exactly. Because the number of satisfied sets will decrease, if we use the reverse to solve the problem, we can't consider the other branches' conclusions, so that we may consider something we don't need to consider.

So we solve the problem iteratively.

#### Class

In this question, we mainly use a single class `Apriori` as follows:

```python
class Apriori:
    '''
    (DataFrame)self.dataSet: the dataSet to mine
    (float)threshold: the minimum the support of transactions(or tuples) should be
    (int)k: the number of features in the transactions.
    (list)retainList: to retain the satified minimum terms.
    (list)support: record the score of the support
    '''
```

We input the data to the `Apriori` object, and then we call the `findMax` to complete the `Apriori Algorithm`.

And the `dataflow graph`:

![image-20220522095858902](asset/image-20220522095858902.png)

And I'd like to give a further explain on the `retainList`

Since we search the solution by layer, we can retain the satisfied terms on layer k, which corresponding to the terms in the form: (x1,,,,xk). 

And a graph to illustrate it:

![image-20220522100946105](asset/image-20220522100946105.png)

And before we implement the algorithm, we need to encode for each features' each values. And we need to remove the "_" in the columns' names.

That would be in the form: 

`feature_name`_`value`

Instead of using `one-hot` encoding

And the code should be:

```python
def replaceToCode(df):
    '''
    Function:
    To replace the feature into int numbers.
    And we need to remove the "_"
    None.

    '''
    for col in df.columns:
        countList=df[col].drop_duplicates().tolist()
        count=len(countList)
        '''
        print("countList = ",countList)
        print("count = ",count)
        '''
        temp=deepcopy(df[col])
        for i in range(count):
            '''
            print("count i = ",countList[i])
            print(df[df[col]==countList[i]][col])
            print("i = ",i)
            '''
            tempIndex=df[df[col]==countList[i]][col].index.tolist()
            tempStr=str(col)+'_'+str(countList[i])
            temp.loc[tempIndex]=tempStr
        df[col]=temp
    tempCol=df.columns.tolist()
    for i in tempCol:
        temp=i
        temp=temp.strip("_")
        df=df.rename(columns={i:temp})
    #print(df)
    return df
```



#### Functions

##### `findMax`

We mainly use this function to deal with the frequent terms mining problem.

In this place, we provide two methods to calculate the support, one for absolute support, one for relative.

In `Apriori Algorithm`, we use the relative support when update the iterms.

This function contains the main logic of our `Apriori algorithm`, and we execute by layer, add the items as the rule:

`x,x,,,(x,x),(x,x),,,(x,x,x),(x,x,x)`

That is, from subset to super set, and this method using the idea we've illustrated above, and decrease the time complexity.

Then the code as follows:

```python
    def findMax(
            self
            ,threshold
            ,k
            ,option="re"
            ):
        '''
        Function:
        Find the max support k-transactions
        Input:
        (float)threshold: the minimum the support of transactions(or tuples) should be
        (int)k: the number of features in the transactions.
        (string)option: "re": relative support,"ab": absolute support
        '''
        self.threshold=threshold
        self.k=k
        row=self.dataSet.shape[0]
        self.total=row
        self.retainList=[]
        if k>len(self.dataSet.columns.tolist()):
            print("Error! The k is bigger than the number of features!")
        else:
            '''
            And we need the single list to denote which single items can be added in.
            list of lists.
            '''
            single=[]
            for col in self.dataSet.columns.tolist():
                countList=deepcopy(self.dataSet[col].drop_duplicates().tolist())
                single.append(countList)
            #print(single)
            for layer in range(self.k):
                
                temp=[]
                if layer==0:
                    ##  the logic is totally different from that belows in the else block ##
                    for i,col in enumerate(self.dataSet.columns):
                        temp1=[]
                        for j in range(len(single[i])):
                            if self.dataSet[self.dataSet[col]==single[i][j]][col].count()>self.threshold*self.total:
                                temp1.append(single[i][j])
                                temp2=[]
                                temp2.append(single[i][j])
                                if len(temp2)>0:
                                    self.retainList.append(temp2)
                                    self.support.append(1.0*self.dataSet[self.dataSet[col]==single[i][j]][col].count()/self.total)
                        if len(temp1)>0:
                            temp.append(temp1)
                            
                    single=deepcopy(temp)
                    #print(len(self.support),len(self.retainList))
                else:
                    
                     '''
                     The items in retainList are in the form of list:[x,y].
                     We search if we can add a element and still satisfy the threshold.
                     '''
                     if option=="ab":
                         temp=self.retainImptAb(single)
                     else:
                         temp=self.retainImptRe(single)
                     self.retainList=deepcopy(temp)
                     #print(len(self.support),len(self.retainList))
                     
                print("retainList = ")
                print(self.retainList)
            
            df=pd.DataFrame(self.retainList)
            support=pd.DataFrame(self.support)
            df=pd.concat([df,support],axis=1)
            df.to_csv("result_K"+str(self.k)+"_row_"+str(self.total)+".csv")
            
```

##### `retainImptRe`

This function is used to enlarge the frequent items, from sub sets to super sets. And we need to calculate the joint distribution's probability:
$$
P(x_1,x_2,,,x_{n-1})\\
P(x_1,x_2,,,x_{n-1},x_n)
$$
We should traverse the features ( or columns), and get the intersetion for each feature's corresponding indexes.

And we need to pass in the frequent single features, and if we want to add element to a frequent term and test if the new support is larger than the threshold, we need to choose from this set. And the codes as follow:

```python
    def retainImptRe(self,single):
        '''
        Function:
        return the important terms.
        Input:
        (list)single: the feasible features can be added in.list of lists.
        note:
        pay attention to skip the same iterms to avoid the conditions like: (x1,x1)
        '''
        temp=deepcopy(self.retainList)
        for item in self.retainList:
            #search for the labels
            #use a list to contain
            checkList=[]
            for i in item:
                posItem_=i.find("_")
                colItem=i[0:posItem_]
                checkList.append(colItem)
                
            for i in range(len(single)):
                #print("i = ",i)
                for j in range(len(single[i])):
                    posSingle_=single[i][j].find("_")
                    colSingle=single[i][j][0:posSingle_]
                    if self.check(colSingle,checkList)==1:
                        continue
                    tempItem=deepcopy(item)
                    tempItem.append(single[i][j])
                    probNew=self.getSup(tempItem)
                    probOld=self.getSup(item)
                    #print("item = ",item)
                    #print("probNew = ",probNew,"probOld = ",probOld)
                    if probNew>=self.threshold * probOld:
                        temp.append(tempItem)
                        self.support.append(probNew)
                    #print(len(self.support),len(temp))
        return temp   
```

### Performance Testing

#### Testing example

Our testing demo is as follows: 

Our results will be stored in the `.csv` file.

```python
df=pd.read_csv("dataSetCleaned.csv")
df=df.set_index(df.columns.tolist()[0])
df=df.dropna()
df=replaceToCode(df)
dftemp=deepcopy(df)
testTime=[]
kList=[]
totalList=[]
threshold=0.7
for i in range(4):
    for j in range(4):
        df=dftemp.loc[dftemp.index.tolist()[0:(j+1)*10000]]
        start = datetime.datetime.now()
        AP=Apriori(df)
        AP.findMax(threshold,i+1)
        end = datetime.datetime.now()
        testTime.append((end-start).microseconds)
        kList.append(i)
        totalList.append((j+1)*10000)
testTime=pd.DataFrame(testTime)
kList=pd.DataFrame(kList)
totalList=pd.DataFrame(totalList)
testTime=pd.concat([testTime,kList,totalList],axis=1)
testTime.columns=['testTime','k','totalData']
testTime.to_csv("testTime"+str(threshold)+".csv")
#AP.countNum()
```



#### Testing result

We set different values of the maximum number of terms in a transaction or a tuple and different value of the scale of  data, then change the value of `support`'s threshold,and get several testing results as follow:

The records of time are in microseconds.

#### Test case 1: threshold=0.7 

| Runtime | maximum number of terms | data scale |
| ------- | ----------------------- | ---------- |
| 982119  | 0                       | 10000      |
| 756502  | 0                       | 20000      |
| 749078  | 0                       | 30000      |
| 972343  | 0                       | 40000      |
| 79386   | 1                       | 10000      |
| 452453  | 1                       | 20000      |
| 970648  | 1                       | 30000      |
| 301480  | 1                       | 40000      |
| 493350  | 2                       | 10000      |
| 804574  | 2                       | 20000      |
| 732085  | 2                       | 30000      |
| 198488  | 2                       | 40000      |
| 967641  | 3                       | 10000      |
| 491752  | 3                       | 20000      |
| 395454  | 3                       | 30000      |
| 617669  | 3                       | 40000      |



#### Test Case 2: threshold=0.8

| Runtime | maximum number of terms | data scale |
| ------- | ----------------------- | ---------- |
| 917784  | 1                       | 10000      |
| 924435  | 1                       | 20000      |
| 877938  | 1                       | 30000      |
| 492837  | 1                       | 40000      |
| 330490  | 2                       | 10000      |
| 459632  | 2                       | 20000      |
| 755267  | 2                       | 30000      |
| 661489  | 2                       | 40000      |
| 79996   | 3                       | 10000      |
| 585836  | 3                       | 20000      |
| 846432  | 3                       | 30000      |
| 130996  | 3                       | 40000      |
| 817981  | 4                       | 10000      |
| 560238  | 4                       | 20000      |
| 73746   | 4                       | 30000      |
| 867017  | 4                       | 40000      |
| 597716  | 5                       | 10000      |
| 411620  | 5                       | 20000      |
| 777543  | 5                       | 30000      |
| 975510  | 5                       | 40000      |

Analysis:

We can clearly see that:

- with the threshold's decreasing, the run time is increasing sharply. I suppose that since many frequent terms only have a  lower `support` score smaller than 0.8, so if we reduce the threshold, there may be a large increase on the number of the terms. So thanks to the geometrical consumption of the term-enlarge-process, the time-consumption is highly magnified.
- For the cases of `threshold = 0.8`, different maximum number of terms of data seems make no sense. But that's because, under the threshold of 0.8, there only a few frequent terms throughout the dataset, so enlarge the maximum number of terms of testing dataset makes no difference.
- But enlarge the scale of dataset will bring difference generally.

### Analysis on the result

We choose the `support = 0.8` as an persuasive example.

And we choose `K = 5` and `row = 40000`.

#### Descriptive statistic

Now we have `1046` record, and in the program I set the output `support` to be absolute `support`.

For the single condition:

There only 5 items. The feature `pdays` with the value `0.999.0` ranks first with the support `0.97275`.

For the two items condition:

The `("pdays"_999.0,"y"_"no")` ranks first.

That tell us that : `pdays` and `y` are the most important features. And when feature `pday` 's value is `999.0`, the value of `y` is likely to be `no`.

And the distribution of the supports (above the 0.8)

![image-20220522162809838](asset/image-20220522162809838.png)
