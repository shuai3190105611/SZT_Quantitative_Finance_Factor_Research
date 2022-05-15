# Decision-tree: an approach to detecting the financial default

Contributor: `Zitao Shuai`

Developed in: `Data-mining course-2022-summer`

instructor: Prof. `Shijian Li`

## Data Set

We use the data set from `kaggle`,  it's mainly about the user features, we need  to classify the users into `Default` or not. And the link is as follows:

[Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

Our label is ：`"default"`

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

And the used tables are in `.\HW2\code` in which you can find the source code too.

## Get start to design the `DTree`

Now let's introduce our main idea:

There are two classes basically, one is a class in the model level, while the other is the node level. Here we list a table to illustrate it.

| class name | function                                                   |
| ---------- | ---------------------------------------------------------- |
| `MyDTree`  | To initializes the model and interacts with users directly |
| `TreeNode` | basic data structure, used to construct the tree           |

And we use a graph to show the workflow:

![image-20220510143102525](E:\大三下\数据挖掘\HW2\asset\image-20220510143102525.png)

And in our designs, the data is stored in both `MyDTree` model and the `DTree`. But the nodes only store the `split data`, and they index the data from predecessors' `dataSet` using the list of indexes generated after the `splitting`  operations in their predecessors. 

## Class and function

### Some details:

Firstly, we split using the **`ID3`**, that is , we calculate the entropy for each factor following the formula:
$$
Entropy=-\sum(\frac{D_i}{D}\times log(\frac{D_i}{D}))
$$
And we select the factor that generates the minimum entropy and use it to split.

Secondly, to save up the main memory, we use **reference** instead of value propagation. And we use an extra index list to help index the data. 

Thirdly, for predictions of each tuple, we search the abstract tree, in each nodes of the searching path, we compare the value of node's current `splitting feature` value of the tuple with each child

Forth, for the consideration of consistency, our API's names are similar to the main current models' names. So do the methods to call these APIs. Like `.fit()`, `.predict()` and so on.

### Classes:

#### ` MyDTree`

It's an external class, mainly for interacting with users and do some high-hierarchy jobs. Like predict, fit, setting. But the implementation must rely on the `TreeNode ` class.

When we call `fit()` and pass data into it, the `MyDTree` object first call the `train()`  then it set the root node, and from the root node, the `DTree` will be generated recursively. 

About the `MyDTree`:

```python
class MyDTree:
    '''
    The desicion tree, implemented myself.
    Add some user-friendly functions.
    Params:
    (DataFrame)dataSet: the used data
    (featureList)featureList: the features of training
    (TreeNode)self.Root: the root of the tree
    (string)label: the name of the label
    (int)max_depth: the maximum deepth
    (int)min_element: the minimum number of the element of a leaf
    (int)max_child: the maximum number of the child nodes
    '''
```

About the settings:

Our model's setting is a dictionary, with some private parameters in the object corresponding to it. And a demo for it：

```python
settings={'max_depth':10,'min_element':10,'max_child':10}
Demo=MyDTree(settings) #init
```

`fit()`

This function is used to fit the model. We call this function in the form: `model.fit(X,label)`, and the `label` is the name of the labels. And the code:

 

```python
    def fit(self,dataSet,label):
        '''
        Function: 
        Initialize. We input the dataSet only in the initialization.
        Input:
        (DataFrame)dataSet: data  set being used.
        (string)label: the label is the name of the selected column
        which is used to be the label of the classification task.
        '''
        self.dataSet=dataSet
        self.featureList=self.dataSet.columns.tolist()
        self.featureList.remove(label)
        self.label=label
        self.train()
    def train(self):
        start = datetime.datetime.now()
        self.Root=TreeNode( self.dataSet
                           ,deepcopy(self.dataSet.index.tolist())
                           ,self.featureList
                           ,self.label
                           ,0
                           ,self.max_depth
                           ,self.min_element
                           ,self.max_child)
        end = datetime.datetime.now()
        print('totally time is ', end - start)
        print("OKKKKKKKKKKKKKKKKKKKKK! Training completed!")
        
```

The Demo:

```python
Demo.fit(dfTrain,'"default"') #training    
```

`predict`

This function is used to predict via the model. We call this function in the form: `model.predict(X,label)`, and the `label` is the name of the labels. And the code:

```python
    def predict(self,dataTest,factor):
        '''
        Function:
        Predict the value of feature: "default".
        We search from root to leaf. Then use the modal number (class) of the leaf.
        '''
        indexList=dataTest.index.tolist()
        rst=pd.DataFrame(columns=[factor],index=indexList)
        for i in indexList:
            rst[i]=self
```



#### `TreeNode`

The `TreeNode` data structure is the most important part of the model. We use a list to construct a tree be storing a list of child nodes.

In this class, most of our operations are in `__init__ `function. And we construct a `DTree` recursively. And after training the model, we can search a corresponding label of the input tuple.

About the `TreeNode`:

```python
class TreeNode:
    '''
    Params:
    The others are same as the MyDTree.
    (float)curEntropy: current entropy
    (string)splitFactor: name of the factor used to split
    (list)childList: the list to contain the child nodes
    '''
```

The code of the main operations when training is as follows:

```python
        ########## calculate the entropy and select  ############
        self.splitFactor=self.getFactor()
        ################  split and check  ######################
        if self.splitFactor is None:
            print("LEVEL: ",self.level,"not suitable factor!")
        else:
            childFeature=deepcopy(self.featureList)
            childFeature.remove(self.splitFactor)
            nodeList=self.dataSet.loc[self.indexList,self.splitFactor].drop_duplicates().tolist()
            #print("The list of the nodes are: ", nodeList)
            
            ################# set the childnode  ####################
            '''
            in this part, we need to set the index and features on the
            passed data, and remove the factor of the featurlist, then 
            set the level.
            and we should calculate the entropy for the children. 
            '''
            if self.level<self.max_depth:
                for i in nodeList:
                    
                    indexChildList=self.dataSet[self.dataSet[self.splitFactor]==i].index.tolist()
                    indexChildList=list(set(indexChildList) & set(self.indexList))
                    ##########     recursiveeeeeeeeeee   god bless!    ########
                    newChild=TreeNode(  self.dataSet
                                       ,indexChildList
                                       ,childFeature
                                       ,self.label
                                       ,self.level+1
                                       ,self.max_depth
                                       ,self.min_element
                                       ,self.max_child)
                    self.childList.append(newChild)
```

And for searching (that will be used in the predict function of the `MyDTree` model):

```python
    def search(self,dataTuple):
        '''
        Function:
        Search for the leaf recursively. And find the corresponding class.
        Input:
        (DataFrame)dataTuple
        '''
        
        if len(self.childList)==0:
            #################         leaf node        ####################
            ####### calculate which class has the maximum number  #########
            temp=self.dataSet.loc[self.indexList]
            temp=temp.groupby([self.label])[self.dataSet.columns.tolist()[0]].count()
            #print(temp)
            max_index=0
            max_count=0
            for i in temp.index:
                #print("i=" ,i)
                if temp[i]>max_count:
                    max_count=temp[i]
                    max_index=i
            return max_index
            #print("max_index = ", max_index," max_count = ",max_count)
        else:
            flag=dataTuple[self.splitFactor]
            for i in range(len(self.childList)):
                if self.childList[i].dataSet.loc[self.childList[i].indexList[0],self.splitFactor]==flag:
                    return self.childList[i].search(dataTuple)
```

## Testing 

### Testing framework

#### Data load and split

We load the cleaned data using `pd.read_csv`, and we drop the `none value`, then set the index. We implement a function to split the data:

```python
def MySplit(dataSet,K,ratio):
    '''
    Function:
    To generate the data for training and testing.
    Input:
    (DataFrame)dataSet: the raw data set.
    (int)K: the number of turns
    (float)ratio: the ratio of splitting
    '''
    returnSet=[]
    for i in range(K):
        train_start=(int(dataSet.shape[0]*i*1.0/K))%dataSet.shape[0]
        train_end=(int(ratio*dataSet.shape[0])+int(dataSet.shape[0]*i*1.0/K))%dataSet.shape[0]
        print(train_start,train_end)
        if train_start>train_end:
            dfPre=dataSet.iloc[train_start:train_end]
            dfTrain=pd.concat([dataSet.iloc[0:train_start],dataSet.iloc[train_end:]])
        else:
            dfTrain=dataSet.iloc[train_start:train_end]
            dfPre=pd.concat([dataSet.iloc[0:train_start],dataSet.iloc[train_end:]])

        returnSet.append((dfTrain,dfPre))
    return returnSet
```

 That's because I want to process the data in the type of `DataFrame`, but `sklearn` can't process this type, so I created a new one.

#### Model saving

As a result of the large scale of the data, we need to save the trained model. So we use the following code to do this:

```python
import joblib
joblib.dump(Demo, r'MyDTree.pkl') #saving
```

And we can import it using this code:

```python
Demo=joblib.load(r'MyDTree.pkl')
```

#### Then the demo code is as follows:

```python
######################  data  #############################


df=pd.read_csv("dataSetCleaned.csv")
df=df.set_index(df.columns.tolist()[0])
df=df.dropna()
######################  demo  #############################

settings={'max_depth':10,'min_element':10,'max_child':10}
Demo=MyDTree(settings) #init
count=0
######################  test framework  ###################
MyResult=pd.DataFrame(columns=['Total_Num','True_Num','True_ratio','Run_time'])
for (dfTrain,dfPre) in MySplit(df,10,0.8):
    Path='MyDTree.pkl'+str(count)
    runTime=Demo.fit(dfTrain,'"default"') #training
    joblib.dump(Demo, Path) #saving
    #Demo=joblib.load(Path)
    rst=Demo.predict(dfPre,'"default"')
    dfPre['rst']=rst
    TrueNum=dfPre[dfPre['rst']==dfPre['"default"']]['rst'].count()
    TotalNum=dfPre['rst'].count()
    print("Total Num:",TotalNum)
    print("True Num:",TrueNum)
    print("True ratio = ",float(TrueNum)/TotalNum)
    print("Run time = ",runTime)
    MyResult.loc[str(count)]=[TotalNum,TrueNum,float(TrueNum)/TotalNum,runTime]
    count+=1
MyResult.to_csv("MyResult.csv")
#Demo=joblib.load(r'MyDTree.pkl')
      
```

### Testing result： `MyDTree` and `sklearn decisign-tree` 

Empirical analysis:

| hyper-params | meaning                                         |
| ------------ | ----------------------------------------------- |
| max_depth    | maximum depth of the leaves                     |
| min_element  | the minimum number of elements of each leaves   |
| max_child    | the maximum number of child nodes of each nodes |

Setting:

| items                                   | number      |
| --------------------------------------- | ----------- |
| number of feature                       | 21          |
| number of different<br> kinds of labels | 4           |
| total rows of cleaned data              | about 42000 |
| train-split ratio                       | 0.8         |
| number of K-fold sets                   | 10          |

`MyDTree`

|      | True_ratio | Run_time     |
| ---- | ---------- | ------------ |
| 0    | 0.884479   | 02:48.001473 |
| 1    | 0.780906   | 01:48.939627 |
| 2    | 0.682627   | 00:44.571716 |
| 3    | 0.69435    | 00:42.203215 |
| 4    | 0.716312   | 00:48.342193 |
| 5    | 0.740479   | 00:56.450815 |
| 6    | 0.765182   | 00:54.923466 |
| 7    | 0.830904   | 00:53.527816 |
| 8    | 0.843351   | 00:50.277397 |
| 9    | 0.843448   | 00:50.652716 |

`sklearn decision-tree` 

We set the same max_depth to 10.

Note:

The input of the `sklearn`'s decision tree should be 'int'. So we need to pre-process the data using the following code:

```python
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
        temp.loc[tempIndex]=i
    df[col]=temp
```



And the testing result:

|      | True_ratio | Run_time     |
| ---- | ---------- | ------------ |
| 0    | 0.871085   | 00:00.077685 |
| 1    | 0.808327   | 00:00.083286 |
| 2    | 0.691066   | 00:00.080293 |
| 3    | 0.689124   | 00:00.079159 |
| 4    | 0.71765    | 00:00.100116 |
| 5    | 0.730639   | 00:00.085051 |
| 6    | 0.740714   | 00:00.079982 |
| 7    | 0.825079   | 00:00.090643 |
| 8    | 0.837946   | 00:00.088392 |
| 9    | 0.798616   | 00:00.086802 |

### Result analysis

Firstly, the resulting two tables show that the skearn's tun time is much shorter than mine.

It may as a result of the optimization it does,  pruning eg.

But pruning is a complex thing, if we operate wrongly, the outperform would be worse.

Out model  perform well in 80% testing sets. (However just a bit, only 1 exceed over 5%.) However I didn't do much optimization, I think it's just because I didn't use the pruning operations, that make the model can predict more accurate. However, the time consumption is so large. I'll try to optimize it in the future if I have some free time.

