# Data visualizer

Contributor: `Zitao Shuai`

Developed in: `Data-mining course-2022-summer`

instructor: Prof. `Shijian Li`

## Motivation

In the current era of big data, data mining, artificial intelligence and big data have gradually become indispensable information technology means for industrial production and daily life in modern society. Big data, as an important resource, should be fully utilized. Data mining is an important means to fully explore the information in massive data and provide support for people's decision making and analysis.  

在当今大数据时代的背景下，数据挖掘和人工智能、大数据逐渐成为现代社会进行工业生产、日常生活所不可或缺的信息技术手段。其中，大数据作为一项重要的资源，应该被得到充分的利用。数据挖掘则是充分发掘海量数据中的信息、并为人们的决策、分析提供支持的重要手段。

Data tend to be large-scale, and it is often difficult to quickly draw patterns from large numbers. It prevents us from making good decisions and reduces productivity; Or we can not take the right method to pre-process the data and feature engineering, affecting the processing efficiency of the following artificial intelligence model. So, to help people make the right choices, we need to visualize the data.  

数据往往是大规模的，人们往往难以从海量的数字中很快总结出规律。这会妨碍我们做出正确的决策，降低工作效率；或者使我们不能采取正确的方法对数据进行预处理和特征工程，影响紧随其后的人工智能模型的处理效率。因此，为了帮助人们做出正确的选择，我们需要进行数据的可视化。

## Data flow and thinking of designing 

We think of the visualization tool as an object that contains the data we enter. After initialization we can do a bunch of things like drawing, the style of drawing is up to us, we can even pre-configure and so on. In addition, we need efficient processing of large-scale data.  

我们把可视化工具视为一个对象，其中包含我们输入的数据。在初始化之后我们可以进行作图等一系列的操作，作图风格都取决于我们，我们甚至可以预先进行配置等等。除此之外，我们还需要对大规模数据的高效的处理。

In this design, my target data set is the financial data set, so my design logic will also focus on the characteristics of this data set. To be specific, these data have a very typical time series, there is a large sequence correlation between the series data, and obey some empirical distribution characteristics. These characteristics include `mean recovery`, `peak and thick tail`, `volatility aggregation` and so on.  

本次设计中，我的目标数据集是金融数据集，因此我的设计逻辑也会围绕这一数据集的特点展开。具体而言，这些数据具有很典型的时间序列，序列数据之间存在较大的序列相关性，并且服从一些经验性的分布特征。这些特征有：**`均值回复性`、`尖峰厚尾性`，`波动聚集性`**等等。

A typical way to use this tool is to first enter data and names into the object at initialization; The next step is to set the object, using the Configuration function to reduce the input of the call parameters later in the drawing. At the same time, we can also set the criteria so that we can distinguish between different points and give them different colors to distinguish them in the drawing. In addition, we will have some specific methods for our business data set, identifying whether there are typical time series characteristics and presenting them visually.  

一个典型的使用本工具的方法是：首先在初始化时将数据和名字输入到对象里面；其次对这个对象进行设置，我们可以通过configuration函数进行设置，这样在之后的绘图中可以减少调用参数的输入。同时我们还可以对判别条件进行设置，使得我们可以对不同的点进行区别，并给与不同的颜色，在作图中区分出来。另外，我们还会有专门面向我们业务数据集的一些方法，识别其中有无典型时间序列特征，并用可视化的方法展现出来。

The data flow graph is as follows:

如下为我们的数据流图：

![data.drawio (1)](.\asset\data.drawio (1).png)

figure 1: data flow graph

## Class and methods

### Important APIs

#### `init`

```python
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
```

Function:

In this section we mainly do the initialization, it is worth noting that we need to create a `config`matrix here, and we need to set the default values. For example, for the upper and lower limits of values, we examine the data type first, and then set the upper and lower limits. For the string class data, we will do nothing.  

在这个部分我们主要是进行初始化，值得注意的是我们需要在这里创建一个`config`矩阵，同时我们需要设置默认的值。例如对于值的上下限，我们先检验数据的类型，然后再为其设置上下限。对于字符串类的数据，我们将不进行操作。

Example:

```python
df=pd.read_csv("ADANIPORTS.csv")
Demo_v=visualizer(df,"Demo 0")
```



#### `coherance`

```python
def coherance(self,Factor1,Factor2,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (string)Factor1: name of the first factor
        (string)Factor2: name of the second factor
        Output:
        pictures
        '''
```

Function:

We use this function to represent the correlation factors between the two factors. Specifically, we make a line graph of the two sequences and make a graph reflecting the magnitude of their correlation.  

我们使用这个函数来表现两个因子之间的相关性因素，具体而言，我们作出两个序列的折线图，并作出反映其相关性大小的图表。

In the implementation, we call the `select_label`function in the class, which filters the data based on the Settings in the `config`function. We temporarily changed the beautiful parts of the code manually, because we didn't make the user interface, and the difference between manually changing the code and implementing a function to change it is not obvious.  

在实现中，我们调用了类里面的`select_label`函数，这个函数会根据`config`函数中的设置筛选数据。我们暂时是手动完成美观部分代码的更改的，因为我们没有制作用户界面，手动更改和单独实现一个函数来更改区别并不明显。

Example:

```python
Demo_v.coherance('Open','High','Date')
```



#### `config_plot`

```python
 def config(self,Factor,upper_bound,
            lower_bound,max_point_num,interval):
        '''
        Function: set the configuration.
        Input: 
        (float)upper_bound: size of the
        (string)Factor: which factor to configurate
        (float)upper_bound: value of upper bound
        (lower)upper_bound: value of lower bound
        (int)max_point_num: maximum number of points
        (int)interval: interval for sampling
        '''
```

Function

In this function, we mainly set the data used for drawing. These variables are shown in the following table:  

这个函数中我们主要完成对绘图所使用的数据的设置，这些变量如下表所示：

| Name          | Type  | Intro                    |
| ------------- | ----- | ------------------------ |
| upper_bound   | float | value of upper bound     |
| lower_bound   | float | value of lower bound     |
| max_point_num | int   | maximum number of points |
| interval      | int   | interval for sampling    |

Example:

```python
Demo_v.config_plot('Open',upper_bound=1000,lower_bound=500,max_point_num=50,interval=10)
```



#### `select_label`

```python
def select_label(self):
        '''
        Function: select the data
        Output: the labels of the selected data
        '''
```

Function:

This function is mainly used to filter the qualified row data, we just need to call this function, it can automatically filter the drawing data.  

这个函数主要用于筛选符合条件的行数据，我们只需要调用这个函数，便能自动对绘图数据进行筛选。

Example:

```python
selected_label=self.select_label()
sub_dataset=self.dataset.loc[selected_label]
```



#### `single`

```python
def single(self,Factor,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (string)Factor: name of the factor
        Output:
        pictures
        '''
```

Function:

The main function of this function is to draw a curve of a single factor. Why do we need to plot a single factor? Because in a real business scenario, the analyst needs to show the movement of an important factor and put the picture on the research report. In general, there will be special marks for particular positions on the curve of this single factor.  

这个函数的主要功能是描绘单一因子的曲线。为什么我们需要为单一因子作图呢？因为在实际业务场景中，分析师需要对某一重要的因子的走势进行展示，并将图片放在研究报告上。一般来说还会为这个单一因子的曲线的某些特殊位置坐上特殊的标记。

Example:

````python
Demo_v.single('Open','Date')

````

```
#### `Multi`

```python
def multi(self,FactorList,label):
        '''
        Function: show the coherance of two factors and draw the graph
        Input:
        (list)Factor: list of names of the factors
        Output:
        pictures
        '''
```

Function:

The main function of this function is to plot a series of curves. When we set the color for each curve, we use RGB for consideration when there are many curves. It doesn't have to be that way. We can use enumerations to do this, because we don't normally pass in more than ten factors. This is because the number of important factors we consider is often less, at the same time, too many curves will also affect the visual effect, reduce man-machine ergonomics, but can not achieve the purpose of visualization work.  

这个函数主要功能是画出一系列的曲线的走势图。我们在为每个曲线设置颜色的时候是用RGB的方式，这是出于曲线个数很多的时候的考虑。事实上不必如此，我们可以使用枚举的方法来达成目的，因为通常情况下我们不会传入超过十个因子。这是因为，我们考虑的重要因子的数目往往较少，同时，曲线数目太多也会影响视觉效果，降低人机工效，反而无法达成做可视化工作的目的。

Example:

```python
Demo_v.multi(FactorList,'Date')
```

#### `heatmap`

```python
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
```

Function:

The purpose of this function is to draw a heat map according to the input matrix, mainly used to analyze the correlation coefficient. The typical application scenario is the process of evaluation of machine learning results. In the process of portfolio configuration, we also use heat maps to show the degree of correlation between assets.  

本函数的用途为：根据输入的矩阵绘制热图，主要用于分析相关系数。典型应用场景为机器学习的结果评估的过程。在资产组合配置的过程中，我们也会使用热图来表示资产之间的相关性大小。

Example:

```
self.heatmap(matrix,'coefficient')
```

#### `cor_total_plot`

Function：

The purpose of this function is to calculate the correlation between factors and draw a heat map of the correlation.  

这个函数的用途是计算各个因子之间的相关性，并绘制相关性的热图。

```python
Demo_v.cor_total_plot()
```

#### `show_stat`

```python
def show_stat(self,FactorList):
        '''
        Function:
        To show the statistic features of the factors
        Input:
        (list)FactorList:the list of factors
        Output: the pictures
        '''
```

Function:

Although the use of statistical graphs can help us to get a good sense of the data, we still need to make a visual representation of the statistics. In this function, we want to use a bar graph to show the maximum, minimum, and average values of each factor in the passed list.  

尽管使用统计图的方式可以帮助我们对数据有个很好的直观感受，但是我们仍然需要对统计量做出直观的表示。在这个函数中，我们希望使用条形图展现传入的列表中各个因子的最大值、最小值、平均值。

Example:

```python
Demo_v.show_stat(FactorList)
```

#### `subline`

```python
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
```

Function:

The function of this function is to reflect the dynamic relationship between the differences of two sequences. Usually used to reflect the difference between a sequence and a reference sequence. For example, we often input the return rate of the portfolio together with the return rate of the benchmark portfolio (such as Shanghai Composite Index) to observe the performance of the strategy when we test back the portfolio.  

这个函数的功能是反映两个序列的差值的动态变化关系。通常情况下，常用于反映某一序列和基准序列的差值。例如，我们常常在回测投资组合的时候，将投资组合收益率和基准组合（例如上证指数）的收益率一起输入，观察策略表现情况。

Example：

```python
Demo_v.subline('Open','High','Date')
```



## Demo data we use

Data：（The dataset is from `kaggle` .）

[NIFTY-50 Stock Market Data (2000 - 2021) | Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)

## Demo Results

#### Demo 1: `coherance`

motivation: 

开盘价（ Open ）和最高价 ( High ) 不一定是密切相关的，不过在大多数股票市场往往存在涨停板限制，因此会有一定程度上相关。为了检验开盘价是否与最高价密切相关，我们对两者调用了`coherance`函数进行可视化处理，并观察其中的特点。

demo code:

```python
#Demo mainly process data and some operations on it
df=pd.read_csv("ADANIPORTS.csv")
Demo_v=visualizer(df,"Demo 0")
Demo_v.print_data()
Demo_v.config_plot('Open',upper_bound=1000,lower_bound=500,max_point_num=50,interval=10)
Demo_v.coherance('Open','High','Date')
```

result:

![Figure_1](E:\大三下\数据挖掘\HW1\asset\Figure_1.png)

figure 2: result of function `coherance`

Analysis:

从图中可以看到如下特点：

- 开盘价相对最高价有滞后性
- 开盘价与最高价相关性较高，二者的相关性维持在0.85以上

我们可以解释为：

开盘价代表了市场参与者对市场价格的预期，而这一预期往往取决于上一个交易日的市场的情况。因此上一个交易日的最高价越高，大概率意味着上一个交易日的成交的价格水平较高，因此市场参与者预期当前交易日的价格也会较高，反之亦然。因此当前开盘价会随上一交易日最高价同向变动。

#### Demo 2: `single`

motivation:

我们希望展示单个因子的走势和趋势、波动等情况。在一些情景中，例如预测期权隐含波动率的时候，我们往往使用自回归族模型，而具体使用其中哪一个，需要观察序列图来初步选择。因此，我们需要单独画出某些因子。

![single](.\asset\single.jpg)

figure 3: result of function `coherance`

Analysis:

- 可以看到开盘价(Open)呈现出来的特点是波动后面往往跟着一些相对小一些的波动，在末尾处存在一些波动上升的趋势。这个性质被称作`波动聚集性`。我们常常用异方差条件自回归在建模这个特性。
- 此外，我们也可以看到序列中的一些突变非常尖锐，并且峰是有偏的，这是我们在之前所提及的尖峰厚尾性，这个主要是设计统计学里面的三阶矩和四阶矩的概念，也是金融时间序列的特性之一。

#### Demo 3: `Multi`

motivation：

我们主要是希望通过将多个序列呈现到一起，观察其相关关系，或者推测其间是否存在`共同演化`的特点，这也是金融时间序列挖掘的重要特征。我在srtp的过程中曾经思考过这一方法的可行性，我认为这与`DTW`等学习形状的方法都较有应用前景，因为共同演化和一些序列中的特征形状，都对应着一些特定市场主体的行为。

![multi](.\asset\multi.png)

figure 4: result of function `Multi`

Analysis:

我们选取的这些序列其实都较为接近，因为他们都是围绕价格这一因素变动的，无非是存在滞后相关性或者存在日内波动的。但是有一点值得我们注意：

- 在出现尖峰的时候，这些价格之间的差距会非常大，这反映出日内的较大波动，说明波动不仅仅是每日开盘时跳变产生的，而是具有持续性的预期的传到的。

#### Demo 4: `heatmap`

motivation：

这里我们使用热图描绘了因子之间的相关性。其实价格之间的高度相关性并不能反映太多的信息，这是因为他们的相关性都太大，熵太小，根据熵权法则，他们在实际决策中应该占有较小的权重。

![heatmap](.\asset\heatmap.png)

figure 5: result of function `heatmap`

Analysis:

根据我们在科研中遇到各个奇怪算法中积累的经验，我们可以把前面几种关系紧密的因子合并在一起作为一个因子，来看这个合成因子和其他因子的关系。我们观察交易量volume，价格和交易量存在较弱的负相关性。两者看似不想关，但是仔细思考，价格越高，消费者越不倾向于购买，因为价格高往往会导致较低的流动性，并且由于资金有限，单个资产价格越高，越不容易配置出有效率的、风险对冲过的资产组合。所以这可能导致交易量和价格存在弱负相关。具体是怎样的机制，还需要我们进行进一步的检验。

#### Demo 5: `show_stat`

motivation:

我们只是为了直观反映出因子的统计特征，因此用条形图的方式来呈现。

![show_stat](.\asset\show_stat.jpg)

figure 6: result of function `show_stat`

#### Demo 6: `subline`

motivation：

正如上文所说，我们经常遇到这样的场景：评估两个序列的好坏，具体而言，就算评估谁的值大或者稳定，等等。因此我们需要对两个序列做差，并将之与原序列比较，来给我们下一步的分析做出直观的支持。

![margin of Open High](.\asset\margin of Open High.jpg)

figure 7: result of function `subline`

Analysis：

从图中可以看到，开盘价和最高价的价差并不大。在开始的时候由于序列整体处于波动状态，两者价差也较大。在中间较为稳定的区间，几乎为0.在之后缓慢上升的区间，两者则具有较为略微的差距。
