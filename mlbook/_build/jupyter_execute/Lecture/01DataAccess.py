<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Data-Access-and-Understanding,-Preprocessing" data-toc-modified-id="Data-Access-and-Understanding,-Preprocessing-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Access and Understanding, Preprocessing</a></span></li><li><span><a href="#Required-Modules" data-toc-modified-id="Required-Modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Required Modules</a></span></li><li><span><a href="#Access-Data" data-toc-modified-id="Access-Data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Access Data</a></span></li><li><span><a href="#Understand-data" data-toc-modified-id="Understand-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Understand data</a></span><ul class="toc-item"><li><span><a href="#Dealing-with-NaNs" data-toc-modified-id="Dealing-with-NaNs-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Dealing with NaNs</a></span></li><li><span><a href="#Descriptive-Statistics" data-toc-modified-id="Descriptive-Statistics-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Descriptive Statistics</a></span></li><li><span><a href="#Visualize-Data" data-toc-modified-id="Visualize-Data-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Visualize Data</a></span><ul class="toc-item"><li><span><a href="#Univariate-Distribution" data-toc-modified-id="Univariate-Distribution-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Univariate Distribution</a></span></li><li><span><a href="#Correlation-between-features-and-target" data-toc-modified-id="Correlation-between-features-and-target-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Correlation between features and target</a></span></li><li><span><a href="#Pairwise-Correlations-between-features" data-toc-modified-id="Pairwise-Correlations-between-features-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Pairwise Correlations between features</a></span></li></ul></li><li><span><a href="#One-step-EDA-with-pandas_profiling" data-toc-modified-id="One-step-EDA-with-pandas_profiling-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>One-step EDA with pandas_profiling</a></span></li></ul></li><li><span><a href="#Transform-Data" data-toc-modified-id="Transform-Data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Transform Data</a></span><ul class="toc-item"><li><span><a href="#Transform-categorical-data" data-toc-modified-id="Transform-categorical-data-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Transform categorical data</a></span><ul class="toc-item"><li><span><a href="#Mapping-of-ordinal-features-to-integers-according-to-their-ordering." data-toc-modified-id="Mapping-of-ordinal-features-to-integers-according-to-their-ordering.-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Mapping of ordinal features to integers according to their ordering.</a></span></li><li><span><a href="#Mapping-of-nominal-variables-to-integers" data-toc-modified-id="Mapping-of-nominal-variables-to-integers-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Mapping of nominal variables to integers</a></span></li><li><span><a href="#One-Hot-Encoding-of-nominal-features" data-toc-modified-id="One-Hot-Encoding-of-nominal-features-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>One-Hot-Encoding of nominal features</a></span></li></ul></li><li><span><a href="#Scaling-of-data" data-toc-modified-id="Scaling-of-data-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Scaling of data</a></span></li></ul></li></ul></div>

# Data Access and Understanding, Preprocessing 
* Author: Johannes Maucher
* Last Update: 06.07.2018

[Go to Workshop Overview (.ipynb)](Overview.ipynb) / [[.html]](Overview.html)

# Required Modules

%matplotlib inline
import pandas as pd
from IPython.display import display
from IPython.display import Image
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
sb.set(style="ticks")

import warnings
warnings.filterwarnings("ignore")

# Access Data

In this workshop **regression models** are trained and evaluated by the example application **Estimation of rental bikes per day**. The task is to predict the daily count of rental bikes from features describing the weather situation and the date and season. The applied dataset is available from [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). After downloading and storing the corresponding .csv-file it can be accessed using *Pandas*:  

bikefile="./Data/bikeday.csv"
bikedf=pd.read_csv(bikefile,index_col=0,na_values=[" ","null"])
display(bikedf.head())

Each of the 731 rows (2years) corresponds to one day. 

**Feature Description:**

- dteday : date
- season : season (1:spring, 2:summer, 3:fall, 4:winter)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not 
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit : 
    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)

**Target:**

- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered

In our regression - experiments only *cnt* is applied as target value. The distinction into *casual* and *registered* is neglected. 



# Understand data
At the very beginning of each datamining task one should try to understand the given data. This task comprises:
- determine how *clean* the data is: Are there missing values, type-errors, value-errors (outliers), etc. 
- determine descriptive statistics
- investigate correlations

*Data visualistion* can help to clarify these questions.

## Dealing with NaNs
Pandas provides comfortable methods to determine and process missing data. In order to demonstrate this, we access a dummy .csv-file with missing values.

playdata1=pd.read_csv("nandata.csv")
display(playdata1)

The number of missing data per column can be determined as follows:

playdata1.isnull().sum(axis=0)

There is 1 missing value in column A, and there are 2 missing values in column B.

Now let's check if there are missing-values in the bike-share dataset:

bikedf.isnull().sum()

There are different approaches to handle missing values. For example, one can just remove all rows with missing values:

print(playdata1.dropna(axis=0))

A less convenient approach is to remove all columns with missing values:

print(playdata1.dropna(axis=1))

We can even remove only rows in which values of a defined subset of columns are missing: 

print(playdata1.dropna(subset=['A']))

Pandas' *fillna()*-method can be applied for replacing NaNs by arbitrary values: 

print(playdata1.fillna(-99))

Another approach is to apply scikit-learn's [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)-class. In the example below missing values are filled with the mean-value of the column. Moreover, instead of filling in the *mean* it is also possible to choose *median* or *most_frequent*.

from sklearn.preprocessing import Imputer
fillerMean = Imputer(missing_values="NaN", strategy='mean', axis=0)
fillerMean.fit(playdata1)
dataImpMean = fillerMean.transform(playdata1.values)
print(dataImpMean)

fillerMed = Imputer(missing_values='NaN', strategy='median', axis=0)
fillerMed.fit(playdata1)
dataImpMed = fillerMed.transform(playdata1.values)
print(dataImpMed)

## Descriptive Statistics
The number of rows and columns of a Pandas dataframe can be determined by it's *shape*-parameter. The column-names are stored in the *columns*-parameter.

print("Number of rows:    ",bikedf.shape[0])
print("Number of columns: ",bikedf.shape[1])
print(bikedf.columns)

**Basic descriptive statistics** of the dataframe's columns are calculated by the *describe()*-method:

print("Descriptive statistics on columns:")
display(bikedf.describe())

## Visualize Data
In order to understand data, a variety of different visualisation techniques can be applied, e.g. histograms, scatter-plots or box-plots.

### Univariate Distribution
Analysing the distribution of single attributes helps to understand data. For example it helps to detect outliers. Outliers should be removed from the data, since they may yield disturbed models. Moreover, knowing the univariate distribution may help us in determining necessary preprocessing steps, such as standardization. For classification tasks, the distribution of the class-labels within the training set is a critical point. In the case of extremely unbalanced label-distributions under- or oversampling can be applied for balancing. 

Univariate distributions can be visualized by e.g. histograms, boxplots or violinplots as demonstrated below in the code-cells below:

**Violinplot of bike-rentals per day.** The 3 violins refer to rentals from casual users, rentals from registered users and their sum.

plt.figure(figsize=(14,7))
sb.set_style("darkgrid")
sb.set_context("notebook")
sb.violinplot(data=bikedf.iloc[:,-3:])
plt.title("Distribution of bike rentals per day: casual users, registered users and their sum")
plt.ylabel("Rentals per day")
plt.show()

**Boxplot of numeric features:**

plt.figure(figsize=(14,7))
sb.set_style("darkgrid")
sb.boxplot(data=bikedf[["atemp","temp","hum","windspeed"]])
plt.title("Distribution of normalized temperature, humidity and windspeed")
plt.ylabel("Normalized values")
plt.show()

plt.figure(figsize=(14,7))
sb.set_style("darkgrid")
sb.distplot(bikedf["temp"])
plt.title("Distribution of normalized temperature")
plt.xlabel("Normalized temperature")
plt.show()

### Correlation between features and target
In order to analyse pairwise corelations of variables correlation plots are suitable. Below it is shown how to generate a matrix of correlation-plots with *Matplotlib*. In this example the correlation of a single input feature vs. the target variable *daily count of bikes* is plotted.

**Q: Which meaningful correlations are visible?**

print(bikedf.columns)

i=1
plt.figure(figsize=(16,16))
for feat in bikedf.columns[1:-3]:
    plt.subplot(3,4,i)
    x=bikedf[feat].values
    xmin=x.min()
    xmax=x.max()
    diff=(xmax-xmin)/(10.0)
    plt.plot(bikedf[feat].values,bikedf['cnt'].values,'.')
    plt.xlim(xmin-diff,xmax+diff)
    plt.title('count vs. '+feat)
    i+=1

### Pairwise Correlations between features

corrmat=bikedf.iloc[:,1:-3].corr()
plt.figure(figsize=(16,12))

sb.set_style("white") #set seaborn style
# Generate a mask for the upper triangle
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sb.diverging_palette(220, 10, as_cmap=True)
sb.heatmap(corrmat,mask=mask,cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

## One-step EDA with pandas_profiling

#!pip install pandas_profiling

import pandas_profiling
eda=pandas_profiling.ProfileReport(bikedf)
#eda
eda.to_file(output_file="bikeRentalDailyEDA.html")

# Transform Data
Most machine learning algorithms require a vector of numeric values at their input. However, data is not always numerical. Therefore, corresponding transformation techniques to convert categorical data into suitable data-structures must be applied in the pre-processing phase.
## Transform categorical data

Image(filename='./Pics/dataTypes.png', width=300)

In the dataframe defined below, *color* and *type* are nominal features, *size* is an ordinal feature and *prize* is numerical.

df = pd.DataFrame([['pullover','blue', 'S', 39.90, 'sale'], ['short','red', 'L', 19.95, 'new'], ['shirt','green', 'M', 14.99, 'new'], ['shirt','blue', 'S', 11.99, 'new']])
df.columns = ['type','color', 'size', 'prize', 'class label']
df

### Mapping of ordinal features to integers according to their ordering. 
Since the ordering must be defined by the user, there can not be an automatic process for this. Instead a corresponding mapping, e.g. in form of a python-dictionary, must be defined. In the following code-snippet such a mapping and the replacement of the string-values by the corresponding integers in the dataframe is implemented.

size_mapping = {
    'L': 3,
    'M': 2,
    'S': 1}
df['size'] = df['size'].map(size_mapping)
display(df)
print("Type of columns: ",df.dtypes)

The inverse mapping can be realised in a similar way:

inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(inv_size_mapping)
df['size'] = df['size'].map(inv_size_mapping)
display(df)

### Mapping of nominal variables to integers
The [LabelEncoder of scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) can be applied to transform non-numeric values of nominal variables to integers:

from sklearn.preprocessing import LabelEncoder
#import sklearn
X= df.values
display(df)
df['size'] = df['size'].map(size_mapping)

enclabel_type= LabelEncoder()
df['type'] = enclabel_type.fit_transform(X[:,0])
enclabel_color= LabelEncoder()
df['color'] = enclabel_color.fit_transform(X[:,1])
enclabel_class= LabelEncoder()
df['class label'] = enclabel_class.fit_transform(X[:,4])
display(df)

The `classes_`-attribute of the *LabelEncoder*, defines the mapping from original value to the assigned integer. The i.th element of the `classes_`-list is the name of the original feature value:

print("Label Mapping of feature type:",enclabel_type.classes_)
print("Label Mapping of feature color:",enclabel_color.classes_)
print("Label Mapping of class:",enclabel_class.classes_)

The inverse mapping can be realized as follows:

newDf=df.copy()
newX=newDf.values
newDf['type'] = enclabel_type.inverse_transform(newX[:,0].astype(int))
newDf['color'] = enclabel_color.inverse_transform(newX[:,1].astype(int))
newDf['class label'] = enclabel_class.inverse_transform(newX[:,4].astype(int))

display(newDf)

### One-Hot-Encoding of nominal features

Now there are only numeric values, but is this a sufficient representation to train a machine learning algorithm?

The answer is: *No*! Usually machine learning algorithms expect continous input. They interpret the values to be ordered. In the example above this would mean that e.g. *pullover* is smaller than *shirt* and *pullover* is closer to *short* than to *shirt*. In order to avoid such miss-interpretations, nominal features are *One-Hot-encoded*. The [scikit-learn OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) transforms each categorical feature with *m* possible values into *m* binary features, with only one active. As the following code snippet demonstrates, one can specify which features shall be one-hot encoded, by assigning the corresponding column-list to the *categorical_features*-parameter. By setting the parameter *sparse=False*, the transformed data is represented as a 2-dimensional numpy-array. Otherwise a sparse-representation would be calculated.   

from sklearn import compose, preprocessing
X= df.values
print(X)

columnsToEncode=[0,1]

oheTransformer = compose.make_column_transformer(
    (preprocessing.OneHotEncoder(categories="auto"), columnsToEncode), remainder="passthrough"
)
X = oheTransformer.fit_transform(X)
#print("Feature indices (i.th feature starts at the position specified in the i.th element of this vector) = ",oheTransformer.feature_indices_)
#print("Regular representation of One-Hot-Encoded Data:")
print(X)

In the example above the first categorical variable is feature 0. There are 3 different values for this feature. Hence, OneHot-Encoding represents this single feature by 3 columns. For a value of 0 there is a 1 in the first column, for a value of 1, there is a 1 in the second column and for a value of 2 there is a 1 in the third column. The same procedure is performed for the second categorial feature at column 1 in the original array. This categorical feature also has 3 different values. The first 6 columns in the new array correspond to the 2 categorical features, the following columns are the features, which are not OneHot-encoded.

> **Note:** Above one-hot-encoding as provided by scikit-learn has been demonstrated. We will apply this function later on, when we build scikit-learn pipelines. The drawback of the scikit-learn function is that it doesn't regard column-names. In order to map the new columns to meaningful names, we implemented our own function `convert2OneHotFeatureNames()`. A better alternative would be the pandas function `get_dummies()`. It provides one-hot-encoding and a corresponding extension of column-names. The use of `get_dummies()` is demonstrated in the code-cell below. 

indfFOH=pd.get_dummies(df,columns=["type","color"])
display(indfFOH)

## Scaling of data
Except decision trees and ensemble methods, which contain decision trees, nearly all machine learning algorithms require features of similar scale at the input. Since the value ranges of practical data can be very different a corresponding scaling must be performed in the preprocessing chain. The most common scaling approaches are *normalization (MinMax-scaling)* and *standardization*.

**Normalization:** In order to normalize feature *x* it's minimum $x_{min}$ and maximum $x_{max}$ must be determined. Then the normalized values $x_n^{(i)}$ are calculated from the original values $x^{(i)}$ by  
$$x_n^{(i)}=\frac{x^{(i)}-x_{min}}{x_{max}-x_{min}}.$$
The range of normalized values is $[0,1]$. A problem of this type of scaling is that in the case of outliers the value range of non-outliers may be very small. 

**Standardization:** In order to standardize feature *x* it's mean value $\mu_x$ and standard deviation $\sigma_x$ must be determined. Then the standardized values $x_s^{(i)}$ are calculated from the original values $x^{(i)}$ by
$$x_s^{(i)}=\frac{x^{(i)}-\mu_x}{\sigma_x}$$
All standardized features have zero mean and a standard deviation of one.

Create dataframe with 2 features:

from sklearn.preprocessing import MinMaxScaler, StandardScaler
rawdat = pd.DataFrame([[45,67000], [32,37500], [51,82500], [47,112000], [58,1800000]])
rawdat.columns = ['age', 'income p.a']
display(rawdat)

Normalize data:

normalizer = MinMaxScaler()
dat_norm = normalizer.fit_transform(rawdat.values)
print(dat_norm)

Standardize data:

standardizer = StandardScaler()
dat_stand = standardizer.fit_transform(rawdat.values)
print(dat_stand)

