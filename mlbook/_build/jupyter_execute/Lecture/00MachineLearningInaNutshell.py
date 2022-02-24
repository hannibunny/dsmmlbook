# Machine Learning in a few lines of Code 
* Author: Johannes Maucher
* Last Update: 26.11.2020

## Goal
This notebook shall demonstrate the implementation of a complete Data Mining Process from data access to model evaluation and interpretation.

<img src="https://maucher.home.hdm-stuttgart.de/Pics/crispIndall.png" style="width:600px" align="center">

The steps of the *Cross Industry Standard Process for Datamining (CRISP)* are depicted above. Each of these steps can be quite complex. In the current notebook however, a simple example is used to provide you a glimpse of
* each of the crisp phases
* the basics of Python packages used in the crisp process.


## Python Packages for Data Mining and Machine Learning

For implementing the entire Data Mining process chain in Python the following Python packages are commonly used:
* [numpy](http://www.numpy.org) and [scipy](https://www.scipy.org) for efficient datastructures and scientific calculations
* [pandas](https://pandas.pydata.org) for typical data science tasks, such as data access, descriptive statistics, joining of datasets, correlation analysis, etc.
* [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org/) and [Bokeh](https://bokeh.pydata.org/en/latest/) for visualisation
* [scikit-learn](https://scikit-learn.org/stable/) for conventional Machine Learning, i.e. all but Deep Neural Networks
* [tensorflow](https://www.tensorflow.org) and [keras](https://keras.io) for Deep Neural Networks

### Scikit-Learn
For conventional Machine Learning scikit-learn provides a comprehensive bunch of algorithms and functions. The basic concepts of scikit-learn are:

* it is primarily built on Numpy. In particular internal and external data structures are [Numpy Arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).
* All algorithms, which somehow transform data belong to the **Transformer**-class, e.g. *PCA, Normalizer, StandardScaler, OneHotEncoder*, etc. These transformers are trained by applying the `.fit(traindata)`-method. Once they are trained, there `.transform(data)`-method can be called in order to transform *data*. If the data, used for training the transformer, shall be transformed immediately after training, the `.fit_transform(data)`-method can be applied.
* All Machine Learning algorithms for supervised and unsupervised learning belong to the **Estimator** class, e.g. *LogisticRegression, SVM, MLP, Kmeans*, etc. These estimators are trained by applying the
`.fit(trainfeatures)`- or `.fit(trainfeatures,trainlabels)`-method. The former configuration is applied for unsupervised-, the latter for supervised learning. Once an estimator is trained, it can be applied for clustering, classification or regression by envoking the `.predict(data)`-method. 
* At their interfaces all **Transformers** and **Estimators** apply *Numpy Arrays*.

## Crisp Process in a few lines of code

### Business Understanding

In this example, structured data is available from a .csv file. Data has been collected by a U.S. insurance company. For 1339 clients the following features are contained:
* age
* sex
* smoker
* Body-Mass-Index (BMI)
* Number of children
* living region
* annual charges 

<p style="color:red">The goal is to learn a model, which predicts annual charges of clients from the other 5 features.</p>

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
from warnings import filterwarnings
filterwarnings("ignore")
np.set_printoptions(precision=3)

### Access Data from .csv file
For accessing data [pandas provides comfortable interfaces](https://pandas.pydata.org/docs/reference/io.html) to a wide range of different data formats, such as csv, Excel, Json, SQL, HDF5 and many others. 

Data of this example is available in a csv-file, which can be accessed as follows:

data="../Data/insurance.csv"
insurancedf=pd.read_csv(data,na_values=[" ","null"])
insurancedf.head()

### Understand Data

At the very beginning of each datamining task one should try to understand the given data. This task comprises:
- determine how *clean* the data is: Are there missing values, type-errors, value-errors (outliers), etc. 
- determine descriptive statistics
- investigate correlations

*Data visualistion* can help to clarify these questions.

**Determine Type of Data:**

<img src="https://maucher.home.hdm-stuttgart.de/Pics/dataTypes.png" width="400" align="middle">

In this example features *sex*, *smoker* and *region* are nominal. All other features are numerical.

#### Numeric features:
For numeric variables standard descriptive statistics such as mean, standard-deviation, quantiles etc. are calculated:

insurancedf.describe()

#### Categorical Features:
For non-numeric features the possible values and their count can be calculated as follows:

catFeats=['sex','smoker','region']
for cf in catFeats:
    print("\nFeature %s :"%cf)
    print(insurancedf[cf].value_counts())
    

#### Some visualization:
The standard Python visualization library is [matplotlib](https://matplotlib.org). Many other packages integrate and/or extend matplotlib's capabilities. For example [pandas integrates matplotlib's plot() function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html), such that this function can be invoked on dataframe-objects. 

ax=insurancedf["charges"].plot(figsize=(14,7),title="annual charges per customer",
                            marker=".",linestyle="None")
ax.set_xlabel("Cliend ID")
ax.set_ylabel("Annual Charges (USD)")

Analysing the distribution of single attributes helps to understand data. For example it helps to detect outliers. Outliers should be removed from the data, since they may yield disturbed models. Moreover, knowing the univariate distribution may help us in determining necessary preprocessing steps, such as standardization. For classification tasks, the distribution of the class-labels within the training set is a critical point. In the case of extremely unbalanced label-distributions under- or oversampling can be applied for balancing. 

Univariate distributions can be visualized by e.g. histograms, boxplots or violinplots as demonstrated below in the code-cells below:

Among other [plot-kinds](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) boxplots can be generated for dataframe-columns:

insurancedf["charges"].plot(kind="box")

Above, the *Pandas* `plot()`-function, which applies *Matplotlib* `plot()`-function is applied for generating the Box-plot. *Seaborn* is another visualisation lib for Python, which is particularly dedicated for statistical visualisations. E.g. it provides more functions to visualize data-distributions and -correlations. Below, a seaborn-*violinplot* is generated: 

sns.violinplot(y=insurancedf["charges"])
plt.show()

### Preprocess Data

#### Transformation of non-numeric Features
Non-numeric features must be transformed to a numeric representation. For this we apply the `LabelEncoder` from scikit-learn, which belongs to the class of *Transformers*:

from sklearn.preprocessing import LabelEncoder
for cf in catFeats:
    insurancedf[cf] = LabelEncoder().fit_transform(insurancedf[cf].values)

insurancedf.head()

#### One-Hot-Encoding of nominal Features

For **non-binary nominal features** a transformation into a numeric value is not sufficient, because algorithms interpret integers as ordinal data. Therefore non-binary nominal features must be **One-Hot-Encoded**. For columns of pandas dataframes the `get_dummies()`-function does the job. In the code-cell below the columns are reordered after One-Hot-Encoding, such that the attribute, which shall be predicted (charges) remains the last column:

insurancedfOH=pd.get_dummies(insurancedf,columns=["region"])
insurancedfOH.head()
ch=insurancedfOH["charges"]
insurancedfOH.drop(labels=['charges'], axis=1, inplace = True)
insurancedfOH.insert(len(insurancedfOH.columns), 'charges', ch)
insurancedfOH.head()

```{note} 
Theory says that nominal features must be One-Hot-encoded. However, in practice prediction-accuracy may be better if One-Hot-encoding is not applied. In order to find out, which option is better, both variants must be implemented and evaluated. Below, the non-One-Hot-Encoded dataset `insurancedf` is applied for modelling. Apply also the One-Hot-encoded dataset `insurancedfOH` and determine, which variant performs better.
```

#### Scaling of data
Except decision trees and ensemble methods, which contain decision trees, nearly all machine learning algorithms require features of similar scale at the input. Since the value ranges of practical data can be very different a corresponding scaling must be performed in the preprocessing chain. The most common scaling approaches are *normalization (MinMax-scaling)* and *standardization*.

**Normalization:** In order to normalize feature *x* it's minimum $x_{min}$ and maximum $x_{max}$ must be determined. Then the normalized values $x_n^{(i)}$ are calculated from the original values $x^{(i)}$ by

$$
x_n^{(i)}=\frac{x^{(i)}-x_{min}}{x_{max}-x_{min}}.
$$

The range of normalized values is $[0,1]$. A problem of this type of scaling is that in the case of outliers the value range of non-outliers may be very small. 

**Standardization:** In order to standardize feature *x* it's mean value $\mu_x$ and standard deviation $\sigma_x$ must be determined. Then the standardized values $x_s^{(i)}$ are calculated from the original values $x^{(i)}$ by

$$
x_s^{(i)}=\frac{x^{(i)}-\mu_x}{\sigma_x}
$$

All standardized features have zero mean and a standard deviation of one.

from sklearn.preprocessing import MinMaxScaler, StandardScaler

normalizer = MinMaxScaler()
normalizer.fit(insurancedf)
insurancedfNormed = normalizer.transform(insurancedf)
print("Min-Max Normalized Data:")
insurancedfNormed

standardizer = StandardScaler()
standardizer.fit_transform(insurancedf)
insurancedfStandardized = standardizer.transform(insurancedf)
print("Standardized Data:")
insurancedfStandardized

```{note}
As can be seen above, both transformers must be fitted to data by applying the `fit()`-method. Within this method the parameters for the transformation must be learned. These are the columnwise `min` and `max` in the case of the `MinMaxScaler` and the columnwise `mean` and `standard-deviation` in the case of the `StandardScaler`. Once these transformers are fitted (i.e. the parameters are learned), the `transform()`-method can be invoked for actually transforming the data. It is important, that in the context of Machine Learning, the `fit()`-method is only invoked for the training data. Then the fitted transformer is applied to transform **training- and test-data**. It is not valid to learn individual parameters for test-data, since in Machine Learning we pretend test-data to be unknown in advance. 
```

### Modelling
In this example a regression-model shall be learned, which can be applied to estimate the annual charges, given the other 6 features of a person. Since we also like to evaluate the learned model, we have to split the set of all labeled data into 2 disjoint sets - one for training and the other for test.

```{note}
Since the goal of this section is to keep things as simple as possible, we neglect One-Hot-Encoding and Scaling here. In an offline experiment it has been shown, that for this data and the applied ML-algorithm, the two transformations yield no significant performance difference.
```

from sklearn.model_selection import train_test_split

Split input-features from output-label:

X=insurancedf.values[:,:-1] # all features, which shall be applied as input for the prediction
y=insurancedf.values[:,-1]  # annual charges, i.e. the output-label that shall be predicted

Note that in the code cell above, the `values`-attribute of pandas dataframes has been invoked. This attribute contains only the data-part of a pandas-dataframe. The format of this data-part is a numpy-array. I.e. the variables `X`and `y` are numpy-arrays:

Split training- and test-partition:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

First 5 rows of the training-partition:

X_test[:5,:]

In scikit-learn a model is learned by calling the `fit(X,y)`-method of the corresponding algorithm-class. The arguments $X$ and $y$ are the array of input-samples and corresponding output-labels, respectively.

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)

In the same way as `LinearRegression` has been applied in the code cell above, any regression algorithm, provided by [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) can be imported and applied. Even conventional feed forward neural networks such as the [Multi Layer Perceptron (MLP) for Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) are provided.

### Evaluation
Once the model has been learned it can be applied for predictions. Here the model output for the test-data is calculated:

ypred=linreg.predict(X_test)

Next, for the first 10 persons of the test-partition the prediction of the model and the true charges are printed:

for pred, target in zip(ypred[:10],y_test[:10]):
    print("Predicted Charges: {0:2.2f} \t True Charges: {1:2.2f}".format(pred,target))

[scikit-learn provides a bunch of metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) for evaluation classification-, regression- and clustering models. For this task we apply the `mean_absolute_error`:

from sklearn.metrics import mean_absolute_error

mae=mean_absolute_error(ypred,y_test)
print(mae)

### Visualisation

For all test-datasets the true-charges are plotted versus the predicted charges. The blue line indicates `predicted=true`:

plt.figure(figsize=(12,10))
plt.plot(ypred,y_test,"ro",alpha=0.5)
plt.plot([np.min(y_test),np.max(y_test)],[np.min(y_test),np.max(y_test)])
plt.xlabel("Predicted Charges")
plt.ylabel("True Charges")
plt.title("Estimated vs. True Charges")
plt.show()

Finally, we split smokers from non-smokers and analyse the model's prediction for both partitions:

y_test_smoker=y_test[X_test[:,4]==1]
y_pred_smoker=ypred[X_test[:,4]==1]

y_test_nonsmoker=y_test[X_test[:,4]==0]
y_pred_nonsmoker=ypred[X_test[:,4]==0]

plt.figure(figsize=(10,8))
plt.plot(y_pred_smoker,y_test_smoker,"ro",label="smoker")
plt.plot(y_pred_nonsmoker,y_test_nonsmoker,"go",label="non smoker")
plt.plot([np.min(y_test),np.max(y_test)],[np.min(y_test),np.max(y_test)])
plt.xlabel("Predicted Charges")
plt.ylabel("True Charges")
plt.title("Estimated vs. True Charges")
plt.legend()
plt.show()

## Appendix
### Modelling of words and documents
In the example above different types of data, numeric and categorial, have been applied. It has been shown how categorical data is mapped to numeric values or numeric vectors, such that it can be applied as input of a Machine Learning algorithm.

Another type of data is text, either single words, sentences, sections or entire documents. How to map these types to numeric representations?

#### One-Hot-Encoding of Single Words
A very simple option for representing single words as numeric vectors, is One-Hot-Encoding. This type of encoding has already been introduced above for modelling non-binary categorial features. Each possible value (word) is uniquely mapped to an index, and the associated vector contains only zeros, except at the position of the value's (word's) index.

For example, assume that the entire set of possible words is 

$$
V=(\mbox{all, and, at, boys, girls, home, kids, not, stay}).
$$

Then a possible One-Hot-Encoding of these words is then

|       |   |   |   |   |   |   |   |   |   |
|-------|---|---|---|---|---|---|---|---|---|
| all   | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| and   | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| at    | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| boys  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| girls | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| home  | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| kids  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| not   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| stay  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

simpleWordDF=pd.DataFrame(data=["all", "and", "at", "boys", "girls", "home", "kids", "not", "stay"])
simpleWordDF

pd.get_dummies(simpleWordDF,prefix="")

#### Word Embeddings

One-Hot-Encoding of words suffer from crucial drawbacks: 

1. The vectors are usually very long - there length is given by the number of words in the vocabulary. Moreover, the vectors are quite sparse, since the set of words appearing in one document is usually only a very small part of the set of all words in the vocabulary.
2. Semantic relations between words are not modelled. This means that in this model there is no information about the fact that word *car* is more related to word *vehicle* than to word *lake*. 
3. In the BoW-model of documents word order is totally ignored. E.g. the model can not distinguish if word *not* appeared immediately before word *good* or before word *bad*.  

All of these drawbacks can be solved by applying *Word Empeddings* and by the way the resulting *Word Empeddings* are passed e.g. to the input of Recurrent Neural Networks, Convolutional Neural Networks or Transformers (see later chapters of this lecture). 

A Word-Embedding maps each word to a dense numeric vector of relatively small size (typical length is 200). The main advantage of these word vectors is that, vectors of similar words are close together in the Euclidean space, whereas vectors of unrelated words are far apart from each other. Word Embeddings are learned from large text-corpora (e.g. the entire Wikipedia) by applying Neural Networks. Learned Word Embeddings are available online. For example the [FastText project](https://fasttext.cc/) provides Word-Embeddings for 157 different languages.

#### Bag of Word Modell of documents

The conventional model for representing texts of arbitrary length as numeric vectors, is the **Bag-of-Words** model. 
In this model each word of the underlying vocabulary corresponds to one column and each document (text) corresponds to a single row of a matrix. The entry in row $i$, column $j$ is just the frequency of word $j$ in document $i$. 

For example, assume, that we have only two documents

* Document 1: *not all kids stay at home*
* Document 2: *all boys and girls stay not at home*

The BoW model of these documents is then

|            | all | and | at   | boys | girls | home | kids | not  | stay |
|------------|-----|-----|------|------|-------|------|------|------|------|
| Document 1 | 1   | 0   | 1    | 0    | 0     | 1    | 1    | 1    | 1    |
| Document 2 | 1   | 1   | 1    | 1    | 1     | 1    | 0    | 1    | 1    |
 


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

corpus = ['not all kids stay at home.',
          'all boys and girls stay not at home.',
         ]
BoW = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()

BoW.toarray()