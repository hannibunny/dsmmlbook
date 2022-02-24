# Machine Learning with Python

* Author: Prof. Dr. Johannes Maucher
* Email: maucher@hdm-stuttgart.de
* Last Update: December, 1st 2020

**Goals of this lecture are:**

* Understand all steps of the datamining process, from data access to visualisation and interpretation of the results.
* Learn how to implement all these process steps in Python, applying libraries like [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/), [Scikit-Learn](http://scikit-learn.org/stable/index.html) etc.
* Understand Machine Learning algorithms, in particular for supervised learning.
* Learn how to integrate these algorithms from scikit-learn into datamining projects.
* Understand Neural Networks, in particular Deep Neural Networks
* Learn how to implement neural network- and deep neural network applications with [Keras](https://keras.io/).
* Understand how to model words and documents for textmining
* Learn how to implement methods for topic extraction and text classification

**Teaching form:** 

* Contents of this course will be tought task-driven. This means that we start from the tasks of the [Live coding exercise-notebooks](#livecoding). In order to solve these tasks theory and programming skills are required from the notebooks in chapter [Data Mining with Python](#data_mining). We will consult these notebooks whenever we require the corresponding skills to solve the tasks of the exercise notebooks. 

**Organisation of Jupyter Notebooks for Lecture, Live-Coding and Assignments**

<img src="https://maucher.home.hdm-stuttgart.de/Pics/MachineLearningLearningPath.png" alt="Drawing" style="width: 800px;"/>

<a id='basic_modules'></a>
## Prerequisites
This course requires basic knowledge in Python and

* [Basics in Numpy (.ipynb)](NP01numpyBasics.ipynb) 
* [Basics in Matplotlib (.ipynb)](PLT01visualization.ipynb) 
* [Basics in Pandas (.ipynb)](PD01Pandas.ipynb)

Moreover, the Python Machine Learning framework [Scikit-Learn](http://scikit-learn.org/stable/index.html) will be extensively applied in this course. The main concepts of this library are:

* it is primarily built on Numpy. In particular internal and external data structures are [Numpy Arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).
* All algorithms, which somehow transform data belong to the **Transformer**-class, e.g. *PCA, Normalizer, StandardScaler, OneHotEncoder*, etc. These transformers are trained by applying the *.fit(traindata)*-method. Once they are trained, there *.transform(data)*-method can be called in order to transform *data*. If the data used for training the transformer shall be transformed immediately after training, the *.fit_transform(data)*-method can be applied.
* All Machine Learning algorithms for supervised and unsupervised learning belong to the **Estimator** class, e.g. *LogisticRegression, SVM, MLP, Kmeans*, etc. These estimators are trained by applying the *.fit(trainfeatures)*- or *.fit(trainfeatures,trainlabels)*-method. The former configuration is applied for unsupervised-, the latter for supervised learning. Once an estimator is trained, it can be applied for clustering, classification or regression by envoking the *.predict(data)*-method. 
* At their interfaces all **Transformers** and **Estimators** apply *Numpy Arrays*.

<a id='data_mining'></a>
## Machine Learning

### Conventional techniques with scikit-learn

1. [Basic Concepts of Data Mining and Machine Learning](00BasicConcepts.ipynb)
    * Definition
    * Categories
    * Validation
    
2. [Regression Model](02RegressionPipe.ipynb)
    * Example Data: Insurance Data
    * Entire Data Mining process from data access to evaluation  
    * One-Hot-Encoding
    * Scaling
    * Learn and apply Regression Model
    * Evaluation Metrics for Regression Models
    
3. [Classification Model](03ClassificationPipe.ipynb) 
    * Example Data: Cleveland Heart Disease Dataset
    * Cleaning, One-Hot-Encoding
    * Building a pipeline of modules for scaling, transformation and classification
    * Evaluation of a Classifier by accuracy, confusion matrix, precision, recall, f1-score
    * Cross-Validation
    * Determine feature importance
    * Fast and efficient model comparison
    
5. [Ensemble Methods: General Concept](05EnsembleMethods.ipynb) 
    * Categorisation of ensemble machine learning algorithms and the main concepts
    
5. [Hyperparameter Optimisation](05Optimisation.ipynb)  
    * Example Data: Predict bike rental
    * Train and evaluate Random Forest Regression model
    * Error visualisation
    * Determining feature importance
    * Hyperparameter Tuning
    * Fast and efficient model comparison
    * Comparison with Extremly Randomized Trees
    


    

2. [Modelling of Words and Texts](11ModellingWordsAndTexts.ipynb)
    * One-Hot-Encoding of Words
    * BoW model of texts
    
    

2. [Feature Selection and Extraction(.ipynb)](02FeatureSelection.ipynb)
    * Univariate Feature Selection Tests
        * Entropy
        * Mutual Information
        * $\chi^2$-Test
        * F-Measure
    * Variance Threshold
    * Feature Selectors in scikitlearn: SelectKBest and SelectPercentile
    * Principal Component Analysis (PCA)
    * Linear Discriminant Analysis (LDA)



6. [Clustering Energy Consumption (.ipynb)](06ClusteringEnergy.ipynb) 
    * Boxplots
    * Enhance data with geo-information
    * Normalization
    * Clusteralgorithms
        - Hierarchical Clustering
        - Kmeans
        - DBSAN
        - Affinity Propagation
    * Visualisation of clusters
    * Dimensionality Reduction
    * Visualisation in Google Maps
    
### Neural Networks and Deep Neural Networks

11. [Neural Networks I: Single Layer Perceptron (SLP)(.ipynb)](SLP.ipynb) 
    * Natural Neuron
    * General Notions for Artificial Neural Networks
    * Single Layer Perceptron (SLP)
        * Architectures for Regression and Classification
        * Gradient Descent- and Stochastic Gradient Descent Learning
    * SLP implementation and demonstration in Python
    * SLP in Scikit-Learn: Implementation, Evaluation, Optimisation

8. [Neural Networks II: Multi Layer Perceptron (MLP)(.ipynb)](MLP.ipynb)
    * MLP Architectures for Regression and Classification
    * Gradient Descent- and Stochastic Gradient Descent Learning
    * MLP implementation and demonstration in Python
    * MLP in Scikit-Learn: Implementation, Evaluation, Optimisation, 
    * Grid Search and Random Search for Hyperparameter-Tuning

9. [Deep Neural Networks: Convolutional Neural Networks(.ipynb)](ConvolutionNeuralNetworks.ipynb) 
    * Overall Architecture of CNNs
    * General concept of convolution filtering
    * Layer-types: 
        * Convolution, 
        * Pooling, 
        * Fully-Connected

10. [Recurrent Neural Networks (RNN)](RecurrentNeuralNetworks.ipynb) 
    * Simple Recurrent Neural Networks (RNNs)
    * Long short-term Memory Networks (LSTMs)
    * Gated Recurrent Units (GRUs)
    * Application Categories of Recurrent Networks

10. [Time-Series Prediction with Recurrent Neural Networks (LSTM) - Prediction of Bike rentals](LSTMbikeRentalPrediction.ipynb)
    * Data Visualisation for Time-Series
    * Building Recurrent Neural Networks with Keras
    * RNN in many-to-one architecture

10. [Time-Series Prediction with Recurrent Neural Networks (LSTM) - Temperature Prediction](LSTMtemperature.ipynb) 
    * Building Recurrent Neural Networks with Keras
    * RNN in many-to-one architecture
    * Weather-Forecasting with MLP, LSTM, GRU

10. [Time-Series Prediction with Recurrent Neural Networks (LSTM) - Stock-Price-Prediction](LSTMstockPricePrediction.ipynb)
    * Building Recurrent Neural Networks with Keras
    * RNN in many-to-one architecture
    * In notebook [LSTM Sequence Modelling on Stock-Price Data](LSTMsequenceModelStock.ipynb) a RNN is applied in many-to-many mode for the same task.

### Define MLP and CNN for Object Classification in Keras

18. [MLP and CNN for Object Classification](K03KerasMLPandCNNcifar.ipynb)

### Apply pretrained Deep Neural Networks

19. [Use of pretrained CNNs for object classification - original task](K04KerasPretrainedClassifiers.ipynb)
20. [Use of pretrained CNNs for object classification - new task: Malaria detection](K04KerasPretrainedMalariaDetection.ipynb)

### Textprocessing-, analysis and -classification

21. [Access RSS Feeds and create corpus](T01crawlRSSFeeds.ipynb) 
    * Crawling of RSS feeds
    * Generate a corresponding RSS feed news corpus
    * *Note:* This notebook just demonstrates how data, which will be applied in the document-classification notebooks, has been collected. 

12. [Bag-of-Word Document Model, Similarity, Topic Extraction](T02topicExtractionRSSfeeds.ipynb) 
    * Access Data from the RSS news corpus (as generated in the previous notebook)
    * Visualization of wordclouds
    * Gensim
    * Calculating document similarity
    * LSI Topic Extraction
    * LDA Topic Extraction

13. [Word Embeddings Theory:](T03DSM.ipynb) 
    * Concept of Word-Embeddings
    * Skip-Gram and CBOW
    * Working with pretrained word-embeddings

13. [Word Embeddings:](T03generateCBOWfromWiki.ipynb)
    * Generate Word Embedding from German Wikipedia Dump
    * Gensim Semantic Similarity analysis based on word-embeddings.

14. [Text Classification with CNNs and LSTMs](T04germanNewsFeedClassification.ipynb) 
    * Text preprocessing and representation with Keras
    * Load and apply pretrained word-embedding
    * News classification with CNN
    * News classification with LSTM

15. [Classification of IMDB Movie Reviews with CNNs](T05CNN.ipynb) 
    * Sentiment analysis on IMDB reviews
    * Load and apply different pretrained word-embeddings
    * Implement and evaluate different CNN architectures

## Mlfow

1. [Logging a scikit-learn model in mlflow](mlflowSklearnSmoker.ipynb)
2. [Logging a keras model in mlflow](mlflowKerasReutersClassification.ipynb)
3. [Analyse mlflow models in jupyter notebook](mlflowAnalyseModels.ipynb)

You may also analyse your mlflow models by typing `mlflow ui` into your console from the directory, which contains your mlflow-directory. Then a local server starts under `http://127.0.0.1:5000`.

<a id='livecoding'></a>
## Live Coding Tasks

1. Exercise 1: [Access and Preprocess](LiveCoding/Exercise1.ipynb)

2. Exercise 2: [Define and Evaluate Processing Pipelines](LiveCoding/Exercise2.ipynb) 

3. Exercise 3: [Hyperparameter Optimization](LiveCoding/Exercise3.ipynb) 

4. Exercise 4: [Implementing Neural Networks with Keras](LiveCoding/Exercise4cifar.ipynb) 