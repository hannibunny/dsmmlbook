#!/usr/bin/env python
# coding: utf-8

# # Machine Learning with Python
# 
# * Author: Prof. Dr. Johannes Maucher
# * Email: maucher@hdm-stuttgart.de
# * Last Update: 24.02.2022

# **Goals of this lecture are:**
# 
# * Understand all steps of the datamining process, from data access to visualisation and interpretation of the results.
# * Learn how to implement all of these process steps in Python, applying libraries like [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/), [Scikit-Learn](http://scikit-learn.org/stable/index.html) etc.
# * Learn how to integrate Machine Learning algorithms from scikit-learn into datamining projects.
# * Understand Neural Networks, in particular Deep Neural Networks
# * Learn how to implement neural network- and deep neural network applications with [Keras](https://keras.io/).
# * Understand how to model words and documents for textmining
# * Learn how to implement methods for text classification
# 

# <a id='basic_modules'></a>
# ## Prerequisites
# This course requires basic knowledge of Python, [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/) and [Matplotlib](http://matplotlib.org/). Moreover, the Python Machine Learning framework [Scikit-Learn](http://scikit-learn.org/stable/index.html) will be extensively applied in this course.

# <a id='data_mining'></a>
# ## Contents
# 
# ### Conventional techniques with scikit-learn
# 
# 1. [Basic Concepts of Data Mining and Machine Learning](00BasicConcepts.ipynb)
#     * Definition
#     * Categories
#     * Validation
#     
# 2. [Regression Model](02RegressionPipe.ipynb)
#     * Example Data: Insurance Data
#     * Entire Data Mining process from data access to evaluation  
#     * One-Hot-Encoding
#     * Scaling
#     * Learn and apply Regression Model
#     * Evaluation Metrics for Regression Models
#     
# 3. [Classification Model](03ClassificationPipe.ipynb) 
#     * Example Data: Cleveland Heart Disease Dataset
#     * Cleaning, One-Hot-Encoding
#     * Building a pipeline of modules for scaling, transformation and classification
#     * Evaluation of a Classifier by accuracy, confusion matrix, precision, recall, f1-score
#     * Cross-Validation
#     * Determine feature importance
#     * Fast and efficient model comparison
#     
# 5. [Ensemble Methods: General Concept](05EnsembleMethods.ipynb) 
#     * Categorisation of ensemble machine learning algorithms and the main concepts
#     
# 5. [Hyperparameter Optimisation](05Optimisation.ipynb)  
#     * Example Data: Predict bike rental
#     * Train and evaluate Random Forest Regression model
#     * Error visualisation
#     * Determining feature importance
#     * Hyperparameter Tuning
#     * Fast and efficient model comparison
#     * Comparison with Extremly Randomized Trees
#     
#     
# ### Neural Networks and Deep Neural Networks
# 
# 11. [Conventional Neural Networks](01NeuralNets.ipynb) 
#     * Natural Neuron
#     * General Notions for Artificial Neural Networks
#     * Single Layer Perceptron (SLP)
#         * Architectures for Regression and Classification
#         * Gradient Descent- and Stochastic Gradient Descent Learning
#     * Gradient Descent- and Stochastic Gradient Descent Learning
#     * Multilayer Perceptron (MLP) Architectures for Regression and Classification
#     * Backpropagation-Algorithm for Learning
#     
# 
# 12. [Recurrent Neural Networks (RNN)](02RecurrentNeuralNetworks.ipynb) 
#     * Simple Recurrent Neural Networks (RNNs)
#     * Long short-term Memory Networks (LSTMs)
#     * Gated Recurrent Units (GRUs)
#     * Application Categories of Recurrent Networks
# 
# 
# 13. [Deep Neural Networks: Convolutional Neural Networks (CNN)](03ConvolutionNeuralNetworks.ipynb) 
#     * Overall Architecture of CNNs
#     * General concept of convolution filtering
#     * Layer-types of CNNs: 
#         * Convolution, 
#         * Pooling, 
#         * Fully-Connected
#         
# 
# 14. [MLP and CNN for Object Classification](03KerasMLPandCNNcifar.ipynb)
#     * Example Data: Cifar-10 Image Dataset
#     * Image Representation in numpy
#     * Define, train and evaluate MLP in Keras
#     * Define, train and evaluate CNN in Keras 
#     
#     
# 19. [Apply pretrained CNNs for object classification - original task](04KerasPretrainedClassifiers.ipynb)
#     * Access image from local file system
#     * Download and apply pretrained CNNs for object recognition in arbitrary images
#     
# 
#     
# 20. [Use of pretrained CNNs for object classification - new task: Classify x-ray images of lungs into healthy and covid-19](05KerasPretrainedCovid.ipynb)
#     * Download pretrained feature-extractor (CNN without the classifier part)
#     * Define new classifier architecture and concatenate it with pretrained classifier
#     * Fine-tune network with task-specific data
#     * Apply the fine-tuned network for object-recognition
#     
#     
# 
# 
# 15. [Time-Series Prediction with Recurrent Neural Networks (LSTM) - Prediction of Bike rentals](06KerasLSTMbikeRentalPrediction.ipynb)
#     * Data Visualisation for Time-Series
#     * Building Recurrent Neural Networks with Keras
#     * RNN in many-to-one architecture
#     
#     
#     
# 16. [Modelling of Words and Texts / Word Embeddings](11ModellingWordsAndTexts.ipynb) 
#     * Concept of Word-Embeddings
#     * Skip-Gram and CBOW
#     * Working with pretrained word-embeddings
#     
#     
#     
# 14. [Text Classification with CNNs and LSTMs](08TextClassification.ipynb)
#     * Example Data: IMDB-Movie Reviews for Sentiment Classification
#     * Text preprocessing and representation with Keras
#     * Load and apply pretrained word-embedding
#     * News classification with CNN
#     * News classification with LSTM
#     
# 
# 
# 
# 
# ### Additional Jupyter Notebooks    
#     
# 
# 1. [Feature Selection and Extraction](12FeatureSelection.ipynb)
#     * Univariate Feature Selection Tests
#         * Entropy
#         * Mutual Information
#         * $\chi^2$-Test
#         * F-Measure
#     * Variance Threshold
#     * Feature Selectors in scikitlearn: SelectKBest and SelectPercentile
#     * Principal Component Analysis (PCA)
#     * Linear Discriminant Analysis (LDA)
# 

# In[ ]:




