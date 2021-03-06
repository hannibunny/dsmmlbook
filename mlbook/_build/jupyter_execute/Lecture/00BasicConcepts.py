#!/usr/bin/env python
# coding: utf-8

# # Basic Concepts of Data Mining and Machine Learning 
# * Author: Johannes Maucher
# * Last Update: 26.11.2020
# 

# [Go to Workshop Overview](Overview.ipynb)

# ## Overview Data Mining Process
# The **Cross-industry standard process for data mining (CRISP)** proposes a common approach for realizing data mining projects: 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/CRISPsmall.png" alt="Drawing" style="width: 400px;"/>
# 
# 
# In the first phase of CRISP the overall business-case, which shall be supported by the data mining process must be clearly defined and understood. Then the goal of the data mining project itself must be defined. This includes the specification of metrics for measuring the performance of the data mining project. 
# 
# In the second phase data must be gathered, accessed, understood and described. Quantitiy and qualitity of the data must be assessed on a high-level. 
# 
# In the third phase data must be investigated and understood more thoroughly. Common means for understanding data are e.g. visualization and the calculation of simple statistics. Outliers must be detected and processed, sampling rates must be determined, features must be selected and eventually be transformed to other formats.  
# 
# In the modelling phase various algorithms and their hyperparameters are selected and applied. Their performance on the given data is determined in the evaluation phase. 
# 
# The output of the evaluation is usually fed back to the first phases (business- and data-understanding). Applying this feedback the techniques in the overall process are adapted and optimized. Usually only after several iterations of this process the evaluation yields good results and the project can be deployed.

# ## Machine Learning: Definition, Concepts, Categories
# 
# ### Definition
# There is no unique definition of Machine Learning. One of the most famous definitions has been formulated in [Tom Mitchell, Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html):
# 
# 
# * A computer is said to learn from **experience E** with respect to some **task T** and some **performance measure P** , if its performance on T, as measured by P, improves with experience E.
# 
# 
# This definition has a very pragmatic implication: At the very beginning of any Machine Learning project one should specify T, E and P! In some projects the determination of these elements is trivial, in particular the *task T* is usually clear. However, the determination of *experience E* and *performance measure P* can be sophisticated. Spend time to specify these elements. It will help you to understand, design and evaluate your project. 
# 
# **Examples:** What would be T, E and P for
# * a spam-classifier
# * an intelligent search-engine, which provides individual results on queries
# * a recommender-system for an online-shop

# ### Categories
# 
# The field of Machine Learning is usually categorized with respect to two dimensions: The first dimension is the question *What shall be learned?* and the second asks for *How shall be learned?*. The resulting 2-dimensional matrix is depicted below:
# 
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/mlCategories.png" style="width:800px" align="center">

# On an abstract level there exist 4 answers on the first question. One can either learn 
# 
# * a classifier, e.g. object recognition, spam-filter, Intrusion detection, ...
# * a regression-model, e.g. time-series prediction, like weather- or stock-price forecasts, range-prediction for electric vehicles, estimation of product-quantities, ...
# * associations between instances, e.g. document clustering, customer-grouping, quantisation problems, automatic playlist-generation, ....
# * associations between features, e.g. market basket analysis (customer who buy cheese, also buy wine, ...)
# * strategie, e.g. for automatic driving or games 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/classReg.PNG" alt="Drawing" style="width: 800px;"/>
# 
# 
# On the 2nd dimension, which asks for *How to learn?*, the answers are:
# 
# * supervised: This category requires a *teacher* who provides labels (target-values) for each training-element. For example in face-recognition the teacher most label the inputs (pictures of faces) with the name of the corresponding persons. In general labeling is expensive and labeled data is scarce. 
# * unsupervised learning: In this case training data consists only of inputs - no teacher is required for labeling. For example pictures can be clustered, such that similar pictures are assigned to the same group.
# * Reinforcement learning: In this type no teacher who lables each input-instance is available. However, there is a critics-element, which provides feedback from time-to-time. For example an intelligent agent in a computer game maps each input state to a corresponding action. Only after a possibly long sequence of actions the agent gets feedback in form of an increasing/decreasing High-Score.  

# #### Supervised Learning
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearning.png" style="width:800px" align="center">

# **Apply Learned Modell:**
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningApply.png" style="width:800px" align="center">

# #### Unsupervised Learning
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningUnsupervised.png" style="width:800px" align="center">

# **Apply learned Model:**
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningUnsupervisedApply.png" style="width:800px" align="center">

# #### Reinforcement Learning
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/bogenschiessen.jpg" style="width:500px" align="center">

# ### General Scheme for Machine Learning
# In Machine Learning one distinguishes  
# * training-phase, 
# * test-phase 
# * operational phase.
# 
# Training and test are shown in the image below. In the training phase training-data is applied to learn a general model. The model either describes the structure of the training data (in the case of unsupervised learning) or a function, which maps input-data to outputs. Once this model is learned it can be applied in the operational phase to map new input-instances to output values (classes-index, cluster-index or numeric function-value). Before applying a learned model in operation it must be tested. In the case of supervised learning testing compares for all test-data the output of the model with the target output. This means that testing also requires labeled data. Test-data and training-data must be disjoint.

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/Learning.png" alt="Drawing" style="width: 800px;"/>

# As shown in the picture above, usually the available data can not be passed directly to the machine-learning algorithm. Instead it must be processed in order to transform it to a corresponding format and to extract meaningful features. The usual formal, accepted by all machine-learning algorithms is a 2-dimensional array, whose rows are the instances (e.g. documents, images, customers, ...) and whose columns are the features, which describe the instances (e.g. words, pixels, bought products, ...): 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/mlDataStructure.PNG" alt="Drawing" style="width: 800px;"/>

# ### Cross Validation

# K-fold cross-Validation is the standard validation method if labeled data is rare. The entire set of labeled data is partitioned into k ($k=10$ in the example below) disjoint subsets. The entire evaluation consists of k iterations. In the i.th iteration, the i.th partition (subset) is applied for validation, all other partitions are applied for training the model. In each iteration the model's performance, e.g. accuracy, is determined on the validation-partition. Finally, the overall performance is the average performance over all k performance values.  
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/CrossValidation.jpg" alt="Drawing" style="width: 800px;"/>
