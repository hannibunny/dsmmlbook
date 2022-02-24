#!/usr/bin/env python
# coding: utf-8

# # Ensemble Methods
# In machine learning ensemble methods combine multiple learners into a single one. The underlying idea is that **a complex task can be better mastered by an ensemble as by an individual**, if each member of
# the ensemble learns something different. A group of experts is supposed to be better than an individual with a quite broad knowledge. Obviously, a group of experts is useless, if all experts know the same. **The
# individual knowledge of the ensemble members should be as diverse as possible.** This requirement can be fulllled by the following approaches:
# * Apply **different training algorithms** (e.g. linear classiers, neural networks, SVMs, ...) and/or different configurations of the training algorithms
# * Use **different training sets** for training the individual models and/or weight the samples of the training sets for each learner in a different way. However, for all variants of this type the set of used features is the same
# * For the individual learners use different representations, i.e. **different feature subsets**
# 
# The single learners which constitute the ensemble are usually quite simple. E.g. a common approach is to apply decision trees as weak learners.
# 
# The different types of ensemble learning methods differ in the way how the individual training-sets are designed and how they combine their learned knowledge.
# On an abstract level two different categories of ensemble learning are distinguished:
# * In the **parallel** approach each member performs individual training from scratch. The individual models are combined by weighted averaging. Methods of this class usually apply **bagging** for trainingdata selection.
# * Boosting models are build by **sequentially** learning member models. The model learned in phase $i$ has influence on the training of all following member models and of course on the overall ensemble model. 
# 
# For both categories algorithms for classication as well as regression exist.
# 
# ## Bagging
# Bagging can be considered to be a **parallel** approach: $B$ individual models are learned independent of each other. If all models are learned, the output of the ensemble is usually the average over the individual outputs:
# 
# $$
# f_{bag}(x)=\frac{1}{B}\sum\limits_{b=1}^B f_b(x)
# $$
# 
# Here $x$ is the given input (feature vector), $f_b(x)$  is the output of the $b$.th model and $f_{bag}$ is the output of the ensemble. For bagging, the same learning algorithm (usually decision tree) is applied for all members of the ensemble. Diversity is provided by applying different training sets for each individual learner. The individual training sets are obtained by **bootstraping**: Let
# 
# $$
# T=\lbrace (x_1,r_1),(x_2,r_2),\ldots,(x_N,r_N) \rbrace
# $$ 
# 
# be the set of $N$ available training instances. For each individual learner randomly select a set of $N$ samples with replacement out of $T$. Due to the selection with replacement the training set of a single learner may contain some samples more than once, whereas other samples from $T$ are not included. There are about $N_S=\frac{2}{3}N$ different elements in each bootstrap training set. The individual model will be adapted closer to the statistics of the training samples, that are contained  more than once, wheras the statistics of the not contained samples are disregarded.
# 
# One main benefit of bagging is that it reduces variance. The variance of a learner refers to it's dependence on the training set. The variance is said to be high, if small changes in the training set yield large variations in the learned model and its output. Overfitted models usually have a large variance. 
# 
# Popular representatives of Bagging Machine Learning algorithms  
# 
# * Random Forests
# * Extremely Randomized Trees
# 
# Both of them are based on randomized decision trees. I.e. each individual learner is a randomized decision tree and the overall model is an ensemble of such trees.

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/bagging.png" style="width:600px" align="center">

# ### Random Forest
# In scikit-learn [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) and [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) are implemented. Each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. In addition, when splitting a node during the construction of the tree, the split that is chosen is no longer the best split among all features. Instead, the split that is picked is the best split among a random subset of the features. As a result of this randomness, the bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensating for the increase in bias, hence yielding an overall better model. In contrast to the original publication, the scikit-learn implementation combines classifiers by averaging their probabilistic prediction, instead of letting each classifier vote for a single class (cited from: [http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)).
# 
# ### Extremely Randomized Trees 
# In extremely randomized trees (see [ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) and [ExtraTreesRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor) classes), randomness goes one step further in the way splits are computed. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias (cited from: [http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)).

# ## Boosting
# Boosting can be considered as a sequential approach. Usually a sequence of weak learners, each using the same learning algorithm, is combined such that each learner focuses on learning those patterns, which were unsufficiently processed by the previous learner. The weak classifiers must only be better than chance. The overall boosting algorithm is a linear combination of the individual weak classifiers.
# 
# ### Adaboost
# Adaboost has been one of the first boosting algorithms. It is an ensemble classifier and it is still applied in a wide range of applications, e.g. in Face Detection. The idea of Adaboost is sketched in the figure below: In the first stage (leftmost picture) all training samples are weighted equally. A weak learner is trained with this training set. Training samples, which are misclassified by the first learner obtain a larger weight in the training set of the second learner. Thus the learner will be more adapted to these previously missclassified patterns. The patterns, which are missclassified by the second learner obtain a larger weight in the training set for the third learner and so on. 

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/boostingSchema.PNG" style="width:800px" align="center">

# The final classifier $f_{boost}(x)$ is a linear combination of the weak classifiers $f_m(x)$:
# 
# $$
# f_{boost}(x)=\sum\limits_{m=1}^M \alpha_m f_m(x)
# $$
# 
# Individual learners $f_m(x)$ with a good performance contribute with a larger weight $\alpha_m$ to the overall classifier, than weakly performing learners.

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/adaboostAlg.JPG" style="width:600px" align="center">

# ### Gradient Tree Boosting
# Gradient Tree Boosting is a generalization of boosting to arbitrary differentiable loss functions. The base-learners are Regression Trees. In [scikit-learn GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier) builds an additive model in a forward stage-wise fashion. It allows for the optimization of arbitrary differentiable loss functions. In each stage *n_classes_* regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced. In [scikit-learn GradientBoostingRegression](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) in each stage a regression tree is fit on the negative gradient of the given loss function.

# In[ ]:




