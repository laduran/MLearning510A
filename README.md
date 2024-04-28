# MLearning510A
University of WashingtonMachine Learning Certificate 510A

Check this Kaggle for using KNeighborsClassifier
https://www.kaggle.com/code/sreeharicr/kneighborsclassifier-social-network-ads

# Lesson 01 - Intro to Statistical Learning 
1. Course requirements
2. Statistical/machine learning and its possible applications, broadly, and specifically:
   1. Estimating f
   2. Tradeoffs between prediction accuracy and model interpretability
   3. Supervised versus unsupervised learning
   4. Regression versus classification problems
3. Model accuracy
   1. Measuring the quality of Fit
   2. Applying the bias-variance trade off
   3. Applying classification setting
4. Basic skills in R with applied practice in the following:
   1. Vector commands
   2. Matrix operations
   3. Data.frames

## Learning Objectives
   1. Define basic terminology of machine learning.
   2. Describe a typical machine learning modeling process.
   3. Demonstrate the ability to distinguish between classification, regression, and clustering problems.
   4. Demonstrate the ability to determine the accuracy of a model.
   5. Demonstrate understanding of the tradeoff between bias and variance in machine learning models.

## Readings
[ISLR](https://www.statlearning.com/) sections 1. Introduction and 2. Statistical Learning.

# Lesson 02 - Linear Regression

## Learning Topics

1. Simple Linear Regression
2. Estimating the Coefficient
3. Multiple Linear Regression
4. Model Selection and Qualitative Predictors
5. Interactions and Nonlinearity

## Learning Objectives

1. Produce a linear regression model with a statistically significant improvement over the null model (a model without input variables).
2. Identify problems associated with the presence of outliers and collinear variables.
3. Produce a regression model with interaction terms with a statistically significant improvement over a model without interaction terms.

## Readings

[ISLR](https://www.statlearning.com/) section 3. Linear Regression

# Lesson 03 - Classification
## Learning Topics

Introduction to Classification
Logistic Regression and Maximum Likelihood
Multivariate Logistic Regression and Confounding
Case-Control Sampling and Multiclass Logistic Regression
Linear Discriminant Analysis (LDA)
Quadratic Discriminant Analysis (QDA)

## Learning Objectives
1. Distinguish discriminative from generative classification models.
2. Identify scalability as an issue for iteratively reweighted least squares.
3. Produce a classification model evaluation using:
   1. ​    Confusion Matrix
   2. ​    Precision
   3. ​    Recall (sensitivity)
   4. ​    F1 measure
   5. ​    Receiver Operating Characteristic (ROC) curve
   6. ​    Area under a ROC curve 
   7. ​    Precision Recall Curve
4. Produce a linear discriminant analysis model with a statistically significant improvement over the null model (a model without input variables).
5. Produce a quadratic discriminant analysis model with a statistically significant improvement over the null model (a model without input variables).

## Readings
- *ISLR* Section 4. Classification, and attempt exercises

- F1 Curve: [Performance measures in Azure ML: Accuracy, Precision, Recall and F1 ScoreLinks to an external site.](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml)
- Precision Recall Curves: [Precision-recall curves – what are they and how are they used?Links to an external site.](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used)


# Lesson 04 - Model Building Part 1

## Learning Topics
1. Exploratory Data Analysis
2. Detection of outliers
3. Data Preprocessing
4. Data Cleansing and Transformations
5. Dealing with missing data

## Learning Objectives
1. Use normalization as part of the modeling process: min max normalization.
2. Use normalization as part of the modeling process: centering and scaling.
3. Use analysis of variance to compare the performance of a pair of models.
4. Use hold-out validation to compare the performance of a pair of models using a large data set.
5. Normalize data.

## Readings

https://ebookcentral.proquest.com/lib/washington/detail.action?docID=634862
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
https://www.itl.nist.gov/div898/handbook/eda/eda.htm
http://hanj.cs.illinois.edu/cs412/bk3_slides/03Preprocessing.pdf
https://www.researchgate.net/publication/268686755/download

# Lesson 05 - Model Building Part 1
## Learning Topics
1. Define the problem (inputs and outputs) and identify constraints (runtime and storage complexity)
2. Identify the evaluation measure(s)
3. Exploring available data
4. Preprocessing data
5. Constructing models
6. Selecting the final model
7. Evaluating the model (on hold-out test data)
8. Deploying the model
9. Monitor model performance
10. Periodically retraining the model

## Learning Objectives
1. Perform model tuning based on hyper parameters.
2. Select the best model after attempting multiple models.
3. Perform recursive feature elimination, producing a statistically significant improvement over a model without 
4. eature selection.
5. Perform wrapper-based feature selection (backward subset selection), producing a statistically significant improvement over a model without feature selection

## Readings
ISLR sections 6.1, 6.2, 6.3 on Linear Model Selection and Regularization.
Howard Seltman. Experimental Design and Analysis, Chapter 4 Exploratory Data Analysis, July 11, 2018
https://www.stat.cmu.edu/~hseltman/309/Book/chapter4.pdfLinks to an external site.
ian Pei, Micheline Kamber,  Jiawei Han. Data Mining: Concepts and Techniques, 3rd Edition., Chapter 3 Data Preprocessing. Morgan Kaufman 2011.
Simple use of pipeline and 3 different regression models using iris data set
https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.htmlLinks to an external site.
https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-2.htmlLinks to an external site.
https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.htmlLinks to an external site.
Feature Selection
https://machinelearningmastery.com/feature-selection-machine-learning-python/Links to an external site.
https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-modelsLinks to an external site.
https://blog.datadive.net/selecting-good-features-part-i-univariate-selection/Links to an external site.
https://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/Links to an external site.
https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/Links to an external site.
Feature Hashing
https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087Links to an external site.
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.htmlLinks to an external site.
Time Series
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/Links to an external site.
A/B Testing for Data Science
https://towardsdatascience.com/data-science-you-need-to-know-a-b-testing-f2f12aff619a


# Lesson 06 - Model Building Part 2

## Learning Topics
1. Cross-Validation
2. The Validation Set Approach
3. Leave-One-Out Cross-Validation
4. k-fold Cross-Validation
5. Bias-Variance Trade-Off for k-fold Cross-Validation
6. Cross-Validation on Classification Problems
7. The Bootstrap
8. The Validation Set Approach
9. Leave-One-Out Cross-Validation

## Learning Objectives
1. Evaluate performance of the selected model.
2. Use k-fold cross validation to compare the performance of a pair of models.
3. Use repeated k-fold cross validation to compare the performance of a pair of models (small data set).
   
## Readings
ISLR section 5. Resampling Methods


# Lesson 07 - Linear Model Selection and Regularization

## Learning Topics
1. Linear Model Selection and Regularization
2. Model Selection
3. Ridge regression
4. Lasso regression
5. Elastic net regression
6. Subset Selection
7. Shrinkage Methods
8. Dimension Reduction Methods

## Learning Objectives
1. Produce a model with l2 regularization, with a statistically significant improvement over a model without regularization.
2. Produce a model with l1 regularization, with a statistically significant improvement over a model without regularization.
3. Produce a model with both l1 and l2 regularization terms, with a statistically significant improvement over a model without regularization.
4. Produce a logistic regression model with a statistically significant improvement over the null model (a model without input variables).
5. Produce a generalized additive model with a statistically significant improvement over the null model (a model without input variables).

## Readings
ISLR section 6. Linear Model Selection and Regularization

Hastie, T., Tibshirani, R., Friedman, J. 2017. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf). Sections 3.4 (Shrinkage Methods), 18.4 (Linear Classifiers with L1 Regularization)

# Lesson 8 - Dimensionality Reduction

## Learning Topics
1. The Curse of Dimensionality
2. Principal Components Analysis (PCA)
3. PCA for high-dimensional data
4. Probabilistic PCA
5. Kernel PCA

## Learning Objectives
1. Be able to make application decisions regarding principal component analysis to train and test data
2. Produce a dimensionality reduction model.

## Readings
ISLR sections 10.1 The Challenge of Unsupervised Learning, 10.2 Principal Components Analysis and attempt exercises.
Bishop, Christopher, M. Pattern Recognition and Machine Learning. Sections 12.1, 12.2, 12.3. (2006 editionLinks to an external site. available at no charge, 2010 version Links to an external site. available for purchase.)
http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf

https://www.amazon.ca/Pattern-Recognition-Machine-Learning-Christopher/dp/0387310738


# Lesson 9 - Forecasting

## Learning Topics

- Forecasting
  - What is Forecasting?
   - Autocorrelation, Noise and Seasonality
   - Evaluation
   - Smoothing
   - Decomposition
   - ARIMA Family of Models
- Ethical Issues
  - What does it mean for an machine learning model to be fair?


## Learning Objectives
Forecasting: You will be able to:

   - Build time series models given time series data.
   - Check for seasonality and autocorrelation.
   - Determine which time series model should they use in a given context.

Ethical Issues: You will be able to:
   - Make a case that in the given machine learning system the data or the model may be biased towards certain groups.
   - Identify the different notions of explainability in machine learning determine when are they needed.
 

## Readings

Read Sections 1 - 4
[Hands-on Time Series Analysis with Python: From Basics to Bleeding Edge Techniques](https://learning.oreilly.com/library/view/hands-on-time-series/9781484259924/)

Frequent Itemset Mining

## Learning Topics 
Frequent Pattern Set Mining
Maximal and Closed Frequent Itemsets
Apriori
FP Tree
Sequential Pattern Mining
Measures of interestingness
Evaluation

## Learning Objectives
You will be able to:

Extract frequent patterns given a corpus of data.
Find the rules which are interesting and non-obvious for a given domain.

## Before Class 
Please complete the following to prepare for this week's class:

Read Witten 4.5.
Review lecture slides - Assemble your in-class questions.