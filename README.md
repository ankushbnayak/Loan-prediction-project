## Loan Prediction using Machine Learning

### Project Statement
The idea behind this project is to build a model that will classify how much loan the user can take. It is based on the user’s marital status, education, number of dependents, and employments.
## Algorithm used
### K-Nearest Neighbors (KNN) Algorithm
K-Nearest Neighbors (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function). Classification is done by a majority vote to its neighbors. The data is assigned to the class which has the nearest neighbors. As you increase the number of nearest neighbors, the value of k, accuracy might increase.

### Source to avail the dataset:- [Loan prediction dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)
## Now, let us understand the implementation of K-Nearest Neighbors (KNN) in Python
### 1. Import the Libraries
We will start by importing the necessary libraries required to implement the KNN Algorithm in Python. We will import the **numpy** libraries for scientific calculation. Next, we will import the **matplotlib.pyplot** library for plotting the graph. We will import two machine learning libraries KNeighborsClassifier from sklearn.neighbors to implement the k-nearest neighbors vote and accuracyscore from **sklearn.metrics** for accuracy classification score.**Seaborn** is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.**pandas** is a Python package providing fast, flexible, and expressive data structures.
```import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


