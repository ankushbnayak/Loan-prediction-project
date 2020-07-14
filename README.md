## Loan Prediction using Machine Learning

### Project Statement
The idea behind this project is to build a model that will classify how much loan the user can take. It is based on the user’s marital status, education, number of dependents, and employments.
## Algorithm used
### K-Nearest Neighbors (KNN) Algorithm
K-Nearest Neighbors (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function). Classification is done by a majority vote to its neighbors. The data is assigned to the class which has the nearest neighbors. As you increase the number of nearest neighbors, the value of k, accuracy might increase.

### Source to avail the dataset:- [Loan prediction dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)
## Now, let us understand the implementation of K-Nearest Neighbors (KNN) in Python
###  Import the Libraries
We will start by importing the necessary libraries required to implement the KNN Algorithm in Python. We will import the **numpy** libraries for scientific calculation. Next, we will import the **matplotlib.pyplot** library for plotting the graph. We will import two machine learning libraries KNeighborsClassifier from sklearn.neighbors to implement the k-nearest neighbors vote and accuracyscore from **sklearn.metrics** for accuracy classification score.**Seaborn** is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.**pandas** is a Python package providing fast, flexible, and expressive data structures.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
### Read the data
```
df=pd.read_csv("loan_train.csv")
df.head()
```

_#Heat map showing values which are not available or null_
```
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![Image](https://github.com/ankushbnayak/ML_Algos/blob/master/Loan_prediction_project/heatmap1.png)


We have to eliminate these null values

_#The get_dummies() function is used to convert categorical variable into dummy/indicator variables (0's and 1's)._
```
sex = pd.get_dummies(df['Gender'],drop_first=True)
married = pd.get_dummies(df['Married'],drop_first=True)
education = pd.get_dummies(df['Education'],drop_first=True)
self_emp = pd.get_dummies(df['Self_Employed'],drop_first=True)
df.drop(['Gender','Loan_ID','Self_Employed','Married','Education'],axis=1,inplace=True)
df = pd.concat([df,married,education,sex,self_emp],axis=1)
df.head()
```
_#Rename the columns after data manipulation._
```
df=df.rename(columns={'Yes':'Married','Male':'Gender','Yes':'Self_Employed'})
```
_#Heat map after introducing dummy variables._


![Image](https://github.com/ankushbnayak/ML_Algos/blob/master/Loan_prediction_project/Heatmap2.png)





