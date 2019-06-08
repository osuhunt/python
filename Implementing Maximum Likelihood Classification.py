"""
Contact:
Zachary Hunt
hunt.590@buckeyemail.osu.edu
"""

# Works with any csv where the last attribute is the class label and assuming all non-
# class attributes are continuous random variables.

# Setting up a dataset to test my function with (using breast cancer data)

import pandas as pd
import numpy as np
import numpy.linalg as linalg
import math
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
full_df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))
cancer_df = full_df.iloc[0:10,1:4]
target_df = [0,0,1,1,1,0,0,1,0,0]
cancer_df['target'] = pd.Series(target_df, index=cancer_df.index)

test_data = full_df.iloc[11, 1:4]
test_data = np.array(test_data)
test_data = np.resize(test_data, (3,1))

train_df = full_df.iloc[0:350, 0:]
test_df = full_df.iloc[350:-1, 0:]

# creating the functions

def MLC(X_values):    
    X = np.array([X_values])
    X_mean = np.mean(X[0], axis=0)
    X_mean = np.resize(X_mean, (len(X_mean),1))
            
    X_cov = np.cov(X[0].transpose())
    
    Results = [X_mean, X_cov]
    
        
    return Results
   
def testing(X, Parameters, Prob):
    Mean = Parameters[0]
    Covariance_Matrix = Parameters[1]
    Probability = Prob
    X = np.resize(np.array([X]), (len(X),1))
    
    
    d = len(X)
    x_sub_mean = X-Mean
    inv_cov = linalg.inv(Covariance_Matrix)
    
    mat_mul1 = np.matmul(x_sub_mean.transpose(), inv_cov)
    mat_mul2 = np.matmul(mat_mul1, x_sub_mean)    
    
    first = -0.5*mat_mul2
    second = np.log(Probability)
    third = -(d/2.0)*np.log(2*math.pi)
    fourth = -(0.5)*np.log(linalg.det(Covariance_Matrix))
    
    return first+second+third+fourth
    


X_train = train_df.iloc[0:,0:-1]
Y_train = train_df.iloc[0:,-1]

Classes = Y_train.unique()
    
Output_Rows_Subsets = []
Prior_Prob = []
    
for i in Classes:
    X_Train_Class = [X_train.iloc[row,0:] for row in range(len(X_train)) if Y_train[row]==i]
    Output_Rows_Subsets.append(X_Train_Class)
    
    len_Classes = [1 for row in range(len(Y_train)) if Y_train[row]==i]
    Prior_Prob.append(len(len_Classes)/len(Y_train))

    
Parameters = [MLC(i) for i in Output_Rows_Subsets]

correct_predictions = 0
total_predictions = 0
for j in range(len(test_df)):
    test_data = test_df.iloc[j,0:-1]
    prob_in_each_class = []
    for i in range(len(Classes)):
        Probability = testing(test_data, Parameters[i], Prior_Prob[i]).item(0)
        prob_in_each_class.append(Probability)
    
    in_class = Classes[prob_in_each_class.index(max(prob_in_each_class))]
    
    if (in_class == test_df.iloc[j,-1]):
        correct_predictions += 1
    total_predictions += 1

print(correct_predictions/total_predictions)

# Results in an accuracy of 94%

# Comparing this accuracy with sklearn
from sklearn import datasets
cancer = load_breast_cancer()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(cancer.data, cancer.target).predict(cancer.data)
Accuracy = (cancer.data.shape[0]-((cancer.target != y_pred).sum()))/(cancer.data.shape[0])
print("Accuracy: {}".format(Accuracy))

# Resuls in an acuracy of 94%, so pretty spot on!
