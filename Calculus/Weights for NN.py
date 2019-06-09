"""
Contact:
Zachary Hunt
hunt.590@buckeyemail.osu.edu
"""

# Data
X_train = np.array([[4.7,1.4],[4.5,1.4],[4.9,1.5],[4.0,1.3],[4.1,1.0],[5.1,1.9],[6.1,2.5],[5.5,2.1],[6.0,1.8],[5.8,1.6]])
y_train = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])

# (neural network given)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import add_dummy_feature
from sympy import symbols, lambdify, init_printing, exp
from random import random
init_printing(use_unicode=True)

min_max_scl = MinMaxScaler()
scl = min_max_scl.fit(X_train)
X_train = scl.transform(X_train)
X_train

# Forward Propogation

w1 = random()
w2 = random()
w3 = random()
w4 = random()
w5 = random()
w6 = random()
we1 = random()
we2 = random()
we3 = random()

def activation_func(x):
    return (1 / (1+math.exp(-x)))
h1_weights = np.array([w1, w3, we1])
print(h1_weights)
h2_weights = np.array([w4, w2, we2])
print(h2_weights)
o1_weights = np.array([we3, w5, w6])
print(o1_weights)
X_train_dummy = add_dummy_feature(X_train, value=1.0)
X_train_dummy = np.array(X_train_dummy)
print(X_train_dummy)
summation_operator_h1 = np.dot(X_train_dummy[0],h1_weights)
print(summation_operator_h1)
summation_operator_h2 = np.dot(X_train_dummy[0],h2_weights)
print(summation_operator_h2)
activation_h1 = activation_func(summation_operator_h1)
print(activation_h1)
activation_h2 = activation_func(summation_operator_h2)
print(activation_h2)
inputo1 = np.array([1, activation_h1, activation_h2])
print(inputo1)
output_o1=np.dot(inputo1,o1_weights)
print(output_o1)
activation_output_o1=activation_func(output_o1)

print(activation_output_o1)

"""
[0.07425857 0.34215019 0.67374959]
[0.50818675 0.31987196 0.42043049]
[0.627602   0.891942   0.72627656]
[[1.         0.33333333 0.26666667]
 [1.         0.23809524 0.26666667]
 [1.         0.42857143 0.33333333]
 [1.         0.         0.2       ]
 [1.         0.04761905 0.        ]
 [1.         0.52380952 0.6       ]
 [1.         1.         1.        ]
 [1.         0.71428571 0.73333333]
 [1.         0.95238095 0.53333333]
 [1.         0.85714286 0.4       ]]
0.3679751854092355
0.7269255301681876
0.590969621386675
0.6741302390863725
[1.         0.59096962 0.67413024]
1.6443176093109684
0.8381215793845853
"""

# Calculating Total Error

error = 0.5*((activation_output_o1 - y_train[0])**2)
error = error[0]
error
# 0.013102311530470557

# Calculating the Gradients

w5_grad = activation_h1*(activation_output_o1*(1-activation_output_o1)) * error
w6_grad = activation_h2*(activation_output_o1*(1-activation_output_o1)) * error
we3_grad = (activation_output_o1*(1-activation_output_o1)) * error
we1_grad = (activation_h1*(1-activation_h1))*(we3_grad*we3)
we2_grad = (activation_h2*(1-activation_h2))*(we3_grad*we3)
w1_grad = (X_train[0][0])*(activation_h1*(1-activation_h1))*(we3_grad*we3)
w2_grad = (X_train[0][0])*(activation_h2*(1-activation_h2))*(we3_grad*we3)
w3_grad = (X_train[0][1])*(activation_h1*(1-activation_h1))*(we3_grad*we3)
w4_grad = (X_train[0][1])*(activation_h2*(1-activation_h2))*(we3_grad*we3)

# Updating the Weights

w5_new = w5 - (0.01*w5_grad)
w6_new = w6 - (0.01*w6_grad)
we3_new = we3 - (0.01*we3_grad)
we1_new = we1 - (0.01*we1_grad)
we2_new = we2 - (0.01*we2_grad)
w1_new = w1 - (0.01*w1_grad)
w2_new = w2 - (0.01*w2_grad)
w3_new = w3 - (0.01*w3_grad)
w4_new = w4 - (0.01*w4_grad)

