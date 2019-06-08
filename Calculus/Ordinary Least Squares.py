"""
Contact:
Zachary Hunt
hunt.590@buckeyemail.osu.edu
"""
# User inputs the ‘response’ vector and the ‘design matrix’

import random
import numpy as np
import math
import pandas as pd
import scipy.stats as stats

def OLS(design_matrix, response_vector):
    design_matrix = design_matrix.reshape(len(design_matrix), 1)
    ones = np.ones((len(design_matrix),1))
    X_matrix = np.hstack((ones, design_matrix))
    Y_matrix = response_vector.reshape(len(response_vector), 1)
    ols = np.linalg.inv(np.matmul(np.transpose(X_matrix),X_matrix)).dot(np.matmul(np.transpose(X_matrix), Y_matrix))
    return ols
