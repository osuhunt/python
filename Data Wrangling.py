



import pandas as pd
import re
import glob
import copy
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

# Writing a function to import all excel file names into a list

def function1():
    """
    :type : None
    :rtype: List[String]
    """
    return(glob.glob('*.xlsx'))
    
    
    pass
    
filenames = function1()

# Writing a function to return the name of the excel file based on a given string
# (string defined by user)

def function2(files, s):
    """
    :type : List[String], String
    :rtype: String
    """
    for file in files:
        if re.search(s, file):
            return file    
    pass
    
file = function2(filenames, s = "Dictionaries")
# Testing with files in my directory

# Writing a function to load into a Pandas DataFrame

def function3(files, s):
    """
    :type : List[String], String
    :rtype: Pandas DataFrame
    """
    file = function2(filenames, s)
    excel = pd.read_excel(file)
    df = pd.DataFrame(excel)
    return df
    
    pass

functions_df = function3(filenames, s = "Functions")
# I have an excel file named "Functions"



