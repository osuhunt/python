"""
The goal of this project is to predict which members of the Paralyzed
Veterans of America were most likely to donate money in order to raise
as much money as possible. Sending out donation requests
costs money and it's important to be strategic in targeting which members
are most likely to donate and to figure out how much they are likely to donate.
The data set is fairly large and messy dataset with over 96000 records and about 
500 variables. The records were related to the individual members and the variables
(columns) represented things like income, family size, race, etc.
This is the ultimate project because the dataset is messy and big. I not only have
to use classification  algorithms to figure out if someone donated or not
but also had to run a regression to predict how much money the people who were 
predicted to donate would actually donate.
"""

# Data used for this project can be found in the 'python' repository labeled as
# 'cup98LRN.csv' as the training data and 'cup98VAL.csv' as the data used to make
# predictions.

import pandas as pd
import re
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Importing the data

xtrain = pd.read_csv('cup98LRN.csv')
df = pd.DataFrame(xtrain)
test = pd.read_csv('cup98VAL.csv')
df1 = pd.DataFrame(test)

# Health check

# There seems to be a lot of missing data. In order to address this, I am going to test
# how dropping the na values would affect the amount of data
print(len(df), 'rows before dropping na')
# 95412 rows before dropping na
test = df.dropna()
print(len(test), 'rows after dropping na')
# 0 rows after dropping na
"""
There isn't a single row that doesn't have at least one missing value.
Thus, I need to select important features. To do this, I will do a diagnostic
regression test to see which features affect donation amount the most, 
thus affecting whether someone donates at all. This regression test won't be used 
to actually predict donation amounts, rather it'll be used to get an overview of 
which features are most important.
"""

# Feature Selection

columns = df.columns
target_columns = []
regex_expressions = ['CHIL[0-9]+', 'ETHC[1-9]+', 'MARR[0-9]+', 'TPE[0-9]+', 'LFC[0-9]+', 'ETH[0-9]+', 'AGE[0-9]+', 'HV[0-9]+', 'HHD[0-9]+', 'OCC[0-9]+', 'EIC[1-9]+', 'EC[1-9]+', 'ANC[1-9]+', 'HC[1-9]+', 'ETHC[1-9]+']
                
for regex in regex_expressions:
    regex = re.compile(regex)
    for column in columns:
        match_obj = regex.match(column)
        if match_obj:
            target_columns.append(match_obj.group())
# data set-up for the linear regression
x_train = df[target_columns]
y_train = df['TARGET_D']

X2 = sm.add_constant(x_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
# From the table / output, I can take the features with a p value of less than 0.1 for a 90% significance level
sig_features = ['CHIL1', 'CHIL2', 'CHIL3', 'TPE5','LFC10','ETH9','ETH14','HHD3','HHD8','OCC4','OCC6','OCC7','OCC9','OCC12','EC1','ANC12','HC2','HC6']
x_train = df[sig_features]
y_train = df['TARGET_B']

# Readdressing the missing values, I can now drop them (if any) based on the significant features
print(len(x_train), 'rows before dropping na')
# 95412 rows before dropping na
test = x_train.dropna()
print(len(test), 'rows after dropping na')
# 95412 rows after dropping na
# No more missing values!
# Now that I have my significant features, I can test a few different classification algorithms

# Initializing the models

from sklearn.svm import SVC
svc = SVC(class_weight="balanced")
svc.fit(x_train, y_train)
    
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, class_weight="balanced")
rf.fit(x_train, y_train)

# Support Vector Classification
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svc, x_train, y_train, cv=3, scoring='accuracy')
print(scores)
# [0.91378714 0.90997988 0.91205232]
print('After cross validation, the mean accuracy for Support Vector Classification is: ', np.mean(scores))
# After cross validation, the mean accuracy for Support Vector Classification is:  0.9119397797465908

# Logistic Regression
scores = cross_val_score(lr, x_train, y_train, cv=3, scoring='accuracy')
print(scores)
# [0.53777708 0.53908313 0.5177499 ]
print('After cross validation, the mean accuracy for Logistic Regression is: ', np.mean(scores))
# After cross validation, the mean accuracy for Logistic Regression is:  0.5315367036936521

# Random Forest
scores = cross_val_score(rf, x_train, y_train, cv=3, scoring='accuracy')
print(scores)
# [0.91630247 0.91230663 0.91491369]
print('After cross validation, the mean accuracy for Random Forest is: ', np.mean(scores))
# After cross validation, the mean accuracy for Random Forest is:  0.9145075945501534

"""
The highest average accuracy comes from the Random Forest Classifier. 
Now that I have figured out which features to use and which classification 
algorithm to use, I can predict whether someone donated or not on the validation data
"""

# Predictions

data = df1[sig_features]
predictions = rf.predict(data)

# Now I can figure out the members whom my model predicted to donate

# Predicted donating members

members_df = df1.copy()
members_df['predictions'] = predictions
donators = members_df[(members_df.predictions == 1)]
donators = donators['CONTROLN']
predicted_donators = []
print('The members who are predicted to donate are: ')
for i in donators:
    predicted_donators.append(i)
predicted_donators

count = 0
for rows in (df['TARGET_B']==1):
    if rows == True:
        count+=1
print('From the learning dataset, the total number of donators was', count)
print('My model predicted that there would be a total number of', len(predicted_donators), 'donators')
# From the learning dataset, the total number of donators was 4843 
# My model predicted that there would be a total number of 5376 donators

# Predicting total donation amount from predicted donators

members_df = df1.copy()
members_df['predictions'] = predictions
donators = members_df[(members_df.predictions == 1)]
donators = donators[sig_features]
temp = df[df.TARGET_D != 0]
x_train = temp[sig_features]
y_train = temp['TARGET_D']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
amounts = model.predict(donators)
print('Total USD expected from predicted donators:', round(np.sum(amounts), 2))
# Total USD expected from predicted donators: 85341.31

# Conclusion / Write Up

"""
Considering there were over 95,000 members, predicting that there will be about 
1000 more donators than there were in the training set is excellent. If I had to
choose between over-predicting the number of people who were to donate versus 
under-predicting the total number of people who were to donate I would absolutely 
over-predicting. The reason over-predicting in this case is of benefit is because
if I were to under-predict the number of requests to send, it would mean I would be 
losing out on precious donations. We have 40,000 USD to run the entire campaign and 
each request for fundraising costs 1 USD. Sending out 5,376 requests will cost 5,376 
USD which may mean some requests wouldn't result in a donation. HOWEVER, these 
requests would be going to people whom, based on certain features (ethnicity, 
age of children, etc.), are more likely to donate - so the request would not be 
completely blind and may pay off in the future. The remaining money, 34,624 USD, can
be used for advertising or other means. Requests from these 5,376 donators is 
expected to result in 85,341.31 USD, which totals 119,965.31 (85341+34624) USD 
considered for disposable funding.

In order to figure out which members would be likely to donate, I used the Paralyzed 
Verterans of America (PVA) dataset which contains 96367 records and 479 variables. 
I uploaded the data into a pandas dataframe and checked the first few records to make 
sure it uploaded correctly. From there, I noticed missing values and performed a health
check to see how many missing values I was dealing with - and it turns out a lot. I had 
to do a feature selection to narrow down which ones were influencing the donations the 
most and I did this through regression. I ran a regression and found which features 
were significant (90%). After feature selection, it turned out the data was clean in 
terms of missing values. With these features, I ran several different models to 
determine which model predicted donators most accurately. In order to avoid bias and 
overfitting, I ran a cross validation and determined the mean accuracy for each algorithm. 
It's important to note that I balanced the class weights when fitting each model because 
the model could theoretically predict all 0's (everyone was a non-donator) and acheive over 
90 percent accuracy based on the fact that there were simply way more people who didn't 
donate versus those that did. After running the cross-validation on all of the models, I 
found that Random Forest was the best at predicting the class with an average of 91.45% 
accuracy. I was then able to run a Random Forest on the validation data-set to predict 
which members were most likely to donate. After determing which members were likely to donate, 
I ran a linear regression to predict how much they were going to donate.
"""



