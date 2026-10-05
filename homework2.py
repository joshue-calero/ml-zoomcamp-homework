import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)

#print(df.head())


"""
PREPARING DATASET
Use only the following columns:
'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
'fuel_efficiency_mpg'
"""
columns = [
'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
'fuel_efficiency_mpg'
]

df = df[columns]

#print(df.head())

"""
EDA
Look at the fuel_efficiency_mpg variable. Does it have a long tail?
"""

sns.histplot(df.fuel_efficiency_mpg, bins = 50)
#plt.show()

# it doesnt have a long tail

"""
QUESTION 1
There's one column with missing values. What is it?
"""

nulls = df.isnull().sum()

#print(nulls)

# horsepower 877 nulls, other columns 0 nulls

"""
QUESTION 2
What's the median (50% percentile) for variable 'horsepower'?
"""

median_horsepower = df['horsepower'].median()

# print(median_horsepower)

# answer 254.0

"""
Prepare and split the dataset
"""

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

"""
QUESTION 3
We need to deal with missing values for the column from Q1.
We have two options: fill it with 0 or with the mean of this variable.
Try both options. For each, train a linear regression model without regularization using the code from the lessons.
For computing the mean, use the training only!
Use the validation dataset to evaluate the models and compare the RMSE of each option.
Round the RMSE scores to 3 decimal digits using round(score, 3). This keeps the imputation difference visible in this release.
Which option gives better RMSE?
Options:

With 0
With mean
Both are equally good
"""


y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()

    return np.sqrt(mse)

"""with 0"""

X_train = df_train.fillna(0).values
X_val = df_val.fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_val.dot(w)

score = rmse(y_val, y_pred)

#print(round(score, 3))

# answer 2.205

"""with mean"""

mean = df_train.horsepower.mean()

X_train = df_train.fillna(mean).values
X_val = df_val.fillna(mean).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_val.dot(w)

score = rmse(y_val, y_pred)

#print(round(score, 3))

# answer 2.202

#correct answer is with mean because RMSE is lower

"""
QUESTION 4
Now let's train a regularized linear regression.
For this question, fill the NAs with 0.
Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
Use RMSE to evaluate the model on the validation dataset.
Round the RMSE scores to 4 decimal digits.
Which r gives the best RMSE?

Options:

0
0.01
0.1
1
5
10
100
"""

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


X_train = df_train.fillna(0).values
X_val = df_val.fillna(0).values

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)

    #print(r, round(score, 4))

# 0 2.2053
# 0.01 2.2058
# 0.1 2.2241
# 1 2.3492
# 5 2.4094
# 10 2.4195
# 100 2.4292

# correct answer is 0 because it has the lowest RMSE

"""
QUESTION 5
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
For each seed, do the train/validation/test split with 60%/20%/20% distribution.
Fill the missing values with 0 and train a model without regularization.
For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
What's the standard deviation of all the scores?
Use np.std and round the result to 3 decimal digits.

Options:

0.006
0.016
0.029
0.036
"""

scores = []

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    y_train = df_train.fuel_efficiency_mpg.values
    y_val = df_val.fuel_efficiency_mpg.values

    df_train = df_train.drop('fuel_efficiency_mpg', axis=1)
    df_val = df_val.drop('fuel_efficiency_mpg', axis=1)

    X_train = df_train.fillna(0).values
    X_val = df_val.fillna(0).values

    w0, w = train_linear_regression(X_train, y_train)

    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)

    scores.append(score)

std = np.std(scores)

# print(round(std, 3))

# answer 0.029

"""
QUESTION 6
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Fill the missing values with 0 and train a model with r=0.001.
What's the RMSE on the test dataset?

Options:

0.236
2.236
22.10
221.0
"""

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

df_full_train = pd.concat([df_train, df_val])

y_full_train = df_full_train.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_full_train['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

X_full_train = df_full_train.fillna(0).values
X_test = df_test.fillna(0).values

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

y_pred = w0 + X_test.dot(w)

score = rmse(y_test, y_pred)

print(round(score, 3))

# answer 2.236