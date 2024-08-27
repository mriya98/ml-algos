import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Linear Regression
def linear_multiplication(w, X):
    '''
    Input: w is a weight parameter (including the bias), and X is a data matrix (n x (d+1)) (including the feature)
    Output: a vector containing the predictions of linear models
    '''
    # To do: Insert you code to get a vectorization of the predicted output computation for a linear model
    return np.dot(X, w)

def make_prediction(x_input, wb):
    # add an extra feature (column in the input) that are just all ones
    x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)
    y = linear_multiplication(wb, x_in)
    
    return y

def closed_form_solution(X, y):
    '''
    Solve linear regression using the normal equation.
    
    Given `X` - n x (d+1) matrix of inputs
          `y` - target outputs
    Returns the optimal weights as a (d+1)-dimensional vector
    '''
    A = np.dot(X.T, X)
    c = np.dot(X.T, y)
    return np.dot(np.linalg.inv(A), c)


'''
# Load the dataset AMES
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
print(housing['DESCR'])

# Load california houseing dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
'''

# Load boston
data_url = "http://lib.stat.cmu.edu/datasets/boston"
'''
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.

    Variables in order:
    CRIM     per capita crime rate by town
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS    proportion of non-retail business acres per town
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX      nitric oxides concentration (parts per 10 million)
    RM       average number of rooms per dwelling
    AGE      proportion of owner-occupied units built prior to 1940
    DIS      weighted distances to five Boston employment centres
    RAD      index of accessibility to radial highways
    TAX      full-value property-tax rate per $10,000
    PTRATIO  pupil-teacher ratio by town
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT    % lower status of the population
    MEDV     Median value of owner-occupied homes in $1000's
'''
# Load dataset from url, we skip first 22 rows as they contain metadata
# Blank space character is used as seperator
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# Single sample is split in two rows in the dataset. Stack them horizonatally.
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# In the original df, the target values are stored in the third column of every
# odd numbered row
y_target = raw_df.values[1::2, 2]

# Since we are building simple model, lets use only 2 columns - INDUS and RM
# we will only work with two of the features: INDUS and RM
x_input = data[:, [2,5]]
# Normalise the features
x_input = preprocessing.normalize(x_input)
# add an extra feature (column in the input) that are just all ones
x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)


# Individual plots for the two features:
plt.title('Industrialness vs Med House Price')
plt.scatter(x_input[:, 0], y_target)
plt.xlabel('Industrialness')
plt.ylabel('Med House Price')
plt.show()

plt.title('Avg Num Rooms vs Med House Price')
plt.scatter(x_input[:, 1], y_target)
plt.xlabel('Avg Num Rooms')
plt.ylabel('Med House Price')
plt.show()

# Calculate WEIGHTS
'''
    Predictions for linear regression can be made using the exact solution.
    However, even the vectorised form can be computationally expensive for
    large datasets. Since, we are using only 2 columns, we can use it here.
'''
w_exact = closed_form_solution(x_in, y_target)

print(w_exact)


# CODE FOR INFERENCE
#x_test = []
#y_pred = make_prediction(x_test, w_exact)







