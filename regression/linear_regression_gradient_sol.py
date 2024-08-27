import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import preprocessing
import warnings


# Data processing

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

def cost(w, X, y):
    '''
    Evaluate the cost function in a vectorized manner for 
    inputs `X` and outputs `y`, at weights `w`.
    '''
    # TODO: Insert your code to compute the cost
    residual = y - linear_multiplication(w, X)  # get the residual
    err = np.dot(residual, residual) / (2 * len(y))
    
    return err

def calculate_gradient(X, w, y):
    #Gradient for Mean-Squared Error cost function: (1/n)*(X_transpose*(X.w - y))
    y_pred = np.dot(X, w)
    error = y_pred - y
    return np.dot(X.T, error) / len(y)

def gradient_descent_solver(X, y, print_every=100,
                               niter=2000, eta=1):
    _, D = np.shape(X)
    # Initialise weights
    w = np.zeros([D])

    idx_res = []
    err_res = []
    for i in range(niter):
        # Calculate graient using current weights
        dw = calculate_gradient(X, w, y)
        # gradient descent
        w = w - eta * dw
        # Report progress
        if i % print_every == print_every - 1:
            total_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (i, total_cost))
            idx_res.append(i)
            err_res.append(total_cost)
    return w, idx_res, err_res


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
y_target = raw_df.values[1::2, 2]
# add a feature 1 to the dataset, then we do not need to consider the bias and weight separately
x_input = np.concatenate([np.ones([np.shape(data)[0], 1]), data], axis=1)
x_input = preprocessing.normalize(x_input)

# FIT the Model using GRADIENT DESCENT
'''
    Predictions for linear regression can be made using the exact solution.
    However, even the vectorised form can be computationally expensive for
    large datasets and it is better to use gradient descent to calculate
    the weight matrix
'''
w_gradDesc, idx, err = gradient_descent_solver(x_input, y_target)

# Plot iter vs error plot
plt.plot(idx, err, color="blue", linewidth=2.5, linestyle="-", label="gradient descent")
plt.legend(loc='upper right', prop={'size': 12})
plt.title('Iterations vs Error in GD')
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.grid()
plt.show()  

# INFERENCE
#y_pred = make_prediction(x_test, w_gradDesc)







