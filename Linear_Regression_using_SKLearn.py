# File: Linear_Regression_using_SKLearn.py

import time
import numpy as np
import matplotlib.pyplot as plt

# scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


def sklearn_lr(Xs, X_trn, X_tst, y_trn, y_tst):
    """  Linear Regression using scikit learn """
    """
        Linear model: y = α + βX
    
        The parameters α and β of the model are selected through the Ordinary 
        least squares (OLS) method. It works by minimizing the sum of squares 
        of residuals (actual value - predicted value). """

    start = time.time()

    # Instantiate the algorithm
    reg = LinearRegression()
    # Fit the model on the training set
    reg.fit(X_trn, y_trn)

    # Pull α and β parameters
    """
        Coefficients are estimated using the least squares criterion. 
        In other words, we find the line (mathematically) which minimizes the 
        sum of squared residuals (or "sum of squared errors"):
    """
    alpha = reg.intercept_
    beta = reg.coef_
    # Print lr coefficients
    print("--- Intercept - α = ", alpha)
    print("--- Slope's - β's = ", beta)

    # Using the Model for Prediction
    predicted_y = reg.predict(X_test)

    # RMSE
    RMSE = np.sqrt(mean_squared_error(y_test, predicted_y))
    print('\tRMSE: Root Mean Squared Error: {%.2f}' % RMSE)
    # MAE
    MAE = mean_absolute_error(y_test, predicted_y)
    print("\tMAE: Mean Absolute Error: {%.2f}" % MAE)
    # R-squared, it's most useful as a tool for comparing different models
    R2 = r2_score(y_test, predicted_y)
    print('\tR^2: R-Square (regression score function): {%.2f%%}' % (R2 * 100))
    """ 
        Since you are doing a classification task, you should be using 
        the metric R-squared (coefficient of determination) instead of 
        accuracy score (accuracy score is used for classification purposes).
    """

    end = time.time()
    print('Execution Time: {%f}' % ((end - start) / 1000)
          + ' seconds.')


if __name__ == '__main__':
    try:
        # Create regression data
        X, y = datasets.make_regression(n_samples=1000,
                                        n_features=3,
                                        noise=20,
                                        random_state=5)
        # Split the data to training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=5)

        # Test using sklearn library
        print('\n ---------- LR using sklearn library ----------')
        sklearn_lr(X, X_train, X_test, y_train, y_test)

    except:
        pass


