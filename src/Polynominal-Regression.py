#Simple implementation of polynominal regression
#On randomly generated data

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def get_data():
    
    #Generate random data
    #Equation used y = 0.8^2 + 0.9x + 2 + noise
    
    #set random seed
    np.random.seed()
    
    x = 6 * np.random.rand(200, 1) - 3
    y = 0.8 * x**2 + 0.9*x + 2 + np.random.randn(200, 1)
    
    return x, y

def plot_data(x, y, x_train, x_pred, regression = False):
    
    plt.plot(x, y, 'b.')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if regression:
        plot_data_with_regression(x, y, x_train, x_pred)
    else:
        plt.show()

def plot_data_with_regression(x, y, x_train, x_pred):
    
    plt.plot(x, y, 'b.')
    plt.plot(x_train, x_pred, color = 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def plot_data_with_polynominal_regression(x_train, y_train, x_test, y_test, poly_features, lr):
    
    x_new = np.linspace(-3, 3, 200).reshape(200, 1)
    x_new_poly = poly_features.transform(x_new)
    y_new = lr.predict(x_new_poly)
    
    plt.plot(x_new, y_new, 'r-', linewidth = 2, label = 'Predictions')
    plt.plot(x_train, y_train, 'b.', label = 'Training data')
    plt.plot(x_test, y_test, 'g.', label = 'Testing data')
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.legend()
    plt.show()
    
    
def LinearRegressionModel(x, y, x_train, y_train, x_test, y_test):
    
    #Initialise linear regression
    #and train the model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    #Calculate predictions
    y_pred = lr.predict(x_test)
    
    #Calculate r2 score
    #The value will be around 25%, which is awful
    print("R2 score for linear regression: ", r2_score(y_test, y_pred))
    
    #Plot random data with regression model
    x_pred = lr.predict(x_train)
    plot_data(x, y, x_train, x_pred, True)
    
def PolynominalRegressionModel(x_train, y_train, x_test, y_test):
    
    #Apply polynomial features
    poly_features = PolynomialFeatures(degree = 2, include_bias = True)
    x_train_trans = poly_features.fit_transform(x_train)
    x_test_trans = poly_features.fit_transform(x_test)
    
    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)
    y_pred = lr.predict(x_test_trans)
    
    #Calculate r2 score
    #The value will be around 90%, which is better than linear regression
    print("R2 score for polynominal regression: ", r2_score(y_test, y_pred))
    
    print("Coefficients: ", lr.coef_)
    print("Intercept: ", lr.intercept_)
    
    plot_data_with_polynominal_regression(x_train, y_train, x_test, y_test, poly_features, lr)

def PolynomialRegressionModelWithDegree(x_train, y_train, x_test, y_test, degree = 2): 
    
    poly_features = PolynomialFeatures(degree = degree, include_bias = False)
    
    x_new = np.linspace(-3, 3, 100).reshape(100, 1)
    x_new_poly = poly_features.fit_transform(x_new)
    
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    
    polynominal_regression = Pipeline([
        ('poly_features', poly_features),
        ('std_scaler', std_scaler),
        ('lin_reg', lin_reg)
    ])
    
    polynominal_regression.fit(x_train, y_train)
    y_new = polynominal_regression.predict(x_new)
    
    #plot data
    
    plt.plot(x_new, y_new, 'r', linewidth = 2, label = 'Degree = {}'.format(degree))
    plt.plot(x_train, y_train, 'b.', label = 'Training data', linewidth=3)
    plt.plot(x_test, y_test, 'g.', label = 'Testing data', linewidth=3)
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.axis([-3, 3, 0, 10])
    plt.show()

def main():
    
    x, y = get_data()
    #Plot random data
    #plot_data(x, y)
    
    #Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
    
    #LinearRegressionModel(x, y, x_train, y_train, x_test, y_test)
    #PolynominalRegressionModel(x_train, y_train, x_test, y_test)
    PolynomialRegressionModelWithDegree(x_train, y_train, x_test, y_test, 10)
    
if __name__ == '__main__':
    main()