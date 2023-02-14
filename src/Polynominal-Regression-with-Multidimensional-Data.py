#Polynominal regression on multidimensional data
#3 dimentions will be used in this program.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def getData():
    
    #Generate random data
    #z = x^2 + y^2 + 0.2x + 0.2y + 0.1xy + 2 + noise
    
    x = 7 * np.random.rand(100, 1) - 2.8
    y = 7 * np.random.rand(100, 1) - 2.8
    z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y + 2 + np.random.randn(100, 1)
    
    return x, y, z

def main():
    
    x, y, z = getData()
    
    data_df = px.data.iris()
    fig = px.scatter_3d(data_df, x = x.ravel(), y = y.ravel(), z = z.ravel())
    fig.show()
    
if __name__ == '__main__':
    main()