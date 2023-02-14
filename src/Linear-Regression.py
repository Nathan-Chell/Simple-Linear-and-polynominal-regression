import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

def get_data():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')

    data = pd.read_csv(os.path.join(data_dir, 'Inc_Exp_Data.csv'))
    
    return data

def plot_data(data):
    
    sns.regplot(
    x = "Mthly_HH_Income",
    y = "Mthly_HH_Expense", 
    data = data)
    
    plt.title("Monthly Income vs Monthly Expense")
    # show the plot
    plt.show()


def main():
    
    data = get_data()
    plot_data(data)

if __name__ == '__main__':
    main()