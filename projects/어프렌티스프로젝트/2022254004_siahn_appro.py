import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    my_data = pd.read_csv('./2022254004_안성인_iaq_data.csv')
    my_data.head()

    my_data.plot(kind='hist', y='SCORE')


if __name__ == '__main__':
    main()
