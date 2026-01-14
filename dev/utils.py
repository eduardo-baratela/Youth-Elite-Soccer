# Setup
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pylab as plt

#########################################################################################################
#########################################################################################################
# ___ DATA
def load_data(group, test_size, sub=True):
    """

    :param group: age class
    :param test_size: 0.2 for 20 %
    :return: dat, X_0, y_0, X = train, X_test, y = train, y_test
    """

    dat0 = pd.read_csv('../../data/dat_vs1', sep=' ', index_col=False)
    dat = dat0[dat0['AK'] == group]
    dat = dat.drop('AK', axis=1).reset_index(drop=True)
    if sub:
        c = 16
    else:
        c = 14
    X_0 = dat.iloc[:, 2:c]
    y_0 = dat.iloc[:, 16]
    X, X_test, y, y_test = train_test_split(X_0, y_0, test_size=test_size, random_state=42, stratify=y_0)
    return dat, X_0, y_0, X, X_test, y, y_test
