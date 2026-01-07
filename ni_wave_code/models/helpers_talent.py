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

def load_data_semisup(group, test_size, sub=True):
    """

    :param group: age class
    :param test_size: 0.2 for 20 %
    :return: dat, X_0, y_0, X = train, X_test, y = train, y_test
    """

    dat0 = pd.read_csv('../../data/dat_vs1', sep=' ', index_col=False)
    dat = dat0[dat0['AK'] == group]
    dat_neg = dat[dat['LZ'] == 0]
    dat_pos = dat[dat['LZ'] == 1]
    dat_neg = dat_neg.drop('AK', axis=1).reset_index(drop=True)
    dat_pos = dat_pos.drop('AK', axis=1).reset_index(drop=True)
    if sub:
        c = 16
    else:
        c = 14
    X_neg_0 = dat_neg.iloc[:, 2:c]
    y_neg_0 = dat_neg.iloc[:, 16]
    X_pos_0 = dat_pos.iloc[:, 2:c]
    y_pos_0 = dat_pos.iloc[:, 16]
    X_neg, X_neg_test, y_neg, y_neg_test = train_test_split(X_neg_0, y_neg_0, test_size=test_size, random_state=42)
    'X_test = X_pos_0 + X_neg_test'
    X_test = pd.concat([X_pos_0, X_neg_test])
    'y_test = y_pos_0 + y_neg_test'
    y_test = pd.concat([y_pos_0, y_neg_test])

    return X_neg, y_neg, X_test, y_test

#########################################################################################################
#########################################################################################################
#___ PLOTS
def dens_plt(var, data):
    for c in {0, 1}:
        subset = data[data['LZ'] == c]
        sns.kdeplot(subset[var],
                    label=c, shade=False, alpha=0.8)
        plt.title(var)

#___ Display how the latent space clusters different classes


def plot_label_clusters(repr, labels, latent_dim, z1=None, z2=None, name=None):
    if z1 is None and z2 is None:
        if latent_dim == 1:
            plt.scatter(repr[:, 0], repr[:, 0])
        # display a 2D plot of the classes in the latent space
        else:
            fig, axs = plt.subplots(latent_dim, latent_dim, figsize=(25,20))
            for i in range(latent_dim):
                for j in range(latent_dim):
                    axs[i, j].scatter(repr[:, i], repr[:, j], c=labels, marker=".")
            #fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.9,
            #                   hspace=1, wspace=1.5)
    else:
        if z1 > latent_dim or z2 > latent_dim:
            print("Dimension too high!")
        else:
            plt.scatter(repr[:, z1], repr[:, z2], c=labels)
            plt.xlabel("z[%d]" % z1)
            plt.ylabel("z[%d]" % z2)
    if name is not None:
        plt.suptitle(name)
    plt.show()


#___ CLUSTERING RESULTS FCT
def cl_result(data, labels, n_clusters):
    n, c = data.shape
    temp = np.zeros((n, c+1))
    temp[:, :-1] = data
    temp[:, -1] = labels
    c_names = list(data.columns)
    c_names.append('cluster')
    cl_df = pd.DataFrame(data=temp, columns=c_names)
    n_cl = np.zeros(n_clusters)
    for i in range(0, n_clusters):
        n_cl[i] = cl_df[cl_df['cluster']==i].shape[0]

    clusters_df = cl_df[cl_df['cluster']==0].mean()
    for i in range(1, n_clusters):
        clusters_df = pd.concat([clusters_df, cl_df[cl_df['cluster']==i].mean()], axis=1, ignore_index=True)

    clusters_df = pd.concat([clusters_df, pd.DataFrame(n_cl.reshape(1,-1))], axis=0)

    return clusters_df, cl_df

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
nmi = normalized_mutual_info_score
ari = adjusted_rand_score
def acc_cl(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size