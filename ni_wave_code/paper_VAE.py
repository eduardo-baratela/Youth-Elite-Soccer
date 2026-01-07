# ___ SET UP
################################################################################################
# load namespaces
import pandas as pd

from models.helpers_talent import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, mean_squared_error
from models.VAE import *
from sklearn.manifold import TSNE
plt.rcParams['figure.figsize'] = (15, 12)
save_dir = 'saved_models'

# <editor-fold desc="DATA">
# ___ DATA
group = 1           # age class 'AK'
test_size = 0.2     # train-test split
sub = False         # subjective meassures

dat0 = pd.read_csv('data/dat_vs1', sep=' ', index_col=False)
dat = dat0[dat0['AK'] == group]
dat = dat.drop('AK', axis=1).reset_index(drop=True)
if sub:
    c = 16
else:
    c = 14
X_0 = dat.iloc[:, 2:c]
y_0 = dat.iloc[:, 16]
X, X_test, y, y_test = train_test_split(X_0, y_0, test_size=test_size, random_state=42, stratify=y_0)
X_names = list(X_0.columns)

# ___ SCALING
# MinMax [0,1]
scaler = StandardScaler().fit(X)
df = scaler.transform(X)
# scaler_test = StandardScaler().fit(X_test)
df_test = scaler.transform(X_test)
y = np.asarray(y)
# </editor-fold>


################################################################################################
################################################################################################
# Function
def learn(inter1_dim, inter2_dim, latent_dim, beta, learning_rate, list, callback, pretrain_epochs, batch_size):
    mod = '/model_{}.{}.{}_{}_{}'.format(inter1_dim, inter2_dim, latent_dim, beta, learning_rate)
    encoder, decoder, vae = BetaVAutoEncoder(original_dim=df.shape[1],
                                             latent_dim=latent_dim,
                                             inter1_dim=inter1_dim,
                                             inter2_dim=inter2_dim,
                                             act='relu',
                                             beta=beta)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(df, df, epochs=pretrain_epochs, batch_size=batch_size, callbacks=[callback])
    losses = vae.evaluate(df,df)
    if beta == 0:
        kl = losses[1]
    else:
        kl = losses[1]/beta

    res = {'name': mod, 'inter1_dim': inter1_dim, 'inter2_dim': inter2_dim,
           'latent_dim': latent_dim, 'beta': beta, 'learning_rate': learning_rate,
           're_loss': losses[0], 'kl_loss': kl, 'loss': losses[0]+kl}
    list.append(res)
    return vae, encoder, decoder, mod
#####

##### TEST
'''
pretrain_epochs = 1000
batch_size = 256
inter1_dim = 10
inter2_dim = 8
callback = EarlyStopping(monitor="loss", min_delta=0.0001, patience=20)
latent_dim = 7
learning_rate = 0.01
result_test = []

beta = 1
vae, encoder, decoder, mod = learn(inter1_dim, inter2_dim, latent_dim, beta, learning_rate,
                                                  result_test, callback, pretrain_epochs, batch_size)
z_mean_test = encoder.predict(df)[0]
z_sd_test = np.exp(0.5 * encoder.predict(df)[1])
print(result_test)
print(z_mean_test)
print(z_sd_test)
'''

####### SEARCH

pretrain_epochs = 5000
batch_size = 256
inter1_dim = 10
inter2_dim = 8
callback = EarlyStopping(monitor="loss", min_delta=0.0001, patience=20)
learning_rate = 0.01

result_dict = []
for latent_dim in range(5, 11):
    for beta in np.arange(0.1, 1., 0.05):
            vae, encoder, decoder, mod = learn(inter1_dim, inter2_dim, latent_dim, beta,
                                               learning_rate, result_dict, callback, pretrain_epochs, batch_size)
            vae.save(save_dir+mod+'/vae')
            encoder.save(save_dir+mod+'/en')
            decoder.save(save_dir+mod+'/de')


# save results as csv
import csv
loss_dic_names = ['name', 'inter1_dim', 'inter2_dim', 'latent_dim', 'beta', 'learning_rate',
                  're_loss', 'kl_loss', 'loss']
with open(save_dir+'/model_results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=loss_dic_names)
    writer.writeheader()
    writer.writerows(result_dict)


################################################################################################
################################################################################################
####### LOAD
model_results = pd.read_csv(save_dir + '/model_results.csv', index_col=False)
bestidx = 73
latent_dim = model_results['latent_dim'][bestidx]

vae = tf.keras.models.load_model(save_dir+model_results["name"][bestidx]+'/vae')
encoder = tf.keras.models.load_model(save_dir+model_results["name"][bestidx]+'/en')
decoder = tf.keras.models.load_model(save_dir+model_results["name"][bestidx]+'/de')

# Prediction
latent_pred = encoder.predict(df)[0]
z_sd = np.exp(0.5 * encoder.predict(df)[1])
latent_succ = encoder.predict(df[y == 1, :])[0]
latent_fail = encoder.predict(df[y == 0, :])[0]
output_st = decoder.predict(latent_pred)
output = pd.DataFrame(scaler.inverse_transform(output_st))

np.sqrt(mean_squared_error(output.iloc[:, 0], X.iloc[:, 0]))
pd.DataFrame(latent_succ).describe()
pd.DataFrame(latent_fail).describe()


##### GENERATE
# Get latent (multivariate) normal distribution
# Sigma = np.cov(np.matrix.transpose(latent_pred))
# Mean = np.mean(latent_pred, axis=0)

gen_out = sampleVAE(de_model=decoder,
                               en_prediction=latent_pred,
                               num_samples=5,
                               label=1,
                               scale=scaler,
                               both=False,
                               conditional=False)

################################################################################################
##### PLOTs
## TSNE
df_embedded = TSNE(n_components=2, init='random').fit_transform(df)
latent_embedded = TSNE(n_components=2, init='random').fit_transform(latent_pred)

fig, ax = plt.subplots()
scatter_x = latent_embedded[:, 0]
scatter_y = latent_embedded[:, 1]
label = y
for l in np.unique(label):
    i = np.where(label == l)
    ax.scatter(scatter_x[i], scatter_y[i], label=l, marker='o')
ax.legend()
plt.show()

fig, ax = plt.subplots()
scatter_x = df_embedded[:, 0]
scatter_y = df_embedded[:, 1]
label = y
for l in np.unique(label):
    i = np.where(label == l)
    ax.scatter(scatter_x[i], scatter_y[i], label=l, marker='o')
ax.legend()
plt.show()


## OTHER
fig, axs = plt.subplots(2, int(np.ceil(latent_succ.shape[1]/2)))
for i in range(latent_succ.shape[1]):
    if i < int(np.ceil(latent_succ.shape[1]/2)):
        row = 0
    else:
        row = 1
    axs[row, i%int(np.ceil(latent_succ.shape[1]/2))].hist([latent_succ[:, i], latent_fail[:, i]], bins=50, density=True)
plt.show()

from models.helpers_talent import plot_label_clusters
plot_label_clusters(latent_pred, y, 10)

# ___ PLOT
fig, axs = plt.subplots(2, int(np.ceil(latent_succ.shape[1]/2)))
for i in range(latent_succ.shape[1]):
    if i < int(np.ceil(latent_succ.shape[1]/2)):
        row = 0
    else:
        row = 1
    axs[row, i%int(np.ceil(latent_succ.shape[1]/2))].boxplot([latent_succ[:,0],latent_fail[:,0]],0,'',
                                                             showmeans=True, meanline=True)
    axs[row, i%int(np.ceil(latent_succ.shape[1]/2))].set_title('{}'.format(i))
plt.show()

from scipy.stats import ttest_ind
for i in range(latent_dim):
    stat, p = ttest_ind(latent_succ[:, i],latent_fail[:, i])
    print('For dim', i+1, 'the test statistic is %.3f and p=%.3f' % (stat, p))

################################################################################################
##### Latent meaning
np.random.seed(43)
sample_dim = 7
sample_ld = sampleVAE1D(decoder, latent_pred, scaler, dim=sample_dim, num_samples_other=1)
dim_name = 'sample_dim{}'.format(sample_dim)
title = 'Simulation of dimension {}. Plotted for all variables'.format(sample_dim)

for i in range(X.shape[-1]):
    plt.plot(sample_ld[dim_name], sample_ld['pred_or{}'.format(i)], label=X.columns[i])
plt.legend()
plt.title(title)
plt.show()

# TODO: Tabelle mit Steigungen (und Signifikanzen)



################################################################################################
##### On latent space
################################################################################################
################################################################################################
# ___ LATENT PREDICTION
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(latent_pred, y)
clf.score(latent_pred,y)
pred_clf = clf.predict(latent_pred)
prob_clf = clf.predict_proba(latent_pred)
confusion_matrix(y,pred_clf)

######################################################################
######################################################################
# --- CLUSTERING
# TODO check for n_cluster in range(2,11) and describe. calc odds-ratios (latent and or). eval n_cluster
# clustering namespaces
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# ___ Number of clusters
'''
from matplotlib import pyplot as plt
distorsions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    distorsions.append(kmeans.inertia_)  # interia_ = total WSS

fig = plt.figure(figsize=(15, 5))
plt.plot(range(1, 10), distorsions)
plt.grid(True)
plt.title('Elbow curve')        # vllt 2?
plt.show()
'''
n_clusters = 4
#_________________________
"""Baseline K-MEANS"""

'''original (scaled) data'''
kmeans_raw_or = KMeans(n_clusters=n_clusters, n_init=20).fit(df)  # 20 mal (gegen lokale minima)
kmeans_y_or = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(df)
accuracy_score(y, kmeans_y_or)

clusters_km_or, df_km_or = cl_result(pd.DataFrame(df), kmeans_y_or, n_clusters)
df_km_or["true"] = y
df_km_or[df_km_or["cluster"] == 0]["true"].mean()
df_km_or[df_km_or["cluster"] == 1]["true"].mean()
df_km_or[df_km_or["cluster"] == 2]["true"].mean()
df_km_or[df_km_or["cluster"] == 3]["true"].mean()

df_original_scale_km_or = X
df_original_scale_km_or["cluster"] = kmeans_y_or
df_original_scale_km_or["true"] = y

km_describe_cl_or = pd.concat((df_original_scale_km_or[df_original_scale_km_or["cluster"] == 0].describe(),
                               df_original_scale_km_or[df_original_scale_km_or["cluster"] == 1].describe(),
                              df_original_scale_km_or[df_original_scale_km_or["cluster"] == 2].describe(),
df_original_scale_km_or[df_original_scale_km_or["cluster"] == 3].describe()))

"""latent"""

kmeans_raw = KMeans(n_clusters=n_clusters, n_init=20).fit(latent_pred)  # 20 mal (gegen lokale minima)
kmeans_y = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(latent_pred)
accuracy_score(y, kmeans_y)

clusters_km, df_km = cl_result(pd.DataFrame(latent_pred), kmeans_y, n_clusters)
df_km["true"] = y
df_km[df_km["cluster"] == 0]["true"].mean()
df_km[df_km["cluster"] == 1]["true"].mean()
df_km[df_km["cluster"] == 2]["true"].mean()
df_km[df_km["cluster"] == 3]["true"].mean()

df_original_scale_km = X
df_original_scale_km["cluster"] = kmeans_y
df_original_scale_km["true"] = y

km_describe_cl = pd.concat((df_original_scale_km[df_original_scale_km["cluster"] == 0].describe(),
                            df_original_scale_km[df_original_scale_km["cluster"] == 1].describe(),
                            df_original_scale_km[df_original_scale_km["cluster"] == 2].describe(),
                            df_original_scale_km[df_original_scale_km["cluster"] == 3].describe()))

# clusters_km.to_csv(path_or_buf='../../data/results/clusters_km.csv', index = True)

######################################################################
######################################################################

"""Gaussian Mixture"""
gmix = GaussianMixture(n_components=n_clusters, random_state=0).fit(latent_pred)
gm_clusters = gmix.predict(latent_pred)
accuracy_score(y, gm_clusters)

clusters_gm, df_gm = cl_result(pd.DataFrame(latent_pred), gm_clusters, n_clusters)

"""Isolation Forest"""
from sklearn.ensemble import IsolationForest
IF = IsolationForest(random_state=0, n_estimators=100, contamination=0.1).fit(latent_pred)
IF.decision_function(latent_pred)
IF.score_samples(latent_pred)

# Train
train_pred_IF = IF.predict(latent_pred)
fail_train_pred_IF = IF.predict(latent_fail)
succ_train_pred_IF = IF.predict(latent_succ)
np.sum(latent_succ[latent_succ == -1])
np.sum(latent_fail[latent_fail == -1])

fail_train_scores_IF = IF.score_samples(latent_fail)
succ_train_scores_IF = IF.score_samples(latent_succ)
np.mean(fail_train_scores_IF)
np.mean(succ_train_scores_IF)




################################################################################################
##### VAE and Cluster Combi
################################################################################################
################################################################################################
# ___ CLUSTERING ON HIDDEN LAYER
from models.clusters import ClusteringLayer

n_clusters = 3

# Create the CLUSTERING LAYER
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output[0])
model = keras.models.Model(inputs=encoder.input, outputs=clustering_layer)              # create the model
model.compile(optimizer=keras.optimizers.SGD(0.01, 0.9), loss='kld')                    # compile

# initialize cluster centers with Kmeans
kmeans_init = KMeans(n_clusters=n_clusters, n_init=20)
y_pred_init = kmeans_init.fit_predict(encoder.predict(df)[0])
y_pred_last = np.copy(y_pred_init)
confusion_init = confusion_matrix(y, y_pred_init)
model.get_layer(name='clustering').set_weights([kmeans_init.cluster_centers_])

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = 15000
update_interval = 140
index_array = np.arange(df.shape[0])
batch_size = 256

tol = 0.000001 # tolerance threshold to stop training

# training
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(df, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            #acc = np.round(acc_cl(y, y_pred), 5)
            nmi_score = np.round(nmi(y, y_pred), 5)
            ari_score = np.round(ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: nmi_score = %.5f, ari_score = %.5f' % (ite, nmi_score, ari_score), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, df.shape[0])]
    loss = model.train_on_batch(x=df[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= df.shape[0] else 0

model.save_weights(save_dir + '/comb_vae_model_final.h5')

model.load_weights(save_dir + '/comb_vae_model_final.h5')

######################################################################
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
# ___ EVAL
df_eval = df
y_eval = np.asarray(y)
# Eval.
q = model.predict(df_eval, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p
y_pred = q.argmax(1)


# plot
import seaborn as sns
sns.set(font_scale=3)
confusion = confusion_matrix(y_eval, y_pred)
repr = encoder.predict(df)[0]
plt.scatter(repr[:,0],repr[:,1])
plt.show()


plt.figure(figsize=(16, 14))
sns.heatmap(confusion, annot=True, fmt="d", annot_kws={"size": 20})
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()




