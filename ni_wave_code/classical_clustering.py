
# ___ Set up

# load namespaces
from models.helpers_talent import *
from sklearn.preprocessing import MinMaxScaler

# DATA
group = 1           # age class 'AK'
test_size = 0.2     # train-test split

# normal with subjective meassures = TRUE/FALSE
# dat, X_0, y_0, _, _, _, _ = load_data(group=group, test_size=test_size, sub=True) # subjectives = True
dat, _, _, X, X_test, y, y_test = load_data(group=group, test_size=test_size, sub=False)

X_0_names = list(X.columns)

# ___ SCALING
df = X
scaler = MinMaxScaler().fit(df)
df = scaler.transform(df)

#__________________________________________________
"""
CLUSTERING
"""
# clustering namespaces
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# ___ Number of clusters
"""
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
"""

n_clusters = 5
#_________________________
"""Baseline K-MEANS"""

kmeans_raw = KMeans(n_clusters=n_clusters, n_init=20).fit(df)  # 20 mal (gegen lokale minima)
kmeans_y = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(df)
accuracy_score(y_0, kmeans_y)

clusters_km, df_km = cl_result(dat, kmeans_y, n_clusters)
# clusters_km.to_csv(path_or_buf='../../data/results/clusters_km.csv', index = True)
"""
# ___ WITH LATENT
kmeans_l = KMeans(n_clusters=n_clusters, n_init=20).fit(latent)  # 20 mal (gegen lokale minima)
kmeans_l_y = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(latent)  # 20 mal (gegen lokale minima)
accuracy_score(y_0, kmeans_l_y)

dfc = np.zeros((n, c+1))
dfc[:, :-1] = dat
dfc[:, -1] = kmeans_l.labels_

for i in range(1, n_clusters+1):
    print(np.sum(dfc[dfc[:, 16] == (i-1), 15])/dfc[dfc[:, 16] == (i-1), 15].shape, i-1)
"""

#_________________________
"""Gaussian mixture"""

gmix = GaussianMixture(n_components=n_clusters, random_state=0).fit(df)
gm_clusters = gmix.predict(df)

clusters_gm, df_gm = cl_result(dat, gm_clusters, n_clusters)

# clusters_gm.to_csv(path_or_buf='../../data/results/clusters_gm.csv', index = True)


