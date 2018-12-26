import pandas as pd

df = pd.read_csv('input/log.csv')

from sklearn.cluster import k_means

n_clusters = 2
centroid, label, inertia = k_means(df, n_clusters, random_state=0)

df0 = df[label == 0]
df0_mean = df0.mean()

df1 = df[label == 1]
df1_mean = df1.mean()
df1_mean


import matplotlib.pyplot as plto
plt.scatter(df0.Recruit, df0.Custom)
plt.scatter(df1.Recruit, df1.Custom)
plt.xlabel('Recruit')
plt.ylabel('Custom');