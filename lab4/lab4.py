#%% [markdown]
# CS982: Big Data Technologies
## Lab 4 - Unsupervised Methods

#%% [markdown]
# 1. Import the necessary libraries (numpy, pandas, scikit-learn packages metrics and clustering)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale

#%% [markdown]
# 2. Import the Balance Scale dataset available at
# http://archive.ics.uci.edu/ml/datasets/balance+scale

#%%
dataset = pd.read_csv("./lab4/balance-scale.data")
dataset.head(10)

#%%
labels = dataset.values[:,0]
data = dataset.values[:,1:5]

#%%
labels = dataset[0]
data = dataset.drop(0,axis=1)

#%%
labels

#%%
data

#%%
scaled_data = scale(data)

#%%
scaled_data

#%%
#dataset = dataset.rename(columns={0 : "class", 1 : "left_weight", 2 : "left_distance", 3 : "right_weight", 4 :"right_distance"})
#dataset["class"] = dataset["class"].astype('category')
#dataset.head(10)

#%%
dataset.dtypes

#%%
dataset.shape

#%% [markdown]
# 3. This data set was generated to model psychological experimental results. Each example is
# classified as having the balance scale tip to the right, tip to the left, or be balanced. The
# attributes are the left weight, the left distance, the right weight, and the right distance. The
# correct way to find the class is the greater of (left-distance * left-weight) and (right-distance *
# right-weight). If they are equal, it is balanced.
#
# Segment the outcome (first column) and remaining data (attributes) so we can use the attributes for clustering.
#%%
dataset_outcome = dataset.loc[:, 0]
dataset_outcome.head(10)

#%%
dataset_variables = dataset.loc[:, [1, 2, 3, 4]]
dataset_variables.head(10)

#%%
dataset_variables.describe()

#%% [markdown]
#4. Scale the data that we are going to use for clustering
#%%
dataset_variables_scaled = pd.DataFrame(scale(dataset_variables))

#%%
dataset_variables_scaled.head(10)

#%%
dataset_variables_scaled.shape

#%%
dataset_variables_scaled.describe()


#%% [markdown]
#5. We know that there are 3 possible categories for the data. Create 3 data clusters using
#Agglomerative Hierarchical Clustering. What are the silhouette score, homogeneity and
#completeness for these clusters? (Helping hand, if you need to convert labels from strings to
#something else look at sklearn.preprocessing.LabelEncoder())
#%%
n_samples, n_features = dataset_variables_scaled.shape
print("n_samples: " + str(n_samples) + " - n_features: " + str(n_features))

#%%
n_digits = len(np.unique(dataset_outcome))
n_digits

#%%
model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage='average', affinity='cosine')

#%%
model.fit(dataset_variables_scaled)


#%% [markdown]
#6. What are the impact of different distance and affinity measures on the silhouette score,
#homogeneity and completeness for these clusters (options available at http://scikitlearn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)? What is the
#best combination?
#%%

#%% [markdown]
7. What are the silhouette score, homogeneity and completeness for different numbers of clusters
created using KMeans?
#%%

#%% [markdown]
8. Additional exercise. Read about Principle Component Analysis and look at this example -
http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html Consider how it
could be applied to clustering in our examples.
#%%

#%% [markdown]
9. Using your interesting dataset that you have downloaded for analysis or example datasets from
the lecture
i. Use some of the techniques explored in the lecture try to examine your dataset
ii. Reflect on your interesting question(s) and think about answering that question with
these methods
iii. Consider which of these methods (if any) might be good for answering your question(s)