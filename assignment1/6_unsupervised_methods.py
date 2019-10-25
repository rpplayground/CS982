
#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 6 - Unsupervised Analysis
#The purpose of this notebook is to run a range of unsupervised models on the data.
#The objective is to determine how successfully clustering models can be used to label given the *ground truth* of *Income GrouP*.
#
# The following high level steps have been applied:
# 1. Hierarchical - bottom up - Agglomerative
# 2. K Means
#
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import assignment1.data_wrangling_functions as dwf
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder


#%%
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa",\
    "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
region_palette = {"North America" : "red", "Europe & Central Asia" : "blue",\
    "Middle East & North Africa" : "pink", "Latin America & Caribbean" : "purple",\
        "East Asia & Pacific" : "green", "South Asia" : "orange", "Sub-Saharan Africa" : "gray"}


#%% [markdown]
### Stage 6.1 - Read in the Data and Prepare it for Analysis
# Read in the file that was generated from the warngling and cleaning scripts.

#%%
# github_path = "C:/Users/Barry/"
github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")
#%% [markdown]
# This step was added retrospectively, because inspection of the data shows that we will be required to work with "Log GDP" in order to generate more meaningful analysis.

#%%
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])
interpolated_data_set_flattened = interpolated_data_set.reset_index()

#%%
mean_by_region_and_year = interpolated_data_set.groupby(level=["Region", "Year"]).mean().reset_index()
mean_by_region_and_year.index.name = 'ID'

#%%
mean_by_region_and_decade = interpolated_data_set.groupby(level=["Region", "Decade"]).mean().reset_index()
mean_by_region_and_decade.index.name = 'ID'


#%%
mean_by_country_and_decade = interpolated_data_set.groupby(level=["Region", "Country", "Decade"]).mean().reset_index()
mean_by_country_and_decade.index.name = 'ID'

#%%
interpolated_data_set_flattened.columns

#%%
interpolated_data_set_flattened.head(10)

#%% [markdown]
### Stage 6.2 - Agglomerative Model
# Blurb.

#%%
# Dropping NA values as they will cause the model to break!
interpolated_data_set_flattened_nulls_removed = interpolated_data_set_flattened.\
    loc[:, ["Income Group",\
         "Log GDP per Capita",\
              "Life expectancy at birth, total (years)",\
                   "Population growth (annual %)",
                        "Year"]]\
        .dropna()

#%%
ground_truth = interpolated_data_set_flattened_nulls_removed.iloc[:,0]
ground_truth.head(10)

#%%
ground_truth.value_counts()

#%%
dataset_variables = interpolated_data_set_flattened_nulls_removed.iloc[:, 1:]
dataset_variables.head(10)

#%%
dataset_variables.describe()

#%% [markdown]
# 4. Scale the data that we are going to use for clustering
#%%
X = pd.DataFrame(scale(dataset_variables))

#%%
X.head(10)

#%%
X.describe()

#%% [markdown]
# 5. We know that there are 3 possible categories for the data. Create 3 data clusters using #Agglomerative Hierarchical Clustering.
# What are the silhouette score, homogeneity and #completeness for these clusters?
# (Helping hand, if you need to convert labels from strings to #something else look at sklearn.preprocessing.LabelEncoder())
#%%
n_samples, n_features = X.shape
print("n_samples: " + str(n_samples) + " - n_features: " + str(n_features))

#%%
target_number_of_clusters = len(np.unique(ground_truth))
target_number_of_clusters

#%%
model = cluster.AgglomerativeClustering(n_clusters=target_number_of_clusters, linkage='average', affinity='cosine')

#%%
model.fit(X)


#%% [markdown]
# 6. What are the impact of different distance and affinity measures on the silhouette score, homogeneity and completeness for these clusters?
# Options available at http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# What is the best combination?
#%%
# To evaluate the performance of the model, we need to encode the L, B, R classification as integers..
Y = LabelEncoder().fit_transform(ground_truth)
Y

#%%
len(Y)

#%%
# Let's look at the labels that the model has assigned
model.labels_

#%%
len(model.labels_)

#%% [markdown]
# Now evaluate the performance of the culstering algorithm

#%% [markdown]
### Silhouette Score 
# The Silhouette Coefficient is calculated using:
# - the mean intra-cluster distance (a), and;
# -  the mean nearest-cluster distance (b) for each sample.
#
#The Silhouette Coefficient for a sample is (b - a) / max(a, b).
#
# Advantages:
# - The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
# - The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

#%%
metrics.silhouette_score(X, model.labels_)

#%% [markdown]
### Completeness Score
# Completeness metric of a cluster labeling given a *ground truth*.
# A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.

#%%
metrics.completeness_score(Y, model.labels_)

#%% [markdown]
### Homogeneity Score
#Homogeneity metric of a cluster labeling given a ground truth.
#A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.

#%%
metrics.homogeneity_score(Y, model.labels_)

#%%
interpolated_data_set_flattened_nulls_removed["agglomeative_cluster"] = model.labels_

#%%
def label_income_group (row):
    if row["Income Group"] == "High income":
        return 0
    if row["Income Group"] == "Upper middle income":
        return 3
    if row["Income Group"] == "Lower middle income":
        return 2
    if row["Income Group"] == "Low income":
        return 1
    return 4

#%%
interpolated_data_set_flattened_nulls_removed['income_group_label'] = \
    interpolated_data_set_flattened_nulls_removed.apply (lambda row: label_income_group(row), axis=1)

interpolated_data_set_flattened_nulls_removed.head(10)

#%%
interpolated_data_set_flattened_nulls_removed['label_diff'] = interpolated_data_set_flattened_nulls_removed['income_group_label']\
     - interpolated_data_set_flattened_nulls_removed['agglomeative_cluster']

interpolated_data_set_flattened_nulls_removed.head(10)

#%%
income_group_palette = {"High income": "red", "Upper middle income": "orange", "Lower middle income": "green", "Low income": "blue"}
cluster_palette = {3: "red", 0: "orange", 2: "green", 1: "blue"}


#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="Income Group",\
    palette = income_group_palette,\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed, ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="agglomeative_cluster",\
    palette = cluster_palette,\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed, ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="label_diff",\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed, ax=ax)

#%%
interpolated_data_set_flattened_nulls_removed_2018 = \
    interpolated_data_set_flattened_nulls_removed.\
        loc[interpolated_data_set_flattened_nulls_removed['Year'] == 2018]

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="Income Group",\
    palette = income_group_palette,\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed_2018, ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="agglomeative_cluster",\
    palette = cluster_palette,\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed_2018, ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="label_diff",\
    #size="Population, total",\
    #hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=interpolated_data_set_flattened_nulls_removed_2018, ax=ax)
