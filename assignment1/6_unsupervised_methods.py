
#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 6 - Application of Unsupervised Methods
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
from scipy.cluster.hierarchy import dendrogram, linkage

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
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156.DS/"
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
analysis_of_decade = interpolated_data_set.xs("1990s", level="Decade", drop_level=False)
analysis_of_decade = analysis_of_decade.groupby(level=["Region", "Income Group", "Country", "Decade"]).\
    mean().reset_index()

#%%
analysis_of_decade

#%% [markdown]
### Stage 6.2 - Agglomerative Model
# Really useful documentation about this model in the [SciKit Learn web site](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
#
# "Recursively merges the pair of clusters that minimally increases a given linkage distance."
#
# Key parameters are as follows:
# - Affinity - method used to compute the linkage - can be: “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”;
# - Linkage - which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
#
#%%
# Dropping NA values as they will cause the model to break!
analysis_of_decade_nulls_removed = analysis_of_decade.\
    loc[:, ["Income Group",\
         "Log GDP per Capita",\
              "Life expectancy at birth, total (years)",\
                   "Renewable energy consumption (% of total final energy consumption)",\
                        "Immunization, DPT (% of children ages 12-23 months)",\
                            "Urban population growth (annual %)",\
                                "Mortality rate, infant (per 1,000 live births)"]]\
        .dropna()

#%%
ground_truth = analysis_of_decade_nulls_removed.iloc[:,0]
ground_truth.head(10)

#%%
ground_truth.value_counts()

#%%
dataset_variables = analysis_of_decade_nulls_removed.iloc[:, 1:]
dataset_variables.head(10)

#%%
dataset_variables.describe()

#%% [markdown]
# 4. Scale the data that we are going to use for clustering
#%%
X = scale(dataset_variables)
X

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
analysis_of_decade_nulls_removed["agglomeative_cluster"] = model.labels_

#%%
def label_income_group (row):
    if row["Income Group"] == "High income":
        return 0
    if row["Income Group"] == "Upper middle income":
        return 2
    if row["Income Group"] == "Lower middle income":
        return 3
    if row["Income Group"] == "Low income":
        return 1
    return 4

#%%
analysis_of_decade_nulls_removed['income_group_label'] = \
    analysis_of_decade_nulls_removed.apply (lambda row: label_income_group(row), axis=1)

analysis_of_decade_nulls_removed.head(10)

#%%
analysis_of_decade_nulls_removed['label_diff'] = analysis_of_decade_nulls_removed['income_group_label']\
     - analysis_of_decade_nulls_removed['agglomeative_cluster']

analysis_of_decade_nulls_removed.head(10)

#%%
income_group_palette = {"High income": "red", "Upper middle income": "orange", "Lower middle income": "green", "Low income": "blue"}
#cluster_palette = {3: "red", 0: "orange", 2: "green", 1: "blue"}
cluster_palette = {3: "green", 0: "red", 2: "orange", 1: "blue"}
diff_palette = {4: "darkred", 3: "red", 2: "orangered", 1: "orange", 0: "black",\
    -4: "darkgreen", -3: "seagreen", -2: "green", -1: "lightgreen"}


#%%
def plot_clusters(dataframe, x_column, y_column, hue_column, hue_palette):
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    f, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x = x_column, y = y_column,\
        hue = hue_column,\
        palette = hue_palette,\
        sizes = 100,\
        alpha = 0.8,\
        linewidth = 0,\
        data=dataframe, ax = ax)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Log GDP per Capita",\
    "Life expectancy at birth, total (years)", "Income Group", income_group_palette)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Log GDP per Capita",\
    "Life expectancy at birth, total (years)", "agglomeative_cluster", cluster_palette)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Log GDP per Capita",\
    "Life expectancy at birth, total (years)", "label_diff", diff_palette)

#%% [markdown]
### Dendogram

#%%
linkage_model = linkage(X, 'ward')
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(linkage_model, leaf_rotation=90., leaf_font_size=8.,)
locs, labels = plt.xticks()
plt.show()

#%%
labels[0]

#%%
X


#%%
def run_agglomerative_model(X, Y, target_number_of_clusters):
    affinity_list = ["euclidean", "l1", "l2", "manhattan", "cosine"]
    linkage_list = ["ward", "complete", "average"]
    results_dataframe = pd.DataFrame([])
    for affinity_option in affinity_list:
        for linkage_option in linkage_list:
            if(linkage_option == "ward" and affinity_option != "euclidean"):
                # If linkage is “ward”, only “euclidean” is accepted. 
                continue
            else:
                agglomerative_model = cluster.AgglomerativeClustering(n_clusters=target_number_of_clusters,\
                    linkage=linkage_option, affinity=affinity_option)
                agglomerative_model.fit(X)
                silhouette_score = metrics.silhouette_score(X, agglomerative_model.labels_)
                completeness_score = metrics.completeness_score(Y, agglomerative_model.labels_)
                homogeneity_score = metrics.homogeneity_score(Y, agglomerative_model.labels_)
                print("Affinity: {}, Linkage: {}, Silhouette: {:0.3f}, Completeness: {:0.3f}, Homogeneity: {:0.3f}".\
                    format(affinity_option, linkage_option, silhouette_score, completeness_score, homogeneity_score))
                results_dataframe = results_dataframe.append({"Affinity" : affinity_option, "Linkage" : linkage_option,\
                    "Silhouette" : silhouette_score, "Completeness" : completeness_score, "Homogeneity" : homogeneity_score}, ignore_index=True)
    return results_dataframe
    

#%%
agglomerative_dataframe = run_agglomerative_model(X, Y, target_number_of_clusters)
agglomerative_dataframe



#%% [markdown]
### Stage 6.3 - K Means
# Useful documentation from Scikit Learn web site:
# [https://scikit-learn.org/stable/modules/clustering.html#k-means] (https://scikit-learn.org/stable/modules/clustering.html#k-means)
# 
# Documentation for the 

#%%
def run_k_means_model(X, Y, max_number_of_clusters):
    results_dataframe = pd.DataFrame([])
    for k_means_number_of_clusters in range(2, max_number_of_clusters+1):
        # Configure the model
        kmeans_model = cluster.KMeans(n_clusters=k_means_number_of_clusters)
        # Run the model
        kmeans_model.fit(X)
        # Score the model
        silhouette_score = metrics.silhouette_score(X, kmeans_model.labels_)
        completeness_score = metrics.completeness_score(Y, kmeans_model.labels_)
        homogeneity_score = metrics.homogeneity_score(Y, kmeans_model.labels_)
        # Print the results
        print("Number of clusters: {}, Silhouette: {:0.3f}, Completeness: {:0.3f}, Homogeneity: {:0.3f}".\
        format(k_means_number_of_clusters, silhouette_score, completeness_score, homogeneity_score))
        # Capture the results
        results_dataframe = results_dataframe.append({"k_means_number_of_clusters" : k_means_number_of_clusters,\
        "silhouette_score" : silhouette_score, "completeness_score" : completeness_score, "homogeneity_score" : homogeneity_score}, ignore_index=True)
    return results_dataframe

#%%
kmeans_dataframe = run_k_means_model(X, Y, 10)


#%% [markdown]
### Principle Components Analysis?
#  Read about Principle Component Analysis and look at this example:
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
#
# Consider how it could be applied to clustering in our examples.
