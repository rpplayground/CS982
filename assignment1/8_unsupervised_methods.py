
#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created Xth October 2019 by Barry Smart.
# 
## Stage 8 - Application of Unsupervised Methods
#The purpose of this notebook is to run a range of unsupervised models on the data.
#The objective is to determine how successfully clustering models can be used to label given the *ground truth* of *Income GrouP*.
#
# The following high level steps have been applied:
# 1. Hierarchical - bottom up - Agglomerative
# 2. K Means
# 3. Principle Component Analysis - as a useful pre-emtpive stage before applying supervised methods - with a view to reducing the number of variables required.
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


#%% [markdown]
### Stage 8.1 - Read in the Data and Prepare it for Analysis
# Read in the file that was generated from the warngling and cleaning scripts.

#%%
#github_path = "C:/Users/Barry/"
github_path = "C:/Users/cgb19156.DS/"
data_path = github_path + "GitHub/CS982/assignment1/"
interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")
#%% [markdown]
# This step was added retrospectively, because inspection of the data shows that we will be required to work with "Log GDP" in order to generate more meaningful analysis.

#%%
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])

#%% [markdown]
### Stage 8.2 - Prepare Data For Models
# Analysis will be carried out based on the average for each country within a specific decade - the logic being that this will enable fluctuations within that decade to be generalised using the mean.
# Based on the analysis of correlations between variables, reduce the number of variables that will be passed to the model.
#

#%%
# Summarise dataset by calculating mean values for the 1990s decade
analysis_of_decade = interpolated_data_set.xs("1990s", level="Decade", drop_level=False)
analysis_of_decade = analysis_of_decade.groupby(level=["Region", "Income Group", "Country", "Decade"]).\
    mean().reset_index()

#%%
# Reduce number of variables and dropping any NA values as they will cause the model to break!
analysis_of_decade_nulls_removed = analysis_of_decade.\
    loc[:, ["Income Group",\
         "Log GDP per Capita",\
              "Life expectancy at birth, total (years)",\
                   "Renewable energy consumption (% of total final energy consumption)",\
                       'Inflation, consumer prices (annual %)',\
                            'Merchandise exports (current US$)',\
                                "Immunization, DPT (% of children ages 12-23 months)",\
                                    "Urban population growth (annual %)",\
                                        "Mortality rate, infant (per 1,000 live births)"]]\
        .dropna()

analysis_of_decade_nulls_removed

#%%
ground_truth = analysis_of_decade_nulls_removed.iloc[:,0]
ground_truth.value_counts()

#%%
dataset_variables = analysis_of_decade_nulls_removed.iloc[:, 1:]
dataset_variables.describe()

#%% [markdown]
### Stage 8.3 - Establish The Variables (X)

#%%
X = scale(dataset_variables)
X

#%%
n_samples, n_features = X.shape
print("n_samples: " + str(n_samples) + " - n_features: " + str(n_features))

#%% [markdown]
### Stage 8.3 - Establish The Ground Truth (Y)

#%%
target_number_of_clusters = len(np.unique(ground_truth))
target_number_of_clusters

#%%
Y = LabelEncoder().fit_transform(ground_truth)
Y

#%% [markdown]
### Stage 8.4 - Agglomerative Model
# Really useful documentation about this model in the [SciKit Learn web site](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
#
# "Recursively merges the pair of clusters that minimally increases a given linkage distance."
#
# Key parameters are as follows:
# - Affinity - method used to compute the linkage - can be: “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”;
# - Linkage - which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
#
# Explore the of different distance and affinity measures on the silhouette score, homogeneity and completeness, based on passing different parameters to the Agglomerative clustering algorithm.
# Options available at http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

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
                #print("Affinity: {}, Linkage: {}, Silhouette: {:0.3f}, Completeness: {:0.3f}, Homogeneity: {:0.3f}".\
                #    format(affinity_option, linkage_option, silhouette_score, completeness_score, homogeneity_score))
                results_dataframe = results_dataframe.append({"Affinity" : affinity_option, "Linkage" : linkage_option,\
                    "Silhouette" : silhouette_score, "Completeness" : completeness_score, "Homogeneity" : homogeneity_score}, ignore_index=True)
    return results_dataframe
    
#%%
agglomerative_paramter_analysis = run_agglomerative_model(X, Y, target_number_of_clusters)
agglomerative_paramter_analysis

#%%
# Save data frame to CSV so that it can be imported into document.
agglomerative_paramter_analysis.to_csv("./assignment1/agglomerative_paramter_analysis.csv")

#%% [markdown]
### Stage 8.5 - K Means
# Useful documentation from Scikit Learn web site:
# [https://scikit-learn.org/stable/modules/clustering.html#k-means] (https://scikit-learn.org/stable/modules/clustering.html#k-means)
# 
# The objective here is to iterative through a range of target clusters for the K-means algorithm to assess how it performs.

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
        # print("Number of clusters: {}, Silhouette: {:0.3f}, Completeness: {:0.3f}, Homogeneity: {:0.3f}".\
        # format(k_means_number_of_clusters, silhouette_score, completeness_score, homogeneity_score))
        # Capture the results
        results_dataframe = results_dataframe.append({"k_means_number_of_clusters" : k_means_number_of_clusters,\
        "silhouette_score" : silhouette_score, "completeness_score" : completeness_score, "homogeneity_score" : homogeneity_score}, ignore_index=True)
    return results_dataframe

#%%
kmeans_parameter_analysis = run_k_means_model(X, Y, 10)
kmeans_parameter_analysis = kmeans_parameter_analysis.set_index("k_means_number_of_clusters")
kmeans_parameter_analysis

#%%
score_palette = {"completeness_score": "blue", "silhouette_score": "green", "homogeneity_score": "purple"}
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-', })
plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
    'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
f, ax = plt.subplots(figsize=(10, 6))
plt.title("Analysis of K-Means Permance\nAs Number of Clusters Is Increased From 2 to 10", fontdict = {"fontsize" : 20})
sns.lineplot(linewidth=3,\
    palette=score_palette,\
    data=kmeans_parameter_analysis, ax=ax)

#%%
# Save results to file so that it can be imported into document.
kmeans_parameter_analysis.to_csv("./assignment1/kmeans_parameter_analysis.csv")

#%% [markdown]
### Stage 8.6 Run Agglomerative Clustering With Optimal Parameters 
# The best combination being:
# - Affinity = Euclidean
# - Linkage = Ward
#

#%%
model = cluster.AgglomerativeClustering(n_clusters=target_number_of_clusters, linkage='ward', affinity='euclidean')

#%%
model.fit(X)

#%%
# Let's look at the labels that the model has assigned
model.labels_

#%%
# The number of labels should tally with the number of observations passed into the model.
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

#%% [markdown]
### Stage 8.7 Visualise Results Of Agglomerative Clustering
# Visualise how the aggromerative model has performed when compared to the "ground truth" labels.
#

#%%
# Splice the labels assigned by the Agglomerative cluster back onto the original data set:
analysis_of_decade_nulls_removed["agglomeative_cluster"] = model.labels_

#%%
# To enable evaluation of model, assign integer labels to each row so that they align in general with the labels assigned by the clustering algorith.
def label_income_group (row):
    if row["Income Group"] == "High income":
        return 2
    if row["Income Group"] == "Upper middle income":
        return 0
    if row["Income Group"] == "Lower middle income":
        return 3
    if row["Income Group"] == "Low income":
        return 1
    return 4

analysis_of_decade_nulls_removed['income_group_label'] = \
    analysis_of_decade_nulls_removed.apply (lambda row: label_income_group(row), axis=1)

#%%
# Now compute the difference between "known truth" (Income Group) and labels assigned by model
analysis_of_decade_nulls_removed['label_diff'] = analysis_of_decade_nulls_removed['income_group_label']\
     - analysis_of_decade_nulls_removed['agglomeative_cluster']

#%%
# Set up palette for plot to enable like with like comparison
income_group_palette = {"High income": "red", "Upper middle income": "orange", "Lower middle income": "green", "Low income": "blue"}
cluster_palette = { 2: "red", 0: "orange", 3: "green", 1: "blue"}
diff_palette = {4: "darkred", 3: "red", 2: "orangered", 1: "orange", 0: "black",\
    -4: "darkgreen", -3: "seagreen", -2: "green", -1: "lightgreen"}


#%%
# Helper function to enable data to be plotted consistently
def plot_clusters(dataframe, title, x_column, y_column, hue_column, hue_palette):
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--', 
        'axes.titlesize' : 18, 'lines.linewidth' : 1, 'axes.labelsize' : 18, 'xtick.labelsize' : 16,\
        'ytick.labelsize' : 16})
    f, ax = plt.subplots(figsize=(10, 6))
    chart = sns.scatterplot(x = x_column, y = y_column,\
        hue = hue_column,\
        palette = hue_palette,\
        sizes = 100,\
        alpha = 0.8,\
        linewidth = 0,\
        data=dataframe, ax = ax)
    chart.axes.set_title(title,fontsize=18)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Plot of Log GDP Per Capita and Life Expectancy\n For Each Country - Average For 1990s:\nShowing \"Ground Truth\" (Income Group) Labels", "Log GDP per Capita",\
    "Life expectancy at birth, total (years)", "Income Group", income_group_palette)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Plot of Log GDP Per Capita and Life Expectancy\n For Each Country - Average For 1990s:\nShowing Labels Assigned by Agglomerative Clustering",\
     "Log GDP per Capita", "Life expectancy at birth, total (years)",\
          "agglomeative_cluster", cluster_palette)

#%%
plot_clusters(analysis_of_decade_nulls_removed, "Plot Showing Differences Between \"Known Truth\"\nand Labels Assigned By Clustering Algorithm",\
    "Log GDP per Capita", "Life expectancy at birth, total (years)",\
        "label_diff", diff_palette)

#%% [markdown]
### Stage 8.8 Generate a Dendrogram
# Visualise how the aggromerative model has been contructed from the ground up.
#

#%%
linkage_model = linkage(X, 'ward')
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(25, 15))
plt.title('Hierarchical Clustering Dendrogram - Applied To A Range of World Development Indicators in the 1990s')
plt.xlabel('Countries')
plt.ylabel('Distance')
dendrogram(linkage_model, leaf_rotation=90., leaf_font_size=9.,)
locs, labels = plt.xticks()
plt.show()

#%%
labels[0]

#%%
linkage_model

#%%
X

#%% [markdown]
### Stage 8.9 - Principle Components Analysis
# Principle component analysis (PCA) is fundamentally a dimensionality reduction algorthim.
# Useful for visualisation, noise filtering, feature extraction and engineering.
# According to Vanderplas 2017 it is a fast and flexible method for dimensionality reduction in data.
# 
# My plan here is to apply a PCA to my data by asking it to fit to a decreasing number of components.
# I will be passing D variables, and intially ask it to fit to D-1 components.
# Then I will step this down to 1 and plot the results.
# 
# Documentation:
# (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)[https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html]
#
# This is also interesting:
# (http://setosa.io/ev/eigenvectors-and-eigenvalues/)[http://setosa.io/ev/eigenvectors-and-eigenvalues/]
#%%
from sklearn.decomposition import PCA

#%%
pca = PCA()

#%%
pca = pca.fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%%
pca.explained_variance_ratio_

#%%
pca.explained_variance_

#%%
pca.components_


