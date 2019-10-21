#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 8 - Analysis Of Core Data Series
#The purpose of this notebook is to do analysis of the core data items that are of interest:
#- Life expectancy;
#- Economic prosperity;
#- Population growth.
#
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import assignment1.data_wrangling_functions as dwf

#%%
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa",\
    "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
region_palette = {"North America" : "red", "Europe & Central Asia" : "blue",\
    "Middle East & North Africa" : "pink", "Latin America & Caribbean" : "purple",\
        "East Asia & Pacific" : "green", "South Asia" : "orange", "Sub-Saharan Africa" : "gray"}


#%% [markdown]
### Stage 7.1 - Read The File
# Read in the file that was generated from the previous script.

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"
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
# In this section I will analyse data for 2018 in more depth with the purpose of generating some simple initial insights into the data.
# Use the xs function to grab a cross section of the data, using the power of the multi-level-index.
analysis_of_2018 = interpolated_data_set.xs(2018, level="Year", drop_level=False)
# Flatten the dataframe to open up all columns for access by the matplotlib and seaborn libraries.
analysis_of_2018_flattened = analysis_of_2018.reset_index()
analysis_of_2018_flattened.index.name = 'ID'

#%% [markdown]
### Stage 7.2 - Heat Map Analysis of Correlations
# In this section I will analyse data for 2018 in more depth with the purpose of:
# - Generating some initial insights into the data;
# - Trying to bring out the correlations between data;
#
#First step is therefore to take a slice out of the data set.
#

#%%
# Compute the correlation matrix
correlation_matrix = interpolated_data_set.corr()

# Generate a mask for the upper triangle
#mask = np.zeros_like(correlation_matrix, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

sns.set(style="ticks")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix,\
    #mask=mask,\
    cmap=cmap,\
    vmax=1, center=0,\
    square=True, linewidths=.5, cbar_kws={"shrink": .5})



#%%
min_power, min_value, max_power, max_value = dwf.find_min_and_max(analysis_of_2018_flattened, "GDP per capita (current US$)")

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
ax.set(xscale="log")
ax.set_xlim(min_value, max_value)
sns.scatterplot(x="GDP per capita (current US$)", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    size="Population, total",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=analysis_of_2018_flattened, ax=ax)

max_x = analysis_of_2018_flattened.loc[analysis_of_2018_flattened["GDP per capita (current US$)"].idxmax()]

# ax.annotate(\
#         s = max_x["Country"],\
#         xy = (max_x["GDP per capita (current US$)"], max_x["Life expectancy at birth, total (years)"]),\
#         bbox = dict(boxstyle="round", fc="0.8"))

#plot_points_of_interest(analysis_of_2018_flattened, [("CHN", "HP")], "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Country", ax)

dwf.label_max_and_mins(analysis_of_2018_flattened, "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Country", ax)


#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
ax.set(xscale="log")
ax.set_xlim(min_value, max_value)
sns.scatterplot(x="GDP per capita (current US$)", y="Population growth (annual %)",\
    hue="Region",\
    size="Population, total",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(10,1000), linewidth=1,\
    data=analysis_of_2018_flattened, ax=ax)


#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
sns.lmplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    height=8, data=analysis_of_2018_flattened)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
sns.lmplot(x="Log GDP per Capita", y="Population growth (annual %)",\
    height=8, data=analysis_of_2018_flattened)




#%%
