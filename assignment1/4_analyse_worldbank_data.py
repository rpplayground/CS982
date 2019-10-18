#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Part 7 - Analyse The Data
# The purpose of this notebook is to do some basic cleaning of the data.
#
#### Design Decisions
# The analysis will be conducted across the specific areas:
#- Looking at the major factors that drive life expectancy;
#- Trying to understand the factors that drive population growth.
#- Using the above to project world population forward based on some economic assumptions to estimate what the maximum populaton will reach.
#
#### Chart Styles to Try
# - Annotated Heatmap
# 
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import assignment1.data_wrangling_functions as dwf


#%% [markdown]
#### Stage 1 - Read The File
# Read in the file that was generated from the previous script.

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"

data_path = github_path + "GitHub/CS982/assignment1/"

interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")

#%%
interpolated_data_set.head(10)

#%%
interpolated_data_set.columns

# Useful link for slicing multi indexes:
# [https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced)
#
# df.loc[(slice('A1', 'A3'), ...), :]
#%%
analysis_of_2018 = interpolated_data_set.xs(2018, level="Year", drop_level=False)

#%%
analysis_of_2018

#%%
analysis_of_2018["Log GDP per Capita"] = np.log10(analysis_of_2018["GDP per capita (current US$)"])


#%%
analysis_of_2018_flattened = analysis_of_2018.reset_index()

#%%
analysis_of_2018_flattened

#%%
#flattened_dataframe["Region Category"] = flattened_dataframe["Region"].astype('category')

#%%
#flattened_dataframe["Region Code"] = flattened_dataframe["Region Category"].cat.codes
#%%
analysis_of_2018_flattened.dtypes
#%%
analysis_of_2018_flattened.describe()
#%%
sns.boxplot(x="Life expectancy at birth, total (years)", y="Region", data=analysis_of_2018_flattened)

#%%
sns.boxplot(x="GDP per capita (current US$)", y="Region", data=analysis_of_2018_flattened)

#%%
analysis_of_2018_flattened["Log GDP per Capita"] = np.log(analysis_of_2018_flattened["GDP per capita (current US$)"])

#%%
sns.boxplot(x="Log GDP per Capita", y="Region", data=analysis_of_2018_flattened)

#%%
#sns.pairplot(analysis_of_2018_flattened, hue="Region")

#%%
analysis_of_2018_flattened["Population, total"].describe()

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa", "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    size="Population, total",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(40, 1000), linewidth=1,\
    data=analysis_of_2018_flattened, ax=ax)

#%%
analysis_of_2018_flattened["Region"].value_counts()

#%%
# Compute the correlation matrix
correlation_matrix = analysis_of_2018.corr()

#%%
# Generate a mask for the upper triangle
mask = np.zeros_like(correlation_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set(style="ticks")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask,\
    cmap=cmap,\
    vmax=.3, center=0,\
    square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%
mean_by_region = analysis_of_2018.groupby(level="Region").mean()

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa", "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
sns.scatterplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=0,\
    data=mean_by_region.reset_index(), ax=ax)


#%%
mean_by_region_and_decade = interpolated_data_set.groupby(level=["Region", "Decade"]).mean()

#%%
mean_by_region_and_decade

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Decade", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_decade.reset_index(), ax=ax)

#%%
mean_by_region_and_year = interpolated_data_set.groupby(level=["Region", "Year"]).mean()

#%%
mean_by_region_and_year

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="GDP per capita (current US$)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="Population growth (annual %)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="Population, total",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(20, 15))
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa", "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
sns.boxplot(x="Decade", y="Population growth (annual %)",\
    hue="Region", data=interpolated_data_set.reset_index(), ax=ax)

#%%
max = analysis_of_2018_flattened["Population, total"].max()

#%%
min = analysis_of_2018_flattened["Population, total"].min()


#%%
import math
def custom_round(x, up=True):
    if up:
        return math.pow(10, math.ceil(math.log10(x)))
    else:
        return math.pow(10, math.floor(math.log10(x)))

#%%
custom_round(max)

#%%
#%%
custom_round(min, up=False)

#%%

#%%
min_power, min_value, max_power, max_value = dwf.find_min_and_max(analysis_of_2018_flattened, "GDP per capita (current US$)")

#%%
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
#         s = max_x["Country Code"],\
#         xy = (max_x["GDP per capita (current US$)"], max_x["Life expectancy at birth, total (years)"]),\
#         bbox = dict(boxstyle="round", fc="0.8"))

#plot_points_of_interest(analysis_of_2018_flattened, [("CHN", "HP")], "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Country Code", ax)

dwf.label_max_and_mins(analysis_of_2018_flattened, "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Country Code", ax)


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
#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
sns.lmplot(x="Log GDP per Capita", y="Population growth (annual %)",\
    height=8, data=analysis_of_2018_flattened)


