#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 7 - Analysis Of Core Data Series
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

#%% [markdown]
### Stage 7.1 - Read The File
# Read in the file that was generated from the previous script.

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")
#%% [markdown]
### Stage 7.2 - Add Additional Calculated Columns
# This step was added retrospectively, because inspection of the data shows that we will be required to work with "Log GDP" in order to generate more meaningful analysis.

#%%
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])
interpolated_data_set.head(10)

# Useful link for slicing multi indexes:
# [https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced)
#
# df.loc[(slice('A1', 'A3'), ...), :]

#%% [markdown]
### Stage 7.3 - Narrow Analysis To Data For 2018
# In this section I will analyse data for 2018 in more depth with the purpose of:
# - Generating some initial insights into the data;
# - Trying to bring out the correlations between data;
#
#%%
# Use the xs function to grab a cross section of the data, using the power of the multi-level-index.
analysis_of_2018 = interpolated_data_set.xs(2018, level="Year", drop_level=False)
# Flatten the dataframe to open up all columns for access by the matplotlib and seaborn libraries.
analysis_of_2018_flattened = analysis_of_2018.reset_index()
analysis_of_2018_flattened.index.name = 'ID'
analysis_of_2018_flattened.describe()

#%% [markdown]
### Stage 7.4 - Analysis of Life Expectancy
#Using a number of techniques to get a feel for the life expectany data:
# - Looking at top 10 and bottom 10 countries in 2018;
# - Distribution of data by region in 2018;
# - Analysing how it has developed over time since 1960.
#
#%% [markdown]
#### Life Expectancy 2018 - Top 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Life expectancy at birth, total (years)"]]\
    .nlargest(10, "Life expectancy at birth, total (years)")
#%% [markdown]
#### Life Expectancy 2018 - Bottom 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Life expectancy at birth, total (years)"]]\
    .nsmallest(10, "Life expectancy at birth, total (years)")
#%% [markdown]
#### Summary - Life Expectancy "League Tables" 2018
#Some shocking revelations:
# - The gap between the bottom and top is over 30 years!;
# - The bottom 10 countries are **ALL** in Sub-Saharan Africa.
#
#Later I should look at how this gap has developed over time.
#TODO add backlog item for this.
# 
#### Life Expectancy 2018 - Analysing Distribution of Data for All Countries
#Now using box plots to look at distribution by country.

#%%
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa", "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]

sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Distribution of Life Expectancy\nby Region in 2018", fontdict = {"fontsize" : 20})
box_plot = sns.boxplot(x="Life expectancy at birth, total (years)",\
    y="Region", order=region_ranking, data=analysis_of_2018_flattened, ax=ax)



#%% [markdown]
#### Notes
# There is a significant outlier for "Latin America & Caribbean" - it would be good to investigate this further.
# TODO - add a backlog item.
#

#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Life expectancy at birth, total (years)"]]\
    .loc[analysis_of_2018_flattened["Region"] == "Latin America & Caribbean"]\
    .nsmallest(10, "Life expectancy at birth, total (years)")

#%%
haiti_data = interpolated_data_set.reset_index()
haiti_data = haiti_data.loc[haiti_data["Country"] == "Lithuania"]
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--'})
f, ax = plt.subplots(figsize=(10, 10))
# ax.set(yscale="log")
#ax.set_ylim(-5, 10)
plt.title("YYY", fontdict = {"fontsize" : 20})
sns.lineplot(x="Year", y="Population, total",\
    color="gray", data=haiti_data, ax=ax)


#%% [mardown]
#### Development of Life Expectancy Over Time
#Lets now examine how the gap in life expectancy across different nations has developed over time.
#
#To do this we  need to:
# - Go back to our original interpolated_data data set so that we have access to the full data set rather than just 2018;
# - Roll up the data by calculating the mean of *ALL* series by Year for each Region.
# - Generage a line plot to show how each series develops over time.
#
#%%
mean_by_region_and_year = interpolated_data_set.groupby(level=["Region", "Year"]).mean()
mean_by_region_and_year

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Development of Life Expectancy by Region\nby Year since 1960", fontdict = {"fontsize" : 20})
sns.lineplot(x="Year", y="Life expectancy at birth, total (years)",\
    hue="Region",\
    #size="depth",\
    #palette="ch:r=-.2,d=.3_r",\
    hue_order=region_ranking,\
    sizes=(1, 8), linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
plt.title("YYY", fontdict = {"fontsize" : 20})
box_plot = sns.boxplot(x="Decade", color="gray",\
    y="Life expectancy at birth, total (years)", orient="v", data=interpolated_data_set.reset_index(), ax=ax)


#%% [markdown]
#### Conclusions - Development of Life Expectancy Over Time
#The following observations can be made from the data above:
# - The gap in life expectancy has closed (more than halved) between 6 of te 7 regions;
# - Meanwhile life expectancy for the Sub-Saharan Africa region has not improved at the same rate, mainly as a result of a plateau in the 1990s;
# - The net result is that the gap between those countries with the worst and best record for life expectancy has not closed appreciably since 1960.
#
#%%
### Stage 7.5 - Analysing Gross Domestic Product (GDP)
#Using a number of techniques to get a feel for the life expectany data:
# - Looking at top 10 and bottom 10 countries in 2018;
# - Distribution of data by region in 2018;
# - Analysing how it has developed over time since 1960.
#
#### GDP 2018 - Top 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "GDP per capita (current US$)"]]\
    .nlargest(10, "GDP per capita (current US$)")
#%% [markdown]
#### GDP 2018 - Bottom 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "GDP per capita (current US$)"]]\
    .nsmallest(10, "GDP per capita (current US$)")

#%% [markdown]
#### Summary - GDP "League Tables" 2018
#Some shocking revelations:
# - The gap between the bottom and top is three orders of magnitude!;
# - Building on the pattern seen with life expectancy, 9 of bottom 10 countries are in Sub-Saharan Africa.
#
#Later I should look at how this GDP gap has developed over time.
#TODO add backlog item for this.
# 
#### GDP 2018 - Analysing Distribution of Data for All Countries
#Now using box plots to look at distribution by country.

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Distribution of Gross Domestic Product (GDP)\nby Region in 2018", fontdict = {"fontsize" : 20})
box_plot = sns.boxplot(x="GDP per capita (current US$)",\
    y="Region", order=region_ranking, data=analysis_of_2018_flattened, ax=ax)

#%% [markdown]
#The data is quite skewed given that it is spread over 4 orders of magnitude.
#So lets try the same plot but use a logarithmic scale.
#

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
ax.set(xscale="log")
plt.title("Distribution of GDP by Region in 2018\nUsing Log10 Scale", fontdict = {"fontsize" : 20})
box_plot = sns.boxplot(x="GDP per capita (current US$)",\
    y="Region", order=region_ranking, data=analysis_of_2018_flattened, ax=ax)

#%% [markdown]
# It will be useful to create a new series of data in the dataframe that is the Log10 of GDP.
# We should go back earlier in the process to add a calculated column which is "log to base 10 of GDP"
# TODO - add a backlog item.
#
#### Development of GDP Over Time
#Lets now examine how the gap in GDP across different nations has developed over time.
#
#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Development of GDP by Region\nby Year Since 1960", fontdict = {"fontsize" : 20})
sns.lineplot(x="Year", y="GDP per capita (current US$)",\
    hue="Region",\
    hue_order=region_ranking,\
    linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
# ax.set(yscale="log")
plt.title("YYY", fontdict = {"fontsize" : 20})
sns.boxplot(x="Decade", color="gray",\
    y="GDP per capita (current US$)", orient="v", data=interpolated_data_set.reset_index(), ax=ax)

#%% [markdown]
#### Conclusions - Development of GDP Over Time
#The following observations can be made from the data above:
# - The gap in GDP has opened up significantly (exponentially?) since 1960;
# - The regions that are being left behind are South Asia and Sub-Saharan Africa ;
# - Looking at this from a wider perspective, it is the upper quartile countries that are accelerating away, leaving 75% of countries therefore lagging behind significantly.
# - This starts to open up serious questions about the distribution of wealth globally;
# - It also begins to indicate the likelihood of close correlation between GDP and Life Expectancy?
#
### Stage 7.6 - Analysing Population Trends
#Using a number of techniques to get a feel for the life expectany data:
# - Looking at top 10 and bottom 10 countries in 2018;
# - Distribution of data by region in 2018;
# - Analysing how it has developed over time since 1960.

#%% [markdown]
#### Population 2018 - Top 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total", "Population growth (annual %)"]]\
    .nlargest(10, "Population, total")
#%% [markdown]
#### Population 2018 - Bottom 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total", "Population growth (annual %)"]]\
    .nsmallest(10, "Population, total")

#%% [markdown]
#### Population Growth 2018 - Top 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total", "Population growth (annual %)"]]\
    .nlargest(10, "Population growth (annual %)")

#%% [markdown]
#### Population Growth 2018 - Bottom 10 Countries

#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total", "Population growth (annual %)"]]\
    .nsmallest(10, "Population growth (annual %)")

#%%
analysis_of_2018_flattened.loc[analysis_of_2018_flattened["Population growth (annual %)"] < 0]["Country"].count()


#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="Population, total",\
    hue="Region",\
    hue_order=region_ranking,\
    linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set(style="ticks")
f, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x="Year", y="Population growth (annual %)",\
    hue="Region",\
    hue_order=region_ranking,\
    linewidth=3,\
    data=mean_by_region_and_year.reset_index(), ax=ax)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
# ax.set(yscale="log")
ax.set_ylim(-5, 10)
plt.title("YYY", fontdict = {"fontsize" : 20})
sns.boxplot(x="Decade", y="Population growth (annual %)",\
    color="gray", data=interpolated_data_set.reset_index(), ax=ax)



#%%
