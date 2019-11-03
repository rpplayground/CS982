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
from pandas.plotting import scatter_matrix
import assignment1.data_wrangling_functions as dwf

#%%
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa",\
    "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
region_palette = {"North America" : "red", "Europe & Central Asia" : "blue",\
    "Middle East & North Africa" : "pink", "Latin America & Caribbean" : "purple",\
        "East Asia & Pacific" : "green", "South Asia" : "orange", "Sub-Saharan Africa" : "gray"}
income_palette = {"High income" : "red", "Upper middle income" : "orange",\
    "Lower middle income" : "green", "Low income" : "blue"}

#%% [markdown]
### Stage 7.1 - Read The File
# Read in the file that was generated from the previous script.

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

interpolated_data_set_short_column_titles, list_of_columns = dwf.assign_short_variable_names(interpolated_data_set, 18)
interpolated_data_set_short_column_titles_flattened = interpolated_data_set_short_column_titles.reset_index()

#%% [markdown]
### Stage 7.2 - Heat Map Analysis of Correlations
# In this section I will analyse data for 2018 in more depth with the purpose of:
# - Generating some initial insights into the data;
# - Trying to bring out the correlations between data;
#
#### Initial Correlation Matric
# Run a quick analysis using a corelation matrix.

#%%
# Compute the correlation matrix
correlation_matrix = interpolated_data_set_short_column_titles.corr()

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

#%% [mardown]
### Stage 7.3 - Pair Plot
# Looking at the correlations above, pick out a smaller set of varaibles that have a strong correlation with Life Expectancy:
# - Mortality Rate, Infant
# - Log GDP Per Capita
# - Renewable Energy Usage
#  
# I will also run this analysis on a single year so as not to skew the results based on time series drift.
#
#%%
# This helper function will help me to grab data for a single year and for a smaller set of variables
def get_data_for_year(dataframe, year, list_of_columns):
    data_for_single_year = dataframe.xs(year, level="Year", drop_level=False)
    # Flatten and select only the columns I need
    data_for_single_year_flattened_and_trimmed = data_for_single_year.reset_index()[list_of_columns]
    return data_for_single_year_flattened_and_trimmed

#["Region", "Country", "Year", "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Population growth (annual %)"]

#%%
data_for_pair_plot = get_data_for_year(interpolated_data_set_short_column_titles, 2018,\
    ['Income Group', 'Life expectancy at..', 'Mortality rate, in..', 'Renewable energy c..', 'Log GDP per Capita'])

data_for_pair_plot = data_for_pair_plot.dropna()
data_for_pair_plot

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--', 
    'axes.titlesize' : 18, 'lines.linewidth' : 3, 'axes.labelsize' : 16, 'xtick.labelsize' : 16,\
    'ytick.labelsize' : 16})
chart = sns.pairplot(data_for_pair_plot, hue="Income Group", palette=income_palette)
chart.fig.suptitle("Analysis of Variables With Strongest Correlations For Year 2018", y=1.02, fontsize=20)

#%% [mardown]
### Stage 7.4 - Correlation Between Log GDP and Life Expectancy
#

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--', 
    'axes.titlesize' : 18, 'lines.linewidth' : 3, 'axes.labelsize' : 16, 'xtick.labelsize' : 16,\
    'ytick.labelsize' : 16})
chart = sns.lmplot(x='Log GDP per Capita', y='Life expectancy at..',\
    height=8, data=data_for_pair_plot)
chart.fig.suptitle("Analysis of Correlation Between\nLog GDP and Life Expectancy", y=1.02, fontsize=20)

#%% [mardown]
### Stage 7.4 - Correlation Between Log GDP and Life Expectancy
#

#%%
# Setting up consistent method of plotting the data with 5 dimensions!
def country_scatterplot(country_dataframe, title, x_column, y_column, hue_column, size_column, x_min, x_max, x_scale="linear", points_of_interest=False, label_maxs_and_mins=True):
    # Set up the style and size of the plot
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    f, ax = plt.subplots(figsize=(10, 10))
    ax.set(xscale=x_scale)
    ax.set_xlim(x_min, x_max)
    # Now plot the data
    chart = sns.scatterplot(x=x_column, y=y_column,\
        hue=hue_column,\
        size=size_column,\
        palette=region_palette,\
        hue_order=region_ranking,\
        sizes=(50,5000), linewidth=1,\
        alpha=1,\
        data=country_dataframe, ax=ax)
    chart.axes.set_title(title,fontsize=20)
    # Grab the handles and lavels for the legend
    h,l = ax.get_legend_handles_labels()
    # Prune the legend to the 7 Regions only - therefore removing the size legend
    new_legend = plt.legend(h[:8], l[:8], loc='best', ncol=1)
    # Add the new legend to the plot
    plt.gca().add_artist(new_legend)
    # Check to see if points of interest have been specified
    if points_of_interest:
        dwf.plot_points_of_interest(country_dataframe, points_of_interest,\
                x_column, y_column, size_column, "Country", ax)
    # Check if maximum and minimum values for x, y and size should be plotted
    if label_maxs_and_mins:
        dwf.label_max_and_mins(country_dataframe, x_column, y_column, size_column, "Country", ax)

    plt.show()


# This helper function will help me to grab data for a single year and for a smaller set of variables
def get_data_for_year(dataframe, year, list_of_columns):
    data_for_single_year = dataframe.xs(year, level="Year", drop_level=False)
    # Flatten and select only the columns I need
    data_for_single_year_flattened_and_trimmed = data_for_single_year.reset_index()[list_of_columns]
    return data_for_single_year_flattened_and_trimmed

#["Region", "Country", "Year", "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Population growth (annual %)"]

#%%
data_for_hans_rosling_scatter_1960 = get_data_for_year(interpolated_data_set, 1960,\
    ["Region", "Country", "Year", "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total"])
data_for_hans_rosling_scatter_1960


#%%
data_for_hans_rosling_scatter_2018 = get_data_for_year(interpolated_data_set, 2018,\
    ["Region", "Country", "Year", "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total"])
data_for_hans_rosling_scatter_2018


#%%
# Going to set the max and min for the X axis (GDP) based on the entire data set of 59 years to show how it has moved
min_power, min_value, max_power, max_value = dwf.find_min_and_max(interpolated_data_set_flattened, "GDP per capita (current US$)")

country_scatterplot(data_for_hans_rosling_scatter_1960, "Hans Rosling Inspired Scatter Plot\nCountry Data Across 4 Dimensions In The Year 1960:\nGDP, Life Expectancy, Region and Population",\
    "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Region", "Population, total",\
        min_value, max_value, x_scale="log",\
            points_of_interest=[("India", "QP"), ("United Kingdom", "QT"), ("Norway", "QT"), ("Sierra Leone", "QP"), ("China", "QP"), ("United States", "HP"), ("Afghanistan", "TR"), ("Rwanda", "OC"), ("Turkey", "QP"), ("Brazil", "QP")],\
                label_maxs_and_mins=False)

#%%
life_expectancy_league_table_1960 = data_for_hans_rosling_scatter_1960[["Region", "Country", "Life expectancy at birth, total (years)", "GDP per capita (current US$)"]].sort_values(by="Life expectancy at birth, total (years)").dropna()

#%%
life_expectancy_league_table_1960.head(10)

#%%
life_expectancy_league_table_1960.tail(10)


#%%
gdp_league_table_1960 = data_for_hans_rosling_scatter_1960[["Region", "Country", "Life expectancy at birth, total (years)", "GDP per capita (current US$)"]].sort_values(by="GDP per capita (current US$)").dropna()

#%%
gdp_league_table_1960.head(10)
#%%
gdp_league_table_1960.tail(10)

#%%
country_scatterplot(data_for_hans_rosling_scatter_2018, "Hans Rosling Inspired Scatter Plot\nCountry Data Across 4 Dimensions In The Year 2018:\nGDP, Life Expectancy, Region and Population",\
    "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Region", "Population, total",\
        min_value, max_value, x_scale="log",\
            points_of_interest=[("India", "QT"), ("United Kingdom", "QT"), ("Haiti", "QT"), ("Norway", "QT"), ("Central African Republic", "QP"), ("San Marino", "QT"), ("Burundi", "QT"), ("China", "QP"), ("United States", "QT"), ("Afghanistan", "QP"), ("Liechtenstein", "HP"), ("Indonesia", "QT"), ("Brazil", "QT"), ("Turkey", "QT"), ("Japan", "QT"), ("Sierra Leone", "QT")],\
                label_maxs_and_mins=False)



#%%
life_expectancy_league_table_2018 = data_for_hans_rosling_scatter_2018[["Region", "Country", "Life expectancy at birth, total (years)", "GDP per capita (current US$)"]].sort_values(by="Life expectancy at birth, total (years)").dropna()

#%%
life_expectancy_league_table_2018.head(10)

#%%
life_expectancy_league_table_2018.tail(10)


#%%
gdp_league_table_2018 = data_for_hans_rosling_scatter_2018[["Region", "Country", "Life expectancy at birth, total (years)", "GDP per capita (current US$)"]].sort_values(by="GDP per capita (current US$)").dropna()

#%%
gdp_league_table_2018.head(10)
#%%
gdp_league_table_2018.tail(10)


# %%
