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
### Stage 7.2 - Prepare Source Data for Analysis
# A few simple steps to prepare the data set for visualisation:
# - Add column : Log10 of GDP - This step was added retrospectively, because inspection of the data shows that we will be required to work with "Log GDP" in order to generate more meaningful analysis.
# - Flatten the data by resetting the index so that the index columns are available for plotting.
#

#%%
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])
interpolated_data_set_flattened = interpolated_data_set.reset_index()

#%%
mean_by_region_and_year = interpolated_data_set.groupby(level=["Region", "Year"]).mean().reset_index()
mean_by_region_and_year.index.name = 'ID'

#%%
mean_by_country_and_decade = interpolated_data_set.groupby(level=["Region", "Country", "Decade"]).mean().reset_index()
mean_by_country_and_decade.index.name = 'ID'

#%%
mean_by_country_and_decade[["Country", "Life expectancy at birth, total (years)"]].sort_values(by=["Country"])

#%% [markdown]
### Stage 7.3 - Create "Slice" Of Data For 2018
# In this section I will analyse data for 2018 in more depth with the purpose of generating some simple initial insights into the data.
#
#%%
# Use the xs function to grab a cross section of the data, using the power of the multi-level-index.
analysis_of_2018 = interpolated_data_set.xs(2018, level="Year", drop_level=False)
# Flatten the dataframe to open up all columns for access by the matplotlib and seaborn libraries.
analysis_of_2018_flattened = analysis_of_2018.reset_index()
analysis_of_2018_flattened.index.name = 'ID'
analysis_of_2018_flattened.shape

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
#### Life Expectancy 2018 - Analysing Distribution of Data for All Countries
#Now using box plots to look at distribution by country.

#%%
# Create a helper function for box plots of different columns by region.
def region_box_plot(data_frame, x_column, plot_title, x_scale="linear"):
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 1.5,\
        'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
    f, ax = plt.subplots(figsize=(10, 6))
    ax.set(xscale=x_scale)
    #plt.title(plot_title)
    box_plot = sns.boxplot(x=x_column, y="Region", order=region_ranking, data=analysis_of_2018_flattened, palette=region_palette, ax=ax)
    box_plot.axes.set_title(plot_title,fontsize=18)
    box_plot.set_xlabel(x_column,fontsize=14)
    box_plot.set_ylabel("Region",fontsize=14)
    box_plot.tick_params(labelsize=12)

#%%
region_box_plot(analysis_of_2018_flattened, "Life expectancy at birth, total (years)", "Distribution of Life Expectancy\nby Region in 2018")

#%% [markdown]
#### Notes
# There is a significant outlier for "Latin America & Caribbean" - it would be good to investigate this further.
# TODO - add a backlog item.
#
# The country is Haiti:
# (https://borgenproject.org/top-10-facts-about-life-expectancy-in-haiti/)[https://borgenproject.org/top-10-facts-about-life-expectancy-in-haiti/]
# 

#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Life expectancy at birth, total (years)"]]\
    .loc[analysis_of_2018_flattened["Region"] == "Latin America & Caribbean"]\
    .nsmallest(10, "Life expectancy at birth, total (years)")

#%%
# haiti_data = interpolated_data_set_flattened.loc[interpolated_data_set_flattened["Country"] == "Haiti"]
# sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--'})
# f, ax = plt.subplots(figsize=(10, 10))
# plt.title("Population of Haiti", fontdict = {"fontsize" : 20})
# sns.lineplot(x="Year", y="Population, total",\
#     color="gray", data=haiti_data, ax=ax)


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
def region_line_plot(data_frame, y_column, plot_title, y_scale="linear"):
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-', })
    plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
        'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
    f, ax = plt.subplots(figsize=(10, 10))
    ax.set(yscale=y_scale)
    plt.title(plot_title, fontdict = {"fontsize" : 20})
    sns.lineplot(x="Year", y=y_column,\
        hue="Region",\
        hue_order=region_ranking,\
        linewidth=3,\
        palette=region_palette,\
        data=data_frame, ax=ax)

#%%
region_line_plot(mean_by_region_and_year, "Life expectancy at birth, total (years)", "Development of Life Expectancy by Region\nby Year since 1960", y_scale="linear")


#%%
f, ax = plt.subplots(figsize=(10, 10))
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
    'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
plt.title("Development of Life Expectancy by Country\nby Decade since 1960", fontdict = {"fontsize" : 20})
sns.swarmplot(x="Decade", y="Life expectancy at birth, total (years)", hue="Region",\
    palette=region_palette, data=mean_by_country_and_decade)

#%% [markdown]
#### Conclusions - Development of Life Expectancy Over Time
#The following observations can be made from the data above:
# - The gap in life expectancy has closed (more than halved) between 6 of te 7 regions;
# - Meanwhile life expectancy for the Sub-Saharan Africa region has not improved at the same rate, mainly as a result of a plateau in the 1990s;
# - The net result is that the gap between those countries with the worst and best record for life expectancy has not closed appreciably since 1960.
#
#%% [markdown]
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
region_box_plot(analysis_of_2018_flattened, "GDP per capita (current US$)", "Distribution of Gross Domestic Product (GDP)\nby Region in 2018")

#%% [markdown]
#The data is quite skewed given that it is spread over 4 orders of magnitude.
#So lets try the same plot but use a logarithmic scale.
#

#%%
region_box_plot(analysis_of_2018_flattened, "GDP per capita (current US$)", "Distribution of Log10 of GDP\nby Region in 2018", x_scale="log")

#%% [markdown]
# It will be useful to create a new series of data in the dataframe that is the Log10 of GDP.
# We should go back earlier in the process to add a calculated column which is "log to base 10 of GDP"
# TODO - add a backlog item.
#
#### Development of GDP Over Time
#Lets now examine how the gap in GDP across different nations has developed over time.
#
#%%
region_line_plot(mean_by_region_and_year, "GDP per capita (current US$)",\
    "Development of GDP Per Capita by Region\nby Year Since 1960", y_scale="log")

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
#ax.set(yscale="log")
plt.title("Development of GDP by Country\nby Decade Since 1960", fontdict = {"fontsize" : 20})
sns.swarmplot(x="Decade", y="GDP per capita (current US$)", hue="Region",\
    palette=region_palette, data=mean_by_country_and_decade)

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
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total"]]\
    .nlargest(10, "Population, total")
#%% [markdown]
#### Population 2018 - Bottom 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population, total"]]\
    .nsmallest(10, "Population, total")

#%% [markdown]
#### Population Growth 2018 - Top 10 Countries
#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population growth (annual %)"]]\
    .nlargest(10, "Population growth (annual %)")

#%% [markdown]
#### Population Growth 2018 - Bottom 10 Countries

#%%#%%
region_box_plot(analysis_of_2018_flattened, "Population growth (annual %)", "Distribution of Population Growth\n by Region In 2018", x_scale="linear")


#%%
analysis_of_2018_flattened.loc[:,["Country", "Region", "Population growth (annual %)"]]\
    .nsmallest(10, "Population growth (annual %)")

#%%
analysis_of_2018_flattened.loc[analysis_of_2018_flattened["Population growth (annual %)"] < 0][["Region", "Country", "Population growth (annual %)"]]

#%%
#%%
puertorico_data = interpolated_data_set_flattened.loc[interpolated_data_set_flattened["Country"] == "Puerto Rico"]
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--'})
plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
    'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
f, ax = plt.subplots(figsize=(10, 3))
plt.title("Population of Puerto Rico Since 1960", fontdict = {"fontsize" : 20})
sns.lineplot(x="Year", y="Population, total",\
     color="gray", data=puertorico_data, ax=ax)


#%%
region_line_plot(mean_by_region_and_year, "Population, total", "Average Country Population by Region\nby Year Since 1960", y_scale="log")

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Population Growth by Country\nby Decade Since 1960", fontdict = {"fontsize" : 20})
sns.swarmplot(x="Decade", y="Population growth (annual %)", hue="Region",\
    palette=region_palette, data=mean_by_country_and_decade)



#%%
