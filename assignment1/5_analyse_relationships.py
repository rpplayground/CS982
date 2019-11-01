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

#analysis_of_2018, list_of_columns = dwf.assign_short_variable_names(analysis_of_2018, 18)

analysis_of_2018_flattened = analysis_of_2018.reset_index()
analysis_of_2018_flattened.index.name = 'ID'
analysis_of_2018_flattened.head(10)

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

#%%
interpolated_data_set_short_column_titles.columns

#%%
# Now flatten and pair down the data set ready for a pair plot:
pruned_data = interpolated_data_set_short_column_titles.xs(1995, level="Year", drop_level=False).reset_index()
pruned_data

#%%
pruned_data = pruned_data[['Income Group',\
    'Life expectancy at..', 'Mortality rate, in..',\
        'Renewable energy c..', 'Log GDP per Capita']]
pruned_data = pruned_data.dropna()
pruned_data

#%%
pruned_data["Income Group"].value_counts()

#%%
from pandas.plotting import scatter_matrix
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '--', 
    'axes.titlesize' : 18, 'lines.linewidth' : 3, 'axes.labelsize' : 16, 'xtick.labelsize' : 16,\
    'ytick.labelsize' : 16})
chart = sns.pairplot(pruned_data, hue="Income Group", palette=income_palette)
chart.fig.suptitle("Analysis of Variables With Strongest Correlations For Year 2018", y=1.02, fontsize=20)
plt.savefig("pairplot.png", type="png")

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

# max_x = analysis_of_2018_flattened.loc[analysis_of_2018_flattened["GDP per capita (current US$)"].idxmax()]

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
sns.lmplot(x="GDP per capita (current US$)", y="Life expectancy at birth, total (years)",\
    height=8, data=analysis_of_2018_flattened)
#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
sns.lmplot(x="Log GDP per Capita", y="Life expectancy at birth, total (years)",\
    height=8, data=analysis_of_2018_flattened)

#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
sns.lmplot(x="Log GDP per Capita", y="Population growth (annual %)",\
    height=8, data=analysis_of_2018_flattened)


#%% [markdown]
### Interactive plot of the data

#%%
def get_data_for_year(dataframe, year):
    data_for_single_year = dataframe.xs(year, level="Year", drop_level=False)
    # Flatten and select only the columns I need
    data_for_single_year_flattened_and_trimmed = data_for_single_year.reset_index()[["Region", "Country", "Year", "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Population growth (annual %)"]]
    # Add log base 10 of population to get better distribution on plot
    data_for_single_year_flattened_and_trimmed["Log Population"] = np.log10(data_for_single_year_flattened_and_trimmed["Population, total"])
    return data_for_single_year_flattened_and_trimmed


#%%
def country_scatterplot(country_dataframe, x_column, y_column, hue_column, size_column, x_min, x_max, x_scale="linear"):
    # Set up the style and size of the plot
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    f, ax = plt.subplots(figsize=(10, 10))
    ax.set(xscale=x_scale)
    ax.set_xlim(x_min, x_max)
    # Now plot the data
    sns.scatterplot(x=x_column, y=y_column,\
        hue=hue_column,\
        size=size_column,\
        palette=region_palette,\
        #hue_order=region_ranking,\
        sizes=(50,5000), linewidth=1,\
        alpha=1,\
        data=country_dataframe, ax=ax)
    
    dwf.plot_points_of_interest(country_dataframe, [("China", "HP")], "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Population, total", "Country", ax)


#%%
test_dataframe = get_data_for_year(interpolated_data_set, 1965)
min_power, min_value, max_power, max_value = dwf.find_min_and_max(interpolated_data_set_flattened, "GDP per capita (current US$)")
country_scatterplot(test_dataframe, "GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Region", "Population, total", min_value, max_value, x_scale="log")

#%%
test_dataframe

# %%
