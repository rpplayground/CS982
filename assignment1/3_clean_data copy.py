#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Stage 6 - Clean Up Data
# The purpose of this notebook is to do some basic cleaning of the data.
#
#### Design Decisions
# - Linear extrapolation
#

#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#%% [markdown]
#### Stage 6.1 - Read The File
# Read in the file that was generated from the previous script.

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
pivoted_worldbank_data = pd.read_pickle(data_path + "pivoted_worldbank_data.pkl")

#%%
pivoted_worldbank_data.shape

#%%
pivoted_worldbank_data

#%% [markdown]
#### Stage 6.2 - Fill In The Missing Data
# Fillin the blanks *only forwards* using linear interpolation.

#%%
interpolated_data_set = pivoted_worldbank_data.groupby(level="Country").apply(lambda group: group.interpolate(method='linear', limit_direction='forward', limit=60))

#%%
pivoted_worldbank_data.isnull().sum(axis = 0) - interpolated_data_set.isnull().sum(axis = 0)

#%%
#interpolated_data_set = interpolated_data_set.fillna(0)

#%% [markdown]
#### Stage 6.3 - Visualise The Results Of Filling In Missing Data
# Generate some visualisations to show how the interpolation has worked.

#%%
# This funtion places each sub plot onto the frame
def create_sub_plot(figure, axes, x_position, y_position, data_series, x_column, y_column, title):
    #sns.lineplot(x="Year", y=y_column, data=data_series, ax=axes[x_position,y_position])
    axes[x_position,y_position].plot(data_series[y_column], color="black")
    axes[x_position,y_position].set_title(title)

# This function configures the frame and then iterates through the results to create slices of the data that can be plotted
def plot_interpolation_results(list_of_dataframes, list_of_countries, column_to_plot, share_x=True):
    country_dimension = len(list_of_countries)
    dataframe_dimension = len(list_of_dataframes)
    # Set up the number of sub plots based on the dimensions of experiments and points
    figure_size_unit = 8 # This governs the size of each subplot on the figure 
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
         'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
    figure, axes = plt.subplots(country_dimension, 2, sharex=share_x, figsize=(figure_size_unit * dataframe_dimension, figure_size_unit * country_dimension))
    for country_index, country in enumerate(list_of_countries):
        for dataframe_index, dataframe in enumerate(list_of_dataframes):
            # Use the "cross section" method to grab the results for a specific experiment and points configuration
            dataframe_to_plot = dataframe.loc[pd.IndexSlice[:,:,[country]], :].loc[:, [column_to_plot]].\
                reset_index(level=["Region", "Income Group", "Country", "latitude", "longitude", "Decade"])
            # Set up the title for the plot
            if dataframe_index == 0:
                title = column_to_plot + "\nData For " + country + " Before Interpolation"
            else:
                title = column_to_plot + "\nData for " + country + " After Interpolation"
            # Send the data off to get plotted
            create_sub_plot(figure, axes, country_index, dataframe_index , dataframe_to_plot, "Year", column_to_plot, title)
    plt.subplots_adjust(hspace=.4)

#%%
plot_interpolation_results([pivoted_worldbank_data, interpolated_data_set], ['Afghanistan', 'Iran', 'Iraq'], "GDP per capita (current US$)", share_x=False)

#%%
plot_interpolation_results([pivoted_worldbank_data, interpolated_data_set], ['Dem. People\'s Rep. Korea', 'Australia', 'Iraq'], "Electric power consumption (kWh per capita)", share_x=True)

#%% [markdown]
#### Stage 6.4 - Check Countries
# Now check if there are any countries in the data set that need to be removed.
#%%
mean_by_country = interpolated_data_set.groupby(level=["Region", "Country"]).mean()
list_of_regions = set(mean_by_country.index.get_level_values("Region"))

#%%
analysis_of_2018 = interpolated_data_set.xs(2018, level="Year", drop_level=False).reset_index(level=["Income Group", "latitude", "longitude", "Decade", "Year"])
analysis_of_2018.style

#%% [markdown]
#### Stage 6.4 - Write To File
# Now we write the resulting data frame to the Pickle file format to preserve all meta data.

#%%
interpolated_data_set.to_pickle(data_path + "interpolated_data_set.pkl")
