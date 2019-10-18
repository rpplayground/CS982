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
interpolated_data_set

#%%
#interpolated_data_set = interpolated_data_set.fillna(0)

#%% [markdown]
#### Stage 6.3 - Visualise The Results Of Filling In Missing Data
# Generate some visualisations to show how the interpolation has worked.

#%%
# Using pd.IndexSlice to slice at "Country" level (ie level 2) for three interesting countries.
pivoted_worldbank_data.loc[pd.IndexSlice[:,:,['Afghanistan', 'Iran', 'Iraq', 'Dem. People\'s Rep. Korea']], :]\
    .loc[:, ["GDP per capita (current US$)"]].reset_index().groupby("Country").plot(x="Year")

#%%
# Now showing the same data for the interpolated data set!
interpolated_data_set.loc[pd.IndexSlice[:,:,['Afghanistan', 'Iran', 'Iraq', 'Dem. People\'s Rep. Korea']], :]\
    .loc[:, ["GDP per capita (current US$)"]].reset_index().groupby("Country").plot(x="Year")

#%%
# Using pd.IndexSlice to slice at "Country" level (ie level 2) for three interesting countries.
pivoted_worldbank_data.loc[pd.IndexSlice[:,:,['Afghanistan', 'Iran', 'Iraq', 'Dem. People\'s Rep. Korea']], :]\
    .loc[:, ["Electric power consumption (kWh per capita)"]].reset_index().groupby("Country").plot(x="Year")

#%%
# Now showing the same data for the interpolated data set!
interpolated_data_set.loc[pd.IndexSlice[:,:,['Afghanistan', 'Iran', 'Iraq', 'Dem. People\'s Rep. Korea']], :]\
    .loc[:, ["Electric power consumption (kWh per capita)"]].reset_index().groupby("Country").plot(x="Year")

#%% [markdown]
#### Stage 6.4 - Write To File
# Now we write the resulting data frame to the Pickle file format to preserve all meta data.

#%%
interpolated_data_set.to_pickle(data_path + "interpolated_data_set.pkl")
