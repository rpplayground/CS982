#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Part 6 - Clean Up Data
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
#### Stage 1 - Read The File
# Read in the file that was generated from the previous script.

#%%
#github_path = "C:/Users/Barry/"
github_path = "C:/Users/cgb19156/"

data_path = github_path + "GitHub/CS982/assignment1/"

pivoted_worldbank_data = pd.read_pickle(data_path + "pivoted_worldbank_data.pkl")

#%%
pivoted_worldbank_data.head(10)

#%%
pivoted_worldbank_data.shape

#%%
pivoted_worldbank_data = pivoted_worldbank_data.dropna()

#%% [markdown]
#### Stage 2 - Fill In The Missing Data
# Fillin the blanks both backwards and forwards using linear interpolation.

#%%
interpolated_data_set = pivoted_worldbank_data.groupby(level=2).apply(lambda group: group.interpolate(method='linear', limit_direction='both', limit=60))

#%% [markdown]
#### Stage 3 - Visualise The Results Of Filling In Missing Data
# Generate some visualisations to show how the interpolation has worked.

#%%
# Using pd.IndexSlice to slice at "Country Code" level (ie level 2) for three interesting countries.
pivoted_worldbank_data.loc[pd.IndexSlice[:,:,['AFG', 'IRN', 'IRQ', 'PRK']], :]\
    .loc[:, ["GDP (current US$)"]].reset_index().groupby("Country Code").plot(x="Year")

#%%
# Now showing the same data for the interpolated data set!
interpolated_data_set.loc[pd.IndexSlice[:,:,['AFG', 'IRN', 'IRQ', 'PRK']], :]\
    .loc[:, ["GDP (current US$)"]].reset_index().groupby("Country Code").plot(x="Year")

#%% [markdown]
#### Stage 4 - Write To File
# Now we write the resulting data frame to the Pickle file format to preserve all meta data.

#%%
interpolated_data_set.to_pickle(data_path + "interpolated_data_set.pkl")
