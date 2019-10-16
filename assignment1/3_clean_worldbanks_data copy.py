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
#### Stage 1 - Inspect Data
# First stage is to load the raw data from file.

#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import scipy

#%%
github_path = "C:/Users/Barry/"
# github_path = "C:/Users/cgb19156/"

data_path = github_path + "GitHub/CS982/assignment1/"

pivoted_worldbank_data = pd.read_pickle(data_path + "pivoted_worldbank_data.pkl")

#%%
pivoted_worldbank_data.head(10)

#%%
interpolated_data_set = pivoted_worldbank_data.groupby(level=2).apply(lambda group: group.interpolate(method='linear', limit_direction='both', limit=60))

#%%
interpolated_data_set

#%%
# Using pd.IndexSlice to slice at "Country Code" level (ie level 2).
pivoted_worldbank_data.loc[pd.IndexSlice[:,:,['AFG', 'IRN', 'IRQ']], :]\
    .loc[:, ["GDP (current US$)"]].reset_index().groupby("Country Code").plot(x="Year")

#%%
# Now showing the same data for the interpolated data set!
interpolated_data_set.loc[pd.IndexSlice[:,:,['AFG', 'IRN', 'IRQ']], :]\
    .loc[:, ["GDP (current US$)"]].reset_index().groupby("Country Code").plot(x="Year")


#%%
analysis_of_2018 = interpolated_data_set.loc[pd.IndexSlice[:,:,:,[2018]], :].loc[:,["GDP (current US$)", "GNI, Atlas method (current US$)", "Life expectancy at birth, total (years)", "Population, total"]]

#%%
analysis_of_2018
#%%
analysis_of_2018["GDP per Capita"] = analysis_of_2018["GDP (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018["GNI per Capita"] = analysis_of_2018["GNI, Atlas method (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018.plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", logx=False, alpha=0.5)

