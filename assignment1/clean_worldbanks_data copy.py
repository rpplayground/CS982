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
import scipy

#%%
reshaped_worldbank_data = pd.read_pickle("./assignment1/reshaped_worldbank_data.pkl")

#%%
reshaped_worldbank_data.columns
#%%
reshaped_worldbank_data.head(10)

#%%
interpolated_data_set = reshaped_worldbank_data.groupby(level=0).apply(lambda group: group.interpolate(method='linear', limit_direction='both', limit=60))

#%% 
reshaped_worldbank_data.loc[['AFG', 'IRN', 'IRQ']].loc[:, ["GDP (current US$)"]].unstack(level=0).plot()
#%% 
interpolated_data_set.loc[['AFG', 'IRN', 'IRQ']].loc[:, ["GDP (current US$)"]].unstack(level=0).plot()

#%%
analysis_of_2018 = interpolated_data_set.xs('2018', level=1).loc[:,["GDP (current US$)", "Life expectancy at birth, total (years)", "Population, total"]]

#%%
analysis_of_2018["GDP per Capita"] = analysis_of_2018["GDP (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018.plot.scatter(x="GDP per Capita", y="Life expectancy at birth, total (years)", logx=True)


#%%
analysis_of_2018

#%%
