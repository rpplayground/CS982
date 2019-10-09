#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Part 1 - Choice Of Dataset and Objectives
# I've chosen to load country level data from the World Bank to explore world development indicators such as:
# - Economic indicators such as Gross Development Product (GDP)
# - Population and mortallity rates
# - Location in the world
#%% [markdown]
### Part 2 - Sourcing The Data
# Instructions about how I sourced the data.
# Notes about licensing.
#%% [markdown]
### Part 3 - First Look At The Data
# First stage is to load the law data and have an initial look at it.

#%%
import numpy as np
import pandas as pd
raw_worldbank_data = pd.read_csv("C:/Users/Barry/Documents/GitHub/CS982/assignment1/world_bank_data.csv.csv")
#%%
raw_worldbank_data.head(10)
#%%
raw_worldbank_data.shape
#%%
raw_worldbank_data.columns
#%% [markdown]
raw_worldbank_data["Series Name"].value_counts()
#%% [markdown]
raw_worldbank_data["Country Name"].value_counts()
#%%
raw_worldbank_data.loc[raw_worldbank_data["Series Name"] == "GDP (current US$)"]["1960 [YR1960]"].describe()

#%%
raw_GDP_data = raw_worldbank_data.loc[raw_worldbank_data["Series Name"] == "GDP (current US$)"]


#%%
raw_GDP_data

#%%
raw_GDP_data.iloc[1,4]

#%%
GDP_data = raw_GDP_data.replace(to_replace='..', value=np.nan)

#%%
GDP_data.describe()

#%%
GDP_data.dtypes

#%%
GDP_data.info

#%%
GDP_data.columns

#%%
