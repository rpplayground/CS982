#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Part 7 - Analyse The Data
# The purpose of this notebook is to do some basic cleaning of the data.
#
#### Design Decisions
# The analysis will be conducted across the specific areas:
#- Looking at the major factors that drive life expectancy;
#- Trying to understand the factors that drive population growth.
#- Using the above to project world population forward based on some economic assumptions to estimate what the maximum populaton will reach.
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

interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")

#%%
interpolated_data_set.head(10)

#%%
interpolated_data_set.columns
#%%
analysis_of_2018 = interpolated_data_set.loc[pd.IndexSlice[:,:,:,[2018]], :]\
    .loc[:,["GDP (current US$)", "Life expectancy at birth, total (years)",\
         "Population, total", "Energy use (kg of oil equivalent per capita)"]]

#%%
analysis_of_2018
#%%
analysis_of_2018["GDP per Capita"] = \
    analysis_of_2018["GDP (current US$)"] / analysis_of_2018["Population, total"]

#%%
analysis_of_2018.reset_index().plot.scatter(x="GDP per Capita", y="Life expectancy at birth, total (years)", logx=False, alpha=0.5)

#%%
analysis_of_2018.reset_index().plot.scatter(x="GDP per Capita", y="Life expectancy at birth, total (years)", logx=True, alpha=0.5)

#%%
analysis_of_2018["Log GDP per Capita"] = np.log(analysis_of_2018["GDP per Capita"])

#%%
flattened_dataframe = analysis_of_2018.reset_index().loc[:,["GDP per Capita", "Life expectancy at birth, total (years)", "Population, total", "Region", "Log GDP per Capita", "Income Group", "Energy use (kg of oil equivalent per capita)"]]

#%%
#flattened_dataframe["Region Category"] = flattened_dataframe["Region"].astype('category')

#%%
#flattened_dataframe["Region Code"] = flattened_dataframe["Region Category"].cat.codes
#%%
flattened_dataframe.dtypes
#%%
flattened_dataframe.describe()
#%%
sns.set(style="ticks")
sns.pairplot(flattened_dataframe, hue="Region")

#%%
sns.boxplot(x="Life expectancy at birth, total (years)", y="Region", data=flattened_dataframe)

#%%
sns.boxplot(x="Log GDP per Capita", y="Region", data=flattened_dataframe)

#%%
sns.boxplot(x="Life expectancy at birth, total (years)", y="Income Group", data=flattened_dataframe)
