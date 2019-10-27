#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 27th October 2019 by Barry Smart.
# 
## Stage 8 - Geographical Visualisation
#The purpose of this notebook is to create some geogrpahical visuals of the data.
#
# The following high level steps have been applied:
#
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#%%
region_ranking = ["North America", "Europe & Central Asia", "Middle East & North Africa",\
    "Latin America & Caribbean", "East Asia & Pacific", "South Asia", "Sub-Saharan Africa"]
region_palette = {"North America" : "red", "Europe & Central Asia" : "blue",\
    "Middle East & North Africa" : "pink", "Latin America & Caribbean" : "purple",\
        "East Asia & Pacific" : "green", "South Asia" : "orange", "Sub-Saharan Africa" : "gray"}


#%% [markdown]
### Stage 6.1 - Read in the Data and Prepare it for Analysis
# Read in the file that was generated from the warngling and cleaning scripts.

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")
#%% [markdown]
# This step was added retrospectively, because inspection of the data shows that we will be required to work with "Log GDP" in order to generate more meaningful analysis.

#%%
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])
interpolated_data_set_flattened = interpolated_data_set.reset_index()

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
interpolated_data_set_flattened.columns

#%%
interpolated_data_set_flattened.head(10)



# %%
