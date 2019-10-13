#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
### Part 5 - Wrangle Data
# The purpose of this notebook is to orchestrate the process of wanrgling the data.
# This has been tackled by completing each of the backlog items classed as "Data Wrangling" and placing them into a logica sequence in the notebook:
# 
# 1. Ingestion
# 2. Trimming and Filtering
# 3. Process Blank Cells
# 4. Unpivoting and Pivoting
# 5. Write To File
# 
#### Design Decisions
# - The notebook is structured such that it can be run end to end any time a new data cut becomes available to transform the raw data into a format for downstream analysis.
# - Any heavy blocks of code have been written as functions in a seperate "data_wrangling_functions.py" Python file and imported into this notebook.
#
#### Stage 1 - Ingestion
# First stage is to load the raw data from file.

#%%
import numpy as np
import pandas as pd
import assignment1.data_wrangling_functions as dwf

#%%
raw_worldbank_data = pd.read_csv("C:/Users/Barry/Documents/GitHub/CS982/assignment1/world_bank_data.csv.csv")

#%% [markdown]
#### Stage 2 - Trimming and Filtering
# This part of the process will:
# - Remove the last 5 rows from the data set as they do not contain data;
# - Filter down to the set of "Series Name" that I am interested in analysing.
#
#%%
raw_worldbank_data.tail(10)
#%%
trimmed_worldbank_data = raw_worldbank_data.head(-5)
#%%
trimmed_worldbank_data.tail(10)
#%%
list_of_series_names = ["Population, total", "GDP (current US$)", "GNI, Atlas method (current US$)", "Electric power consumption (kWh per capita)", \
    "Energy use (kg of oil equivalent per capita)", "Personal remittances, paid (current US$)", "Start-up procedures to register a business (number)", \
        "Life expectancy at birth, total (years)", "Inflation, consumer prices (annual %)", "Fertility rate, total (births per woman)", "Mobile cellular subscriptions (per 100 people)"]

filtered_worldbank_data = trimmed_worldbank_data.loc[trimmed_worldbank_data['Series Name'].isin(list_of_series_names)]
#%%
filtered_worldbank_data["Series Name"].value_counts()
#%% [markdown]
#### Stage 3 - Process Blank Cells
# One simple step is required at this stage to clean up the cells that to contain no data : that is to replace the instances of ".." with NaN.

#%%
cleansed_world_data = filtered_worldbank_data.replace(to_replace='..', value=np.nan)

#%% [markdown]
#### Stage 4 - Unpivoting and Pivoting
# This part of the process will:
# - Rename the year columns - a pre-quisite to support the next step;
# - Unpivot the data such that the individual year columns are collapsed into single column to achieve a "thin and tall" data structure;
# - Pivot the data such that the individual series data are each placed into their own columns to achieve a "fatter and less tall" data stucture.
# 
#%%
reshaped_worldbank_data, column_list = dwf.trim_year_column_names(cleansed_world_data)
#%%
reshaped_worldbank_data.shape
#%%
reshaped_worldbank_data = pd.melt(reshaped_worldbank_data, id_vars=['Series Name', 'Series Code', 'Country Name', 'Country Code'])
#%%
reshaped_worldbank_data = reshaped_worldbank_data.rename(columns = { "variable" : "Year"})
#%%
reshaped_worldbank_data = reshaped_worldbank_data.astype({"value" : "float"})

#%%
reshaped_worldbank_data.shape
#%%
reshaped_worldbank_data.head(5)
#%%
reshaped_worldbank_data = pd.pivot_table(reshaped_worldbank_data, index=["Country Code", "Year"], columns="Series Name", values="value")
#%%
#reshaped_worldbank_data = reshaped_worldbank_data.reset_index()
#%%
reshaped_worldbank_data.shape
#%%
reshaped_worldbank_data.describe()

#%% [markdown]
#### Stage 5 - Write To File
# Now we write the resulting data frame to the Pickle file format to preserve all meta data.

reshaped_worldbank_data.to_pickle("./assignment1/reshaped_worldbank_data.pkl")


#%%
