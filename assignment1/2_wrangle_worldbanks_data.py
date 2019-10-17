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
# 4. Unpivoting
# 5. Add Country Netadata
# 6. Pivoting
# 7. Write To File
# 
#### Design Decisions
# - The notebook is structured such that it can be run end to end any time a new data cut becomes available to transform the raw data into a format for downstream analysis.
# - Any heavy blocks of code have been written as functions in a seperate "data_wrangling_functions.py" Python file and imported into this notebook.
#
#### Stage 1 - Ingest Data
# First stage is to load the raw data from CSV file as generated from the World Bank's [open data portal](https://databank.worldbank.org/home.aspx).
# Two files are of interest:
# - The larger file containing "world development indicators" organised by country and year.
# - A smaller "coutry metadata" file containing classifications for each coutry such as region and income group.
#

#%%
import numpy as np
import pandas as pd
import assignment1.data_wrangling_functions as dwf

#%%
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"

data_path = github_path + "GitHub/CS982/assignment1/"

raw_worldbank_data = pd.read_csv(data_path + "world_bank_data.csv")
#%%
raw_worldbank_country_metadata = pd.read_csv(data_path + "world_bank_country_metadata.csv")
#%%
raw_worldbank_country_metadata.loc[raw_worldbank_country_metadata["Code"] == "PRK"]

#%% [markdown]
#### Stage 2 - Trimming and Filtering
# This part of the process will:
# - Trim the last 5 rows from the data set as they do not contain data (could be considered redundant given next step below);
# - Filter down to the set of "Series Name" that I am interested in analysing.
#
#%%
raw_worldbank_data.tail(10)
#%%
trimmed_worldbank_data = raw_worldbank_data.head(-5)
#%%
trimmed_worldbank_data.tail(10)

#%% [markdown]
# So here the data components that are likely to have the biggest impact on the two areas need to be considered:
####Life Expectency:
# 1. Measures of economic properity:
# 2. Access to energy;
# 3. Environment measurements;
# 4. Access to education and technology;


#%%
list_of_series_names = ["Life expectancy at birth, total (years)",\
    # Measures of economic prosperity
    "GDP per capita (current US$)",\
    "Inflation, consumer prices (annual %)",\
    "Market capitalization of listed domestic companies (current US$)",\
    "Tax revenue (% of GDP)",\
    "Merchandise exports (current US$)",\
    # Access to clean, reliable energy
    "Electric power consumption (kWh per capita)",\
    "Energy use (kg of oil equivalent per capita)",\
    "Power outages in firms in a typical month (number)",\
    # Environmental measures
    "Population density (people per sq. km of land area)",\
    "Urban population growth (annual %)",\
    "Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)",\
    "Mortality caused by road traffic injury (per 100,000 people)",\
    "Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)",\
    "Urban population growth (annual %)",\
    # Access to education, healthcare and technology
    "Immunization, DPT (% of children ages 12-23 months)",\
    "Mobile cellular subscriptions (per 100 people)",\
    "Account ownership at a financial institution or with a mobile-money-service provider, young adults (% of population ages 15-24)"]

filtered_worldbank_data = trimmed_worldbank_data.loc[trimmed_worldbank_data['Series Name'].isin(list_of_series_names)]
#%%
filtered_worldbank_data["Series Name"].value_counts()
#%% [markdown]
#### Stage 3 - Process Blank Cells
# One simple step is required at this stage to clean up the cells that to contain no data : that is to replace the instances of ".." with NaN.

#%%
cleansed_world_data = filtered_worldbank_data.replace(to_replace='..', value=np.nan)

#%% [markdown]
#### Stage 4 - Unpivoting
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


#%% [markdown]
#### Stage 5 - Add Country Metadata
#
#%%
raw_worldbank_country_metadata.columns
#%%
country_metadata = raw_worldbank_country_metadata[["Code", "Short Name", "Long Name", "Income Group", "Region"]]
#%%
country_metadata.head(10)
#%%
merged_data = reshaped_worldbank_data.merge(country_metadata, left_on="Country Code", right_on="Code")

#%%
merged_data["Decade"] = merged_data["Year"].str.slice(start=0, stop=3) + "0s"

#%%
merged_data = merged_data.astype({"Year" : "int"})


#%%
merged_data.head(10)

#%%
merged_data.dtypes

#%% [markdown]
#### Stage 6 - Pivoting
# This part of the process will:
# - Pivot the data such that the individual series data are each placed into their own columns to achieve a "fatter and less tall" data stucture.
#
#%%
pivoted_worldbank_data = pd.pivot_table(merged_data, index=["Region", "Income Group", "Country Code", "Decade", "Year"], columns="Series Name", values="value")

#%%
pivoted_worldbank_data.shape
#%%
pivoted_worldbank_data.describe()
#%%
pivoted_worldbank_data.head(100)

#%%
pivoted_worldbank_data.dtypes



#%% [markdown]
#### Stage 7 - Write To File
# Now we write the resulting data frame to the Pickle file format to preserve all meta data.

#%%
pivoted_worldbank_data.to_pickle(data_path + "pivoted_worldbank_data.pkl")

