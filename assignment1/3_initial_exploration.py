#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
## Assignment 1 - Exploring Data
# File Created first created 9th October 2019 by Barry Smart.
# 
#%% [markdown]
### Stage 3 - Initial Exporation
# This stage is concerned with loading  the raw data and have an initial look at it to identify data wrangling steps.

#%%
# I am going to make use of the common numpy and pandas libraries to load, wrangle and perform basic analysis of the data
import numpy as np
import pandas as pd
import assignment1.data_wrangling_functions as dwf


#%% [markdown]
#### Stage 3.1 - Import The Data
# First step is to read the data from file and have a look at the first 10 rows...

#%%
# Read the raw World Bank data from the CSV file
github_path = "C:/Users/Barry/"
#github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
raw_worldbank_data = pd.read_csv(data_path + "world_bank_data.csv")
# Have a look at the first 10 rows
raw_worldbank_data.head(10)

#%% [markdown]
# The data is structured with repeating rows of data, where each row is governed by:
# - The "Series Name" (a "Series Code" is also provided) - an example being "Population, total"
# - The "Country Name" - (a "Country Code" is also provided) - an example being "Argentina"
# There are then columns for each of the years from 1960 to 2018.

#### Stage 3.2 - Closer Look At Shape and Structure : Series Name
# Next step is to take closer look at the shape and structure of the data.
# First run some basic statistics about number of rows and columns.
# In particular, analyse the "Series Name" column to determine how many discrete data series are available.

#%%
raw_worldbank_data.shape

#%%
raw_worldbank_data["Series Name"].value_counts().head(20)

#%%
raw_worldbank_data["Series Name"].value_counts().count()

#%%
raw_worldbank_data.tail(10)

#%% [markdown]
# Looking at this list of series, there are a couple of backlog items that need to be created.
#
# There are trailing rows at the end of the data set that need to be removed - added [item](https://trello.com/c/P6isPTdY) to backlog.
# 
# I also need to select the subset of "Series Name" categories that I want to work with.  There are too many!
#  On this basis, I have added a backlog [item](https://trello.com/c/bPYXSpNU) to filter to the following "Series":
# - Population, total
# - GDP (current US$)
# - GNI, Atlas method (current US$)
# - Electric power consumption (kWh per capita)
# - Energy use (kg of oil equivalent per capita)
# - Personal remittances, paid (current US$)
# - Start-up procedures to register a business (number)
# - Life expectancy at birth, total (years)
# - Inflation, consumer prices (annual %)
# - Fertility rate, total (births per woman)
# - Mobile cellular subscriptions (per 100 people)

#%% [markdown]
#### Stage 3.2 - Closer Look At Shape and Structure : Country Name
# Look at more depth at the country data to understand how many countries we have data for.

#%%
raw_worldbank_data["Country Name"].value_counts()

#%% [markdown]
#### Conclusions Regarding Shape of Data
# This structure is going to make life difficult for further analysis as we tend to have the variables organised in discrete columns.
#
# So added [item](https://trello.com/c/8w28nF2y) to the backlog is a need to re-structure this data such that:
# - Country Column - categorical - the contry to which the data is related;
# - Year Column - categorical - the year to which the data is related;
# - Discrete Columns for Each Series to be Explored - continuous - a set of numerical data for that country/year combination.  Where column title is the "Series "
#
# In preparation for the item above, the year column names that are currently in the form "MMMM [MMMM]" (where M is a numeric character) will need to be transformed into the format "MMMM".  I've created another backlog [item](https://trello.com/c/MLKQ1zJp) to do this.
#
#### Look At Distruibution Of Continuous Data
# Let's now take a deeper look at the GDP data for a single year.

#%%
raw_worldbank_data.loc[raw_worldbank_data["Series Name"] == "GDP (current US$)"]["1960 [YR1960]"].describe()

#%% [markdown]
# This is concerning - I was expecting to mean, standard deviation, max, min etc.
# But this looks like the data is not being treated as numerical but as categorical.  So there must be some cleaning up required!
# 
# Let's dig into this further by taking a "thin slice" of the data - in this case GDP data for 1960.

#%%
thin_slice = raw_worldbank_data.loc[raw_worldbank_data["Series Name"] == "GDP (current US$)"].loc[:, "Series Name" : "1960 [YR1960]"]

#%%
thin_slice.head(10)

#%%
thin_slice.iloc[1,2]

#%%
thin_slice.dtypes

#%% [markdown]
# So the presence of a double full stop ".." indicates that there is no data available.  It's clear that whilst population data is available for all countries since 1960, some indicators such as GDP are not available.
# We will need to replace this with an empty cell in order to process the data.
# Let's do that quickly for this "thin slice" to see if it has the desired impact by replacing the ".." with NA (empty cell) AND setting the 1960 column to be a floating number rather than "object".

#%%
thin_slice = thin_slice.replace(to_replace='..', value=np.nan)

#%%
thin_slice = thin_slice.astype({"1960 [YR1960]" : "float"})

#%%
thin_slice.dtypes

#%%
thin_slice["1960 [YR1960]"].describe()

#%% [markdown]
# Those steps above seem to work for the "thin slice" - that approach just needs to scaled up to the full data set.  So I've added two further items to the backlog:
# - First backlog [item](https://trello.com/c/Q4u53d2D) is to replace ".." with empty cells;
# - Second backlog [item](https://trello.com/c/nQjCChjm) is to set the data types of the continuous data columns to "float".
#
#### Further Thoughts
# Some further throughts also added to backlog:
#
# 1. Addition of country meta data - backlog [item](https://trello.com/c/gQOicDsM) - eg categorical data for region / continent to allow high order analysis
# 2. Integration of other data sets - backlog [item](https://trello.com/c/EG2lMu3x) - eg from the WHO?
# 
