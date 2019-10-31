#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 6 - Application of Supervised Methods
#The purpose of this notebook is to run a range of supervised models on the data.
#The objective is to determine how successfully clustering models can be used to label given the *ground truth* of *Income GrouP*.
#
# The following high level steps have been applied:
# 1. X
# 2. Y
#
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression

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
analysis_of_decade = interpolated_data_set.xs("1990s", level="Decade", drop_level=False)
analysis_of_decade = analysis_of_decade.groupby(level=["Region", "Income Group", "Country", "Decade"]).\
    mean().reset_index()

#%%
analysis_of_decade

#%%
target_column = 'Life expectancy at birth, total (years)'

feature_columns = [ \
    #'Account ownership at a financial institution or with a mobile-money-service provider, young adults (% of population ages 15-24)',\
    #'Electric power consumption (kWh per capita)',\
    #'Energy use (kg of oil equivalent per capita)',\
    #'Fossil fuel energy consumption (% of total)',\
    'Immunization, DPT (% of children ages 12-23 months)',\
    'Inflation, consumer prices (annual %)',\
    #'Market capitalization of listed domestic companies (current US$)',\
    'Merchandise exports (current US$)',\
    'Mobile cellular subscriptions (per 100 people)',\
    'Mortality caused by road traffic injury (per 100,000 people)',\
    #'Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)',\
    #'Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)',\
    'Mortality rate, infant (per 1,000 live births)',\
    'Population density (people per sq. km of land area)',\
    'Population growth (annual %)', 'Population, total',\
    #'Power outages in firms in a typical month (number)',\
    'Renewable energy consumption (% of total final energy consumption)',\
    'Suicide mortality rate (per 100,000 population)',\
    #'Tax revenue (% of GDP)', 'Urban population growth (annual %)',\
    'Log GDP per Capita']

#%%
# Helper function to extract cross section of data based on year
def grab_year(data_frame, year):
    single_year_data_frame = interpolated_data_set.xs(year, level="Year", drop_level=False)
    return single_year_data_frame

single_year_data_frame = grab_year(interpolated_data_set, 2015)
single_year_data_frame

#%%
# Helper function to extract target and feature dataframes
def get_target_and_features(data_frame, target_column, feature_column_list):
    # Build a list of all columns
    all_columns = feature_column_list + [target_column]
    print(all_columns)
    # Grab only those columns from a "flattened" dataset
    data_frame = data_frame.reset_index()
    data_frame = data_frame[all_columns]
    data_frame = data_frame.dropna()
    target_dataframe = data_frame[target_column]
    features_dataframe = data_frame[feature_columns]
    return target_dataframe, features_dataframe

#%%
target_dataframe, features_dataframe = get_target_and_features(single_year_data_frame, target_column, feature_columns)

#%%
target_dataframe

#%%
features_dataframe

#%% [markdown]
### Stage 6.2 - Prepapre The Model
# Read in the file that was generated from the warngling and cleaning scripts.

#%%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features_dataframe, target_dataframe, test_size=0.25)
lm = LinearRegression()
lm.fit(X_train, Y_train)

#%%
print(lm.intercept_)

#%%
print(lm.coef_)

#%%
pd.DataFrame(list(zip(features_dataframe.columns, lm.coef_)), columns = ['Features', 'Coefficients'])

#%%
predict = lm.predict(X_test)

#%%
print(metrics.mean_squared_error(Y_test, predict))

