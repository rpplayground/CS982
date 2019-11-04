#%% [markdown]
#University of Strathclyde - MSc Artificial Intelligence and Applications
#
#CS982 - Big Data Technologies
#
#File Created first created 9th October 2019 by Barry Smart.
# 
## Stage 9 - Application of Supervised Methods
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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#%% [markdown]
### Stage 6.1 - Read in the Data and Prepare it for Analysis
# Read in the file that was generated from the warngling and cleaning scripts.

#%%
#github_path = "C:/Users/Barry/"
github_path = "C:/Users/cgb19156/"
data_path = github_path + "GitHub/CS982/assignment1/"
interpolated_data_set = pd.read_pickle(data_path + "interpolated_data_set.pkl")
#%% [markdown]
# Adding "Log GDP" in order to generate more meaningful analysis.
interpolated_data_set["Log GDP per Capita"] = np.log10(interpolated_data_set["GDP per capita (current US$)"])

### Stage 6.2 - Inear Regression
# Linear regression models are popular because they can be fit very quickly and are very interpretable.
# Here I will apply linear regression to build a predictive model for Life Expectancy based on the 4 variables that strong correlatons with it.
#

#%%
# Set target column:
linear_regression_target_column = 'Life expectancy at birth, total (years)'

# Select the feature columns:
linear_regression_feature_columns = [ \
    #'Account ownership at a financial institution or with a mobile-money-service provider, young adults (% of population ages 15-24)',\
    #'Electric power consumption (kWh per capita)',\
    #'Energy use (kg of oil equivalent per capita)',\
    #'Fossil fuel energy consumption (% of total)',\
    'Immunization, DPT (% of children ages 12-23 months)',\
    #'Inflation, consumer prices (annual %)',\
    #'Market capitalization of listed domestic companies (current US$)',\
    #'Merchandise exports (current US$)',\
    #'Mobile cellular subscriptions (per 100 people)',\
    #'Mortality caused by road traffic injury (per 100,000 people)',\
    #'Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)',\
    #'Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)',\
    'Mortality rate, infant (per 1,000 live births)',\
    #'Population density (people per sq. km of land area)',\
    #'Population growth (annual %)', 'Population, total',\
    #'Power outages in firms in a typical month (number)',\
    'Renewable energy consumption (% of total final energy consumption)',\
    #'Suicide mortality rate (per 100,000 population)',\
    #'Tax revenue (% of GDP)', 'Urban population growth (annual %)',\
    'Log GDP per Capita']

#%%
# Helper function to extract cross section of data based on year - I've chosen here to grab data for a single year.
def grab_year(data_frame, year):
    single_year_data_frame = data_frame.xs(year, level="Year", drop_level=False)
    return single_year_data_frame

single_year_data_frame_1990 = grab_year(interpolated_data_set, 1990)
single_year_data_frame_1990

#%%
# Helper function to extract target and feature dataframes
def get_target_and_features(data_frame, target_column, feature_column_list):
    # Build a list of all columns
    all_columns = feature_column_list + [target_column]
    # Grab only those columns from a "flattened" dataset
    data_frame = data_frame.reset_index()
    data_frame = data_frame[all_columns]
    data_frame = data_frame.dropna()
    target_dataframe = data_frame[target_column]
    features_dataframe = data_frame[feature_column_list]
    return target_dataframe, features_dataframe

#%%
# Extract the target and feature dataframes from the source data using the helper function:
linear_regression_target_dataframe, linear_regression_features_dataframe = get_target_and_features(single_year_data_frame_1990, linear_regression_target_column, linear_regression_feature_columns)

#%%
linear_regression_target_dataframe

#%%
linear_regression_features_dataframe

#%%
# Split the data ready for training and testing the model
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(linear_regression_features_dataframe, linear_regression_target_dataframe, test_size=0.20)

# Train the model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, Y_train)

#%%
print("Intercept: ", linear_regression_model.intercept_)

#%%
pd.DataFrame(list(zip(linear_regression_features_dataframe.columns, linear_regression_model.coef_)), columns = ['Features', 'Coefficients'])

#%%
# Now test the model and determine how well it performs:
linear_regression_prediction = linear_regression_model.predict(X_test)
print(metrics.mean_squared_error(Y_test, linear_regression_prediction))

#%%
# Now apply the trained model to future years to see how it performs:
list_of_results = []
for year in range(1990, 2019):
    single_year_data_frame = grab_year(interpolated_data_set, year)
    target_dataframe, features_dataframe = get_target_and_features(single_year_data_frame, linear_regression_target_column, linear_regression_feature_columns)
    #print(features_dataframe.shape)
    predict = linear_regression_model.predict(features_dataframe)
    mean_squared_error = metrics.mean_squared_error(target_dataframe, predict)
    result = { "Year" : year, "Mean Squared Error" : mean_squared_error}
    list_of_results = list_of_results + [result]
results_dataframe = pd.DataFrame(list_of_results)
results_dataframe


#%%
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-', })
plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 3,\
    'axes.labelsize' : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16})
f, ax = plt.subplots(figsize=(10, 10))
plt.title("Performance of Predictive Model for Life Expectancy\nTrained Using Data From 1990\nMean Squared Error When Applied to Years 1991 to 2018 ", fontdict = {"fontsize" : 20})
sns.lineplot(x="Year", y="Mean Squared Error",\
    linewidth=3,\
    data=results_dataframe, ax=ax)


### Stage 6.3 - Naive Bayes
# 
#%%
# Set target column:
nb_target_column = 'Income Group'

# Select the feature columns:
nb_feature_columns = [ \
    'Life expectancy at birth, total (years)',\
    'Immunization, DPT (% of children ages 12-23 months)',\
    'Mortality rate, infant (per 1,000 live births)',\
    'Renewable energy consumption (% of total final energy consumption)',\
    'Log GDP per Capita']

#%%
# Grab a single year
single_year_data_frame = grab_year(interpolated_data_set, 2000)

# Extract the target and feature dataframes from the source data using the helper function:
nb_target_dataframe, nb_features_dataframe = get_target_and_features(single_year_data_frame_1990, nb_target_column, nb_feature_columns)

nb_target_dataframe

#%%
# Now assign integer labels for the target classifications
nb_target_dataframe_encoded = LabelEncoder().fit_transform(nb_target_dataframe)
nb_target_dataframe_encoded

#%%
nb_features_dataframe

#%%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(nb_features_dataframe, nb_target_dataframe_encoded, test_size = 0.25)

#%%
naive_bayes_model = GaussianNB()

#%%
naive_bayes_model.fit(X_train, Y_train)

#%%
naive_bayes_prediction = naive_bayes_model.predict(X_test)
print("Naive Bayes model accuracy score: ", metrics.accuracy_score(Y_test,naive_bayes_prediction))

#%%
print(metrics.classification_report(Y_test, naive_bayes_prediction))

#%%
print(metrics.confusion_matrix(Y_test,naive_bayes_prediction))
