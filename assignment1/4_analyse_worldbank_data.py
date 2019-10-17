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
analysis_of_2018 = interpolated_data_set.loc[pd.IndexSlice[:,:,:,[2018]], :].loc[:,["GDP (current US$)", "Life expectancy at birth, total (years)", "Population, total"]]

#%%
analysis_of_2018
#%%
analysis_of_2018["GDP per Capita"] = analysis_of_2018["GDP (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018["GNI per Capita"] = analysis_of_2018["GNI, Atlas method (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018.plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", logx=False, alpha=0.5)

#%%
analysis_of_2018["Region Code"] = analysis_of_2018["Region"].cat.codes, 
#%%
analysis_of_2018.reset_index().plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", logx=False, alpha=0.5)

#%%
analysis_of_2018.reset_index().plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", c=analysis_of_2018["Region"].cat.codes, logx=False, alpha=0.5)

#%%
analysis_of_2018.reset_index()["Region"]
#%%
flattened_dataframe = analysis_of_2018.reset_index().loc[:,["GDP per Capita", "Life expectancy at birth, total (years)", "Population, total", "Region"]]

#%%
flattened_dataframe["Region Category"] = flattened_dataframe["Region"].astype('category')

#%%
flattened_dataframe["Region Code"] = flattened_dataframe["Region Category"].cat.codes
#%%
flattened_dataframe.dtypes
#%%
flattened_dataframe.describe()

#%%
flattened_dataframe.plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", c="Region Code", logx=False, alpha=0.5)

#%%
sns.set(style="ticks")
sns.pairplot(flattened_dataframe, hue="Region")