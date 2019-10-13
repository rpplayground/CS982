#%% [markdown]
# 1. Import the necessary libraries (numpy and pandas)

#%%
import numpy as np
import pandas as pd

#%% [markdown]
# 2. Import the Pokemon dataset available at https://www.kaggle.com/alopez247/pokemon  
#%%
pokemon_data = pd.read_csv("C:/Users/cgb19156/github/CS982/Lab 2/pokemon_alopez247.csv")

#%% [markdown]
# 3. Print the first 10 entries 
#%%
pokemon_data.head(10)

#%% [markdown]
# 4. How many observations and columns are there?
#%%
pokemon_data.shape

#%% [markdown]
# 5. Print the names of all of the columns
#%%
pokemon_data.columns

#%% [markdown]
# 6. Sort by attack power from high to low
#%%
pokemon_data["Attack"].sort_values(ascending = False)

#%% 
# The same thing can be achieved as follows:
pokemon_data.Attack.sort_values(ascending = False)
 
#%% [markdown]
# 7. Describe the defence power for each Pokemon type
#%%
pokemon_grouped_by_type1 = pokemon_data.groupby("Type_1")

#%%
pokemon_data.Defense.describe()

#%%
pokemon_grouped_by_type1

#%%
pokemon_grouped_by_type1.Defense.describe().sort_values(by=["mean"], ascending=False)

#%% [markdown]
# 8. What are the mean, median, max and minimum of the total column for each Pokemon type? 
#%%
pokemon_grouped_by_type1.Total.mean().sort_values(ascending=False)

#%%
pokemon_grouped_by_type1.Total.median().sort_values(ascending=False)

#%%
pokemon_grouped_by_type1.Total.max().sort_values(ascending=False)

#%%
pokemon_grouped_by_type1.Total.min().sort_values(ascending=False)
 
#%% [markdown]
# 9. What is the most common Pokemon type? 
#%%
pokemon_grouped_by_type1.Name.count().sort_values(ascending=False)

#%% [markdown]
# So "Water" is the most common type of pokomon.

#%% [markdown]
# 10. Find an interesting dataset online and download it for analysis. Good places to look include:
#
# i. Kaggle - https://www.kaggle.com/datasets
#
# ii. UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/datasets.html
#
# Explore the dataset starting with some techniques that we explored in the lecture and this lab 