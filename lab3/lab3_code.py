#%% [markdown]
#### 1. Import the necessary libraries (numpy, pandas, matplotlib and seaborn)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% [markdown]
#### 2. Import the Titanic dataset available at https://www.kaggle.com/c/titanic/data (it is train.csv that you want)


#%%
titanic_data = pd.read_csv("lab3\\train.csv")

#%%
titanic_data.head(10)

#%%
titanic_data.dtypes

#%% [markdown]
#### 3. Produce some box plots for numeric values in the dataset

#%%
titanic_data.plot(kind='box', subplots=True, layout=(3,3))

#%% [markdown]
#### 4. Plot a bar chart showing the number of survivors and fatalities. Include a title on the chart

#%%
titanic_data['Survived'].value_counts()
#%%
titanic_data['Survived'].value_counts().plot.bar()
plt.ylabel('Count')
plt.title('Survivor Counts')
plt.show()

#%% [markdown]
#### 5. Produce a horizontal bar chart showing all passenger classes, ordered with smallest number at top and largest at bottom

#%%
titanic_data['Pclass'].value_counts().sort_values(ascending=True)

#%%
titanic_data['Pclass'].value_counts().sort_values(ascending=True).plot.barh()
plt.title('Count Of Passengers By Class')
plt.show()

#%% [markdown]
#### 6. Produce a density plot for number of siblings (SibSp)

#%%
titanic_data['SibSp'].plot.density()
plt.show()

#%% [markdown]
#### 7. Produce a stacked bar chart showing:the number of each gender in each passenger class

#%%
titanic_data.groupby(['Pclass','Sex']).size().unstack().plot(kind='bar',stacked=True)
plt.title("Count of Passengers By Class and Sex")
plt.show()

#%% [markdown]
#### 8. Produce a heatmap showing the correlation between each numerical variable. What shows a strong correlation?

#%%
plt.figure(figsize=(20,15))
corr = titanic_data.corr()
sns.heatmap(corr)

#%% [markdown]
#### 9. Produce a single scatter plot showing:
# - age and passenger class
# - as well as age and number of siblings
# Different symbols should be used to represent the two different comparisons.

#%%
plt.scatter(titanic_data['Age'], titanic_data['Pclass'], marker = "^", alpha=0.5)
plt.scatter(titanic_data['Age'], titanic_data['SibSp'], marker = "s", alpha=0.5)
plt.show()

#%% [markdown]
#### 10. Play around with different visualisations of this dataset

#%% [markdown]
#### 11. Using your interesting dataset from last week explore the dataset starting with some techniques that we explored in the lecture and this lab
