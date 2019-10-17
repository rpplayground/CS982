#%%
analysis_of_2018 = interpolated_data_set.loc[pd.IndexSlice[:,:,:,[2018]], :].loc[:,["GDP (current US$)", "GNI, Atlas method (current US$)", "Life expectancy at birth, total (years)", "Population, total"]]

#%%
analysis_of_2018
#%%
analysis_of_2018["GDP per Capita"] = analysis_of_2018["GDP (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018["GNI per Capita"] = analysis_of_2018["GNI, Atlas method (current US$)"] / analysis_of_2018["Population, total"]
#%%
analysis_of_2018.plot.scatter(x="GNI per Capita", y="Life expectancy at birth, total (years)", logx=False, alpha=0.5)