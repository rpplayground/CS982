#%%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import string 
import math

#%%
song1 = open(".\lab8\song1.txt","r").read()

#%%
song2 = open(".\lab8\song2.txt","r").read()


# %%
def word_count(document):
    word_count_dictionary = dict()
    # This step removes puncutation
    document = "".join(l for l in document if l not in string.punctuation)
    # This step converts everything to lower case before creating a list of all words in doument
    words = document.lower().split()
    for word in words:
        if word in word_count_dictionary:
            word_count_dictionary[word] += 1
        else:
            word_count_dictionary[word] = 1
    result = pd.DataFrame.from_dict(word_count_dictionary, orient='index')
    result = result.rename(columns={0: "WordCount"})
    total_word_count = len(words)
    result["TF"] = result["WordCount"] / total_word_count
    return result

#%% [markdown]
# #Question 1 - Part 1
# Calculate the term frequency (tf) values of the words all and world in each song.


#%%
word_count(song1).loc[["all", "world"]]

#%%
word_count(song2).loc[["all", "world"]]

#%% [markdown]
## Question 1 - Part 2
# If a music-lover entered the query "all world" into a music lyric retrieval system, which document would be retrieved first and why?

# ANSWER:
# If a basic approach is being used:
# 1. The term frequency for "all" and "world" would be summed across all documents;
# 2. The document that returns the highest result would be selected - so in this case Song1.

## Question 2
# Inverse document frequency (IDF) is based on the following formula:
# log(N/nt), where
# N = total number of documents in the collection
# nt = number of documents that contain term t

#%%
raw_data = { "apple" : 10, "banana" : 50, "plum" : 25 }
for name in raw_data:
    total_number_of_documents = float(100)
    document_ratio = total_number_of_documents / raw_data[name]
    print("The document ratio {} is {}.".format(name, document_ratio))
    IDF = math.log(document_ratio, 10)
    print("The inverse document frequency for {} is {}.".format(name, IDF))


# %% [markdown]
# # Quesion 3
# Below is a table showing the term weights for 6 indexing terms in 2 documents and 1 query.
# Using the simple matching function, calculate the similarity of each document to the query.
# Which document should be retrieved first?

#%%
data_table = pd.read_csv('.\lab8\question3_data.csv')
data_table = data_table.set_index('document')

# %%
data_table

# %%
def similarity(dataframe, document, query):
    columns = dataframe.shape[1]
    sum = 0
    for column in range(0, columns):
        # Implementation of the simple similarity matching function:
        sum = sum + dataframe.loc[document][column] * dataframe.loc[query][column]
    return sum

#%%
similarity(data_table, "Document1", "Query")

#%%
similarity(data_table, "Document2", "Query")

# %% [markdown]
# # Quesion 4
# Below is a table showing the term weights for 6 indexing terms in 2 documents and 1 query.
# Using the simple matching function, calculate the similarity of each document to the query.

#%%
def cosine_similarity(dataframe, document, query):
    columns = dataframe.shape[1]
    numerator = 0
    sum_of_squares_document = 0
    sum_of_squares_query = 0
    for column in range(0, columns):
        # Implementation of the simple similarity matching function:
        numerator = numerator + dataframe.loc[document][column] * dataframe.loc[query][column]
        sum_of_squares_document = sum_of_squares_document + (dataframe.loc[document][column] ** 2)
        sum_of_squares_query = sum_of_squares_query + (dataframe.loc[query][column] ** 2)
    denominator = math.sqrt(sum_of_squares_document) * math.sqrt(sum_of_squares_query)
    cosine = numerator / denominator
    return cosine


# %%
cosine_similarity(data_table, "Document1", "Query")

# %%
cosine_similarity(data_table, "Document2", "Query")


# %%
