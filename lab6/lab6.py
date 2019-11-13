#%%
import pymongo
from pymongo import MongoClient
import urllib.parse

#%%
username = urllib.parse.quote_plus('cgb19156')
username

#%%
password = urllib.parse.quote_plus('eeZiebah5tae')
password

#%% [markdown]
# Create a `connection` object, which is an instance of a MongoDB. The parameter is the connection string, a full mongodb URI.

#%%
connection = MongoClient('mongodb://%s:%s@devweb2019.cis.strath.ac.uk' % (username, password))
# The `server_info()` function called on object `conn` returns information about the the MongoDB Server we're connected to.
connection.server_info()

#%%
# List the databases available on the server:
connection.list_database_names

#%% [markdown]
# Since the connection is now open, you can acccess any of the databases within that Mongo server.
# You can use dictionary-style access to acces the database called `kxb17114`
db = connection['cgb19156']

#%%
db

#%%
# Get/create a collection. Since `create` is set to True, a `create` command is sent and the Collection is created.
collection = pymongo.collection.Collection(db, "collection_for_lab6", create=True)
collection

#%%
# Import the datetime library.
import datetime
# Create a dictionary (key-values pairs) called `entry`.
entry = {"author" : "Barry Smart", "text" : "Mountain biking is fun!", "date" : datetime.datetime.utcnow(), "level" : 9 }
# Insert the newly created dictionary in the collection we have created in the previous step. Retrieve the ID of the inserted object via `inserted_id`
post_id = collection.insert_one(entry).inserted_id
post_id

#%%
# Import pprint, which is a useful library to print Python data structures in a pretty way, hence why the p in the name.
import pprint
# pprint the first document in the collection
pprint.pprint(collection.find_one())

#%%
# pprint the first document in the collection that has the key equivalent `_id` and its corresponding value equivalent to the value of the variable
# `post_id`, the ID of the previously inserted object.
pprint.pprint(collection.find_one({'_id': post_id}))

#%%
from random import randrange
title_list = [\
    "Knife",\
        "Past Tense",\
            "The Midnight Line",\
                "Go Back",\
                    "The Player Of Games",\
                        "The Affair",\
                            "The Invaders Plan",\
                                "Last Arguement of Kings",\
                                    "The Hunt For Red October",\
                                        "The Bridge"]

#%%
# Insert one at a time...
for author in ["Barry Smart", "Iain Banks", "Tom Clancy", "Joe Abecrombie", "Jo Nesbo"]:
    random_count = randrange(10)
    for i in range(0,random_count):
        entry = {"author" : author, "text" : title_list[i], "date" : datetime.datetime.utcnow(), "level" : randrange(100)}
        collection.insert_one(entry)

#%%
# Insert as a list of entries...
entry_list = []
for author in ["Barry Smart", "Iain Banks", "Tom Clancy", "Joe Abecrombie", "Jo Nesbo"]:
    random_count = randrange(10)
    for i in range(0,random_count):
        entry = {"author" : author, "text" : title_list[i], "date" : datetime.datetime.utcnow(), "level" : randrange(100)}
        entry_list.append(entry)
        pprint.pprint(entry_list)

#%%
entry_list

#%%
collection.insert_many(entry_list)

#%%
for entry in collection.find({"author": "Iain Banks"}):
    pprint.pprint(entry)

#%%
import re
pattern = re.compile('Barry')
for entry in collection.find({"author": pattern}):
    pprint.pprint(entry)

#%%
import re
pattern = re.compile('[Tt]he')
for entry in collection.find({"text": pattern}):
    pprint.pprint(entry)

#%%
list_of_items = collection.find({"text": pattern})
for entry in list_of_items:
    pprint.pprint(entry["_id"])

#%%
single_item = collection.find({"text": pattern})[3]

#%%
entry_id = single_item["_id"]

#%%
pprint.pprint(collection.find_one({'_id': entry_id}))

#%%
# The function `estimated_document_count` returns the count of all documents in the dummy collection.
collection.estimated_document_count()