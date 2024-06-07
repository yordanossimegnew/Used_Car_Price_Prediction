#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction Project 
# ##### By: Yordanos Simegnew

# #### 1. Importing the necessary libraries

# In[1]:


# importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)


# #### 2. Loading the Dataset

# In[2]:


# Loading the dataset
df = pd.read_csv("C:\\Users\\yozil\\Desktop\\My projects\\used car price prediction\\used_car_price_prediction\\data\\raw\\car data.csv")
df


# #### 3. First look at the Data

# In[3]:


# displaying 4 sample records
df.sample(4)


# In[4]:


# displaying the name of columns
df.columns


# In[5]:


# shape of the dataframe
df.shape


# In[6]:


# general information on the dataframe
df.info()


# #### 4. Data Cleaning

# ##### 4.1 Standardizing column names

# Here we perform two tasks
# 1. Standardizing the name of each column name such that it appears in the (title case format)
# 2. Adding units to columns that need units

# In[7]:


# 1. standardizing text columns
df.columns = df.columns.str.title()


# Adding units to columns that need units,
# Here the columns that need units are 
# 1. Selling price and present price have to be both interms of ($)

# In[8]:


df.rename(columns = {"Selling_Price":"Selling_Price($)", "Present_Price":"Present_Price($)"}, inplace = True)


# In[9]:


df.sample(3)


# ##### 4.2 Standardizing Text columns 

# In[10]:


# here we standardize all columns with text type of data to a title case format.
def text_maker(text):
    return text.str.title()


# In[11]:


# displaying the text columns of 4 sample records
df.select_dtypes("object").sample(4)


# In[12]:


# applying the text maker function specifically to the text columns and updating the original dataframe
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(text_maker)


# In[13]:


# overwritting the existing text columns with the new 
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(text_maker)


# In[14]:


# displaying sample records from our dataset
df.sample(3)


# ##### 4.3 Removing Unecessary Space from the text columns (if any)

# In[15]:


# let's define a function to remove unecessary space form the text columns
def space_remover(text):
    return text.str.strip()


# In[16]:


# now let's apply this function in our text columns and update our dataframe
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(space_remover)


# In[17]:


# displaying sample records from our dataset
df.sample(3)


# ##### 4.4 Removing Duplicated Records(if any)

# In[18]:


# let's check for the existance of any duplicated records
df.duplicated().any()


# In[19]:


# the above result shows we have a duplicated records, let\s see how many duplicated records we have
df.duplicated().sum()


# In[20]:


# as shown from the above result we have two duplicated records, let's see this duplicated records
df[df.duplicated()]


# In[21]:


# now let's remove this duplicated records, since they make our model to be biased.
df.drop_duplicates(inplace = True)


# In[22]:


df.shape


# In[23]:


df.info()


# In[24]:


# now let's fix our index column.
df.reset_index(inplace = True)


# In[25]:


df.sample(3)


# In[26]:


df.drop("index", axis = 1, inplace = True)


# In[27]:


df.index


# In[28]:


df.sample(3)


# In[29]:


df.info()


# ##### 4.5 Handling Missing Values (if any)

# In[30]:


# from the above result of general information we can see that we don't have any missing value in any columns but we can check it here again
df.isnull().any()


# In[31]:


df.isnull().sum()


# In[32]:


# so we don't have any missing values in the dataset


# ##### 4.6 Extracting infromation from a column

# In[33]:


# here we can extract the age of the car from the year column.


# In[34]:


# displaying 3 sample records
df.sample(3)


# In[35]:


# first let's see the exact time of today
today  = datetime.today()
today


# In[36]:


# now let's extract the year from the todays date
year = today.year
year


# In[37]:


# now let's extract the year column from our today year to get the age of each car
df.insert(df.columns.get_loc("Year")+1,"Age",(year - df.Year))


# In[38]:


df.sample(3)


# ##### 4.7 Handling an outliers

# In[39]:


# for the case of this project we consider all results out of 4 standard deviation from the mean as an outliers.
# now let's define a function that can show us the outlier limits for each numerical columns.
def outlier_limit(col):
    mean = col.mean()
    std = col.std()
    upper_limit = mean +  4 * std
    lower_limit = mean -  4 * std
    return upper_limit, lower_limit


# In[40]:


df.info()


# In[41]:


outlier_limit(df.select_dtypes(["int64","float64"]))


# In[42]:


df.select_dtypes(["int64","float64"]).apply(outlier_limit)


# In[43]:


# now let's see the records which violate our ouliter limits
def outliers(dff):
    outliers_df = pd.DataFrame()
    for col in dff.columns:
        mean = df[col].mean()
        std = df[col].std()
        upper_limit = mean + 4 * std
        lower_limit = mean - 4 * std
        outlier_data = dff[(df[col] > upper_limit) | (df[col] < lower_limit)]
        outliers_df = pd.concat([outliers_df,outlier_data]).drop_duplicates()
    return outliers_df


# In[44]:


# let's see the number of  outlier records
outliers(df.select_dtypes(["int64","float64"])).shape[1]


# In[45]:


# as we can see from the above result we have 6 outlier records, now let's see this records specifically
outliers(df.select_dtypes(["int64","float64"]))


# In[46]:


# for the case of this project we handle outliers by just removing the records, so let's remove this records
outliers_index = outliers(df.select_dtypes(["int64","float64"])).index
outliers_index


# In[47]:


# now we have to drop this index numbers from our original dataframe.
df.drop(outliers_index, inplace = True)
df


# In[48]:


df.shape


# In[49]:


df.index


# In[50]:


# now let's fix the index of the dataframe
df.reset_index(inplace = True)


# In[51]:


df.sample(3)


# In[52]:


df.drop("index",axis = 1, inplace = True)


# In[53]:


df.sample(3)


# In[54]:


### now we are done with the data cleaning process.


# In[55]:


### Now let's export our cleaned dataframe
df.to_csv("C:\\Users\\yozil\\Desktop\\My projects\\used car price prediction\\used_car_price_prediction\\data\\processed\\cleaned_car_data.csv")


# In[ ]:




