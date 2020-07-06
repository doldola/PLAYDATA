#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


#원본적용 implace = True


# In[4]:


#데이터프레임 인덱싱


# In[5]:


#df.columns 컬럼들 반환


# In[6]:


data = np.arange(20)
data


# In[7]:


data1 = data.reshape(5,4)


# In[8]:


df = pd.DataFrame(data1, columns=["year", "nmae", "points", "penalty"], index=(["one", "two", "three", "four", "five"]))
df[['year', 'points']]
df['zeros'] = np.arange(5)
df['wtf'] = df['year'] - df['points']
df["high points"] = df['wtf'] - df['penalty'] > 2.0


# In[9]:


df


# In[10]:


#index와 columns 이름 지정
df.index.name = 'order' #인덱스들을 포괄하는 이름을 주는것
df.columns.name = 'Info'#컬럼들을 포괄하는 이름을 주는것
df


# In[12]:


df.columns


# In[14]:


df.index


# In[21]:


df.year


# In[22]:


df['year']


# In[23]:


df


# In[25]:


df[0:3] #행 인덱싱


# In[28]:


df.loc[["one","three"]] 


# In[29]:


#loc 인덱싱 loc -> location의 약어
#iloc 인덱싱 정수기반 -> integer location
df.loc["one"] #한개만 불러올시 시리즈로 반환
#여러개 불러올시 데이터프레임으로 반환


# In[30]:


df.loc["one","year"]


# In[32]:


df.loc["one":"two","year":'points']


# In[33]:


df.loc[:,'year']


# In[34]:


df


# In[37]:


df.loc["six",:] = [11,2,1,2,2,2,'False']


# In[48]:


df.iloc[:,2:3]


# In[49]:


df.iloc[[0,1,3],[1,2]]


# In[51]:


df['year'] > 4


# In[53]:


df.loc[df['nmae'] == 2, ['nmae','points']]


# In[55]:


df.loc[(df['points']>2) & (df['points']<15),:]


# In[ ]:




