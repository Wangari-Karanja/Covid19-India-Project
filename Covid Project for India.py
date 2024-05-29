#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as datetime


# In[3]:


covid_df = pd.read_csv(r'C:\Users\HP\Documents\Simplilearn projects\covid_19_india.csv')
covid_df.head(3)


# In[4]:


covid_df.info()


# In[5]:


covid_df.shape


# In[6]:


print(covid_df.isnull().sum())


# In[7]:


covid_df = covid_df.dropna()
print(covid_df.isnull().sum())


# In[9]:


covid_df.drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], inplace=True, axis=1)
covid_df.head(2)


# In[10]:


covid_df['Date'] = pd.to_datetime(covid_df['Date'], format='%Y-%m-%d')
covid_df.head(2)


# In[73]:


vaccine_df = pd.read_csv(r'C:\Users\HP\Documents\Simplilearn projects\covid_vaccine_statewise.csv')
vaccine_df.head(3)


# In[74]:


vaccine_df.info()


# In[75]:


vaccine_df.shape


# In[76]:


print(vaccine_df.isnull().sum())


# In[ ]:





# In[77]:


vaccine_df.head()


# In[22]:


#Finding duplicates

duplicates = covid_df[covid_df.duplicated()]
duplicates


# In[23]:


#Active cases

covid_df['Active cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df['Active cases']


# In[96]:


covid_df.head(3)


# In[27]:


statewise = pd.pivot_table(covid_df, values=['Confirmed', 'Deaths', 'Cured'], index = 'State/UnionTerritory',
                          aggfunc=max)
statewise


# In[29]:


statewise['Recovery rate'] = (statewise['Cured']/statewise['Confirmed']) * 100


# In[30]:


statewise['Mortality rate'] = (statewise['Deaths']/statewise['Confirmed']) * 100


# In[32]:


statewise = statewise.sort_values(by = 'Confirmed', ascending=False)


# In[33]:


statewise


# In[43]:


#top 10 active states

top10_activecases = covid_df.groupby(by='State/UnionTerritory').max()[['Active cases', 'Date']].sort_values(by=['Active cases'],ascending = False).reset_index()
top10_activecases = top10_activecases.head(10)


# In[45]:


fig = plt.figure(figsize=(15, 5))

ax = sns.barplot(data=top10_activecases, x = 'State/UnionTerritory', y='Active cases')
plt.title('Most Active Cases in India')
plt.xlabel('State/Union Territoy')
plt.ylabel('No of Active casees')

plt.show()


# In[47]:


#Top 10 states with highest deaths

top10_deaths = covid_df.groupby(by='State/UnionTerritory').max()[['Deaths', 'Date']].sort_values(by='Deaths', ascending=False).reset_index()
top10_deaths=top10_deaths.head(10)


# In[48]:


fig = plt.figure(figsize=(15,5))

ax = sns.barplot(data=top10_deaths, x='State/UnionTerritory', y='Deaths')
plt.title('Top 10 States with Highest Deaths')
plt.xlabel('State/Union Territory')
plt.ylabel('No of Deaths')

plt.show()


# In[51]:


#Top 10 States with most cured patients

top10_cured = covid_df.groupby(by='State/UnionTerritory').max()[['Cured', 'Date']].sort_values(by='Cured', ascending=False).reset_index()
top10_cured = top10_cured.head(10)


# In[54]:


#Plot the graph

fig = plt.figure(figsize=(15, 5))

ax=sns.barplot(data=top10_cured, x='State/UnionTerritory', y='Cured')
plt.title('States Highest Number of Cured Patients')
plt.ylabel('No of cured patients')
plt.xlabel('State/Union Territory')

plt.show()


# In[58]:


#Growth trend

fig = plt.figure(figsize=(10,6))

ax=sns.lineplot(data=covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Uttar Pradesh',])], x='Date', y='Active cases', hue='State/UnionTerritory')
plt.title('Covid Growth Trend')
plt.xlabel('Date')
plt.ylabel('No of Active cases')

plt.show()


# In[78]:


vaccine_df.head(3)


# In[79]:


vaccine_df.rename(columns={'Updated On':'Vaccine Date'}, inplace=True)
vaccine_df.head(2)


# In[80]:


vaccine_df.info()


# In[83]:


vaccination = vaccine_df.drop(['Sputnik V (Doses Administered)','AEFI', '18-44 Years (Doses Administered)', '45-60 Years (Doses Administered)',
                '60+ Years (Doses Administered)'], axis=1)

vaccination.info()


# In[97]:


#Male vs Female Vaccination

male = vaccination['Male(Individuals Vaccinated)'].sum()
female = vaccination['Female(Individuals Vaccinated)'].sum()

px.pie(names= ['Male', 'Female'], values=[male, female], title='Male and Female Vaccination')


# In[85]:


#Remove rows where state is India

vaccine = vaccine_df[vaccine_df.State!='India']
vaccine


# In[89]:


vaccine.rename(columns={'Total Individuals Vaccinated': 'Total'}, inplace=True)
vaccine.info()


# In[91]:


#Most Vaccinated States

max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values(by='Total', ascending=False)
max_vac


# In[95]:


top5_vac = max_vac.head(5)
fig = plt.figure(figsize=(15, 5))

ax= sns.barplot(data=top5_vac, x=top5_vac.index,y=top5_vac.Total)
plt.title('Most Vaccinated States')
plt.xlabel('State')
plt.ylabel('Total')

plt.show()


# In[102]:


#Top 5 states with the least vaccination rates

top5_leastvac = max_vac.tail(5)

fig = plt.figure(figsize=(15, 5))
ax=sns.barplot(data=top10_leastvac, x=top5_leastvac.index, y=top5_leastvac.Total)

plt.title('Least Vaccinated States')
plt.xlabel('State')
plt.ylabel('Total')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




