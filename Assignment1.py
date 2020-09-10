#!/usr/bin/env python
# coding: utf-8

# In[161]:


##Part 1 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# In[134]:


npdat = np.array([['','Cool','Beans'],
                ['1',9,6],
                ['2',6,9]])
npdat


# In[135]:


pdfrm=(pd.DataFrame(data=npdat[1:,1:],index=npdat[1:,0],columns=npdat[0,1:]))
pdfrm


# In[136]:


plt.plot([3, 6, 9, 10], [10, 9, 6, 3])
plt.axis([0, 6, 0, 20])
plt.show()


# In[137]:


fram_train = pd.read_csv("C:/Users/Administrator/Documents/School/PredictiveModels/WebsiteDataSets/Framingham_Training")
Train = pd.DataFrame(fram_train[['Sex','Age']])
Traink1 = KMeans(n_clusters = 4).fit(Train)
Traink1
Clust = Traink1.labels_
Train1Clust1 = Train.loc[Clust == 0]
Train1Clust1.describe()


# In[138]:


##Part 2


# In[139]:


re = pd.read_csv("C:/Users/Administrator/Documents/School/MachineLearning/Real.csv")
re.head()


# In[140]:


re.describe()


# In[141]:


# No and Transaction date will be dropped as they are not needed in this
# They do not provide any information that s needed 


# In[142]:


re.drop(['No'], axis=1, inplace=True)
re.drop(['X1transactiondate'],axis=1, inplace=True)


# In[143]:


plt.figure()
sns.heatmap(re.corr(),annot=True)


# In[144]:


# As far as the Data goes we can clearly see that the most correlation 
# is in the number of surrounding convenience stores though Logitude and 
# Latitude is high as well so we will stick with those three.
# We can now split the data with a 80/20 split 
re.drop(['X2houseage'],axis=1, inplace=True)
re.drop(['X3distancetothenearestMRTstation'],axis=1, inplace=True)


# In[145]:



np.random.seed(0)
re_train,re_test = train_test_split(re, train_size=0.80, test_size=0.20)


# In[146]:


# Separation of Y
rey_train = re_train.pop('Yhousepriceofunitarea')
reX_train = re_train


# In[147]:


#Fit the model Linear
lm = LinearRegression()
lm.fit(reX_train, rey_train)


# In[148]:


reX_train_lm = sm.add_constant(reX_train)
lm_1 = sm.OLS(rey_train, reX_train).fit()


# In[159]:


# Now we will evaluate the variables using vif to be sure they are 
#statistically signifiant 
vif = pd.DataFrame()
X = reX_train_lm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[150]:


# All of the features are 
# Now we will pop the test data so that we can separate the X and y data
rey_test = re_test.pop('Yhousepriceofunitarea')
reX_test = re_test


# In[157]:


rey_test_pred = lm_1.predict(reX_test)


# In[158]:


preddf = pd.DataFrame({'Actual':rey_test,'Predictions':rey_test_pred})
preddf['Predictions']= round(preddf['Predictions'])
preddf.head()


# In[160]:


sns.regplot('Actual','Predictions',data=preddf)


# In[163]:


# Its clear that with the current features we are seeing a pretty bad
# model we should re-evaluate adding the 2 dropped variables again
print('Squared Error',metrics.mean_squared_error(rey_test,rey_test_pred))


# In[ ]:




