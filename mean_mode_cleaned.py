
# coding: utf-8

# In[1]:

#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
df=pd.read_csv("LBW_Dataset.csv")
inte=df.interpolate()


# In[2]:

#age is a numerical column. so the nan values are replaced by the median of the present values
age=df['Age'].median()
#print(age)
age=round(age)
df['Age']=df['Age'].fillna(age)
        


# In[3]:

#weight is a numerical column. so the nan values are replaced by the median of the present values
weight=df['Weight'].median()
weight=round(weight)
df['Weight']=df['Weight'].fillna(weight)


# In[4]:

#Delivery phase is a categorical column. so the nan values are replaced by the mode of the present values
delivery_phase=int(df['Delivery phase'].mode())
#delivery_phase=round(delivery_phase)
#print(delivery_phase)
df['Delivery phase']=df['Delivery phase'].fillna(delivery_phase)


# In[6]:

#HB is a numerical column. so the nan values are replaced by the median of the present values
hb=df['HB'].median()
hb=round(hb,2)
df['HB']=df['HB'].fillna(hb)


# In[7]:

#IFA is a categorical column. so the nan values are replaced by the mode of the present values
ifa=int(df['IFA'].mode())
df['IFA']=df['IFA'].fillna(ifa)


# In[8]:
#BP is a numerical column. so the nan values are replaced by the median of the present values

bp=df['BP'].median()
df['BP']=df['BP'].fillna(bp)


# In[9]:
scaler_object=MinMaxScaler()
#education is a categorical column. so the nan values are replaced by the mode of the present values
education=int(df['Education'].mode())
education
df['Education']=df['Education'].fillna(education)


# In[11]:

#residence is a categorical column. so the nan values are replaced by the mode of the present values
residence=int(df['Residence'].mode())
residence
df['Residence']=df['Residence'].fillna(residence)


# In[12]:

#community is a categorical column. so the nan values are replaced by the mode of the present values
community=int(df['Community'].mode())
community
df['Community']=df['Community'].fillna(community)



# In[14]:
#all values are now normalised via the min-max normalisation
skip=['Community','Residence','Education','IFA','Delivery phase']
for i in df.columns[:-1]:
    if i not in skip:
        mini=df[i].min()
        maxi=df[i].max()
        if mini != maxi:
            for j in range(len(df[i])):
                df[i][j]=(df[i][j]-mini)/(maxi-mini)

#del df[0]
#X=df.iloc[:,:-1]
#X.iloc[:,[1,2,4,6]]=scaler_object.fit_transform(X.iloc[:,[1,2,4,6]])
#df.iloc[:,:-1]=X
    
df.to_csv('mode_median_cleaned.csv',index=False)


# In[46]:


#interpolate
'''inte=df.interpolate()
inte
inte.to_csv('interpolated_cleaned.csv')
'''

