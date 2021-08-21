

import pandas as pd
import numpy as np
df=pd.read_csv("LBW_Dataset.csv")
inte=df.interpolate()




#age is a numerical column. so the nan values are replaced by the median of the present values
age=df['Age'].median()
#print(age)
age=round(age)
df['Age']=df['Age'].fillna(age)
        




#weight is a numerical column. so the nan values are replaced by the median of the present values
weight=df['Weight'].median()
weight=round(weight)
df['Weight']=df['Weight'].fillna(weight)




#Delivery phase is a categorical column. so the nan values are replaced by the mode of the present values
delivery_phase=int(df['Delivery phase'].mode())

df['Delivery phase']=df['Delivery phase'].fillna(delivery_phase)




#HB is a numerical column. so the nan values are replaced by the median of the present values
hb=df['HB'].median()
hb=round(hb,2)
df['HB']=df['HB'].fillna(hb)




#IFA is a categorical column. so the nan values are replaced by the mode of the present values
ifa=int(df['IFA'].mode())
df['IFA']=df['IFA'].fillna(ifa)



#BP is a numerical column. so the nan values are replaced by the median of the present values

bp=df['BP'].median()
df['BP']=df['BP'].fillna(bp)



scaler_object=MinMaxScaler()
#education is a categorical column. so the nan values are replaced by the mode of the present values
education=int(df['Education'].mode())

df['Education']=df['Education'].fillna(education)




#residence is a categorical column. so the nan values are replaced by the mode of the present values
residence=int(df['Residence'].mode())

df['Residence']=df['Residence'].fillna(residence)




#community is a categorical column. so the nan values are replaced by the mode of the present values
community=int(df['Community'].mode())

df['Community']=df['Community'].fillna(community)




#all values are now normalised via the min-max normalisation
skip=['Community','Residence','Education','IFA','Delivery phase']
for i in df.columns[:-1]:
    if i not in skip:
        mini=df[i].min()
        maxi=df[i].max()
        if mini != maxi:
            for j in range(len(df[i])):
                df[i][j]=(df[i][j]-mini)/(maxi-mini)


    
df.to_csv('mode_median_cleaned.csv',index=False)



'''inte=df.interpolate()
inte
inte.to_csv('interpolated_cleaned.csv')
'''

