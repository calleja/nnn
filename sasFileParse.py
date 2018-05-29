# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:20:18 2018

@author: CallejaL
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn import  preprocessing

path='G:/Property _CommercialModeling/SAS Data Sets/'

multi_15=pd.read_sas(path+'ie_re_mult_filer15.sas7bdat')
best_15=pd.read_sas(path+'ie_re_best_filer15.sas7bdat')

#filter the desired columns
cut=best_15.filter(regex='(FILER_RELATN|TOT_INC|IE_TC|IE_BC|_INC|_EXP|BORO|BLOCK|LOT|IE_SEQ|IE_YEAR|UPDTNO|TRPL_NET_LEASE)',axis=1)

#remove all "condo blocks" fields, ie CND_BLOCK
cnd_block=re.compile('^(?!CND_BLOCK).')
cut=cut.filter(regex=cnd_block,axis=1) 

'''remove incomplete filings and those outside of the study's purview..
a) remove coops, tax-exempt, vacant land, TC 1AC, 2ABC
b) no income information... all inc and exp categories are None
c) no total expense and income   
    '''
#('^(?!CND_BLOCK).')    
cut['IE_BC'].value_counts() #convert BC to string first
cut['IE_BC']=cut['IE_BC'].str.decode("utf-8")

#negative lookbehind... remove observatinos with BC starting w/below letters... could inform this step by extracting all NNN from Vision and keeping only those BCs
#bc=re.compile('(?<!V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z).')
bc1=re.compile('^(V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z)')
test=cut['IE_BC'].str.contains(bc1)
cut1=cut[~test]  

#TODO Study
# None values... may need to impute/substitute 0s for these... None rows are removed in the next process
cut1[['F_RE_TOT_INC','F_TOT_EXP']].isnull
cut1[cut1['F_RE_TOT_INC']==None].filter(regex='EXP').head()
any(cut1['F_RE_TOT_INC']==0)
cut1[cut1['F_RE_TOT_INC'].isnull()].filter(regex='EXP').head()
cut1[cut1['F_TOT_EXP'].isnull()].filter(regex='INC',axis=1)
cut1.to_csv('G:/Property/Luis_C/TripleNetLease/rpieSystemFiles/inspection1.csv')

#TODO STUDY conclusion: doesn't appear to be any null values in the total expense line items, the preiovus data step may have imputed these 0s, thus should consider treating 0s as null values

#remove filings with all 0 values... very few, only 49 meet the criteria
#identify filings w/ and w/o total expense/total income: 'F_RE_TOT_INC' and 'F_TOT_EXP'
zero_vals=cut1.loc[(cut1['F_RE_TOT_INC']==0)  & (cut1['F_TOT_EXP']==0),:]
#will remove these filings... applying an inverse index selection
cut11=cut1.index.difference(zero_vals.index)
cut12=cut1.loc[cut11,:]

#substitute for 1 where F_RE_TOT_INC = 0
#calculate expense ratio
cut12.loc[cut12['F_RE_TOT_INC']==0.0,'F_RE_TOT_INC']=1
cut12['exp_ratio']=cut12['F_TOT_EXP']/cut12['F_RE_TOT_INC']
#this is challenging: don't know how to handle: can either impute a random number, like the mean value, can set these aside, develop the model, then trace back, or can do a deeper dive and discover a pattern in these filings... recall that these are multiple filers, and so there may well be more than one filing for these parcels.
cut12.loc[cut12['F_RE_TOT_INC']==0.0,'exp_ratio']=1 #this affects 1,3725 parcels

#exploratory histogram plot of the distribution of observations for expense ratio
cut12['exp_ratio'].head()

np.amax(cut12['exp_ratio'].values)
np.amin(cut12['F_RE_TOT_INC'].values)

plt.hist(cut12['exp_ratio'],bins=25)
plt.show()

#convert to numpy.. the below two arrays are lengthy, combining for 4474 observations where expenses are higher than income... in some cases much higher... I will have to scale the data, and may even want to trim the extremes
over_1=cut12.loc[(cut12['exp_ratio']>1) & (cut12['exp_ratio']<250),'exp_ratio'].values

over_2=cut12.loc[cut12['exp_ratio']>250,'exp_ratio'].values
#what are the quantiles
over_2
#plot the expense ratio distribution
plt.close()
test=plt.hist(over_1,25,density=True)
plt.show()

#is the distribution of expense ratio at all normal? No, it is heavily skewed.
sm.qqplot(cut12['exp_ratio'].values,line='s')





''' end Best filing code '''

''' multiple filer code '''
#convert all string/obj fields to string from unicode
multi_15.BORO.head()
#presenting a lot of difficulties, may need to decode the series piecemeal and not systematically/iteratively, like below
multi_test=multi_15.apply(lambda x: x.str.decode("utf-8") if x.dtype == object else x,axis=0)

#inspect field data types and names
cols=multi_15.dtypes
cols.to_csv('G:/Property/Luis_C/TripleNetLease/rpieSystemFiles/multiFiler15_colnames.csv')

#explore filer information... self-identifying information
multi_15['FILER_RELATN'].value_counts()

#filter the desired columns
cut=multi_15.filter(regex='(FILER_RELATN|TOT_INC|IE_TC|IE_BC|_INC|_EXP|BORO|BLOCK|LOT|IE_SEQ|IE_YEAR|UPDTNO|TRPL_NET_LEASE)',axis=1)
cut.dtypes

#remove all "condo blocks" fields, ie CND_BLOCK
cnd_block=re.compile('^(?!CND_BLOCK).')
cut=cut.filter(regex=cnd_block,axis=1) 

'''remove incomplete filings and those outside of the study's purview..
a) remove coops, tax-exempt, vacant land, TC 1AC, 2ABC
b) no income information... all inc and exp categories are None
c) no total expense and income   
    '''
#('^(?!CND_BLOCK).')    
cut['IE_BC'].value_counts() #convert BC to string first
cut['IE_BC']=cut['IE_BC'].str.decode("utf-8")

#negative lookbehind... remove observatinos with BC starting w/below letters... could inform this step by extracting all NNN from Vision and keeping only those BCs
#bc=re.compile('(?<!V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z).')
bc1=re.compile('^(V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z)')
test=cut['IE_BC'].str.contains(bc1)
cut1=cut[~test]  

#TODO Study
# None values... may need to impute/substitute 0s for these... None rows are removed in the next process
cut1[['F_RE_TOT_INC','F_TOT_EXP']].isnull
cut1[cut1['F_RE_TOT_INC']==None].filter(regex='EXP').head()
any(cut1['F_RE_TOT_INC']==0)
cut1[cut1['F_RE_TOT_INC'].isnull()].filter(regex='EXP').head()
cut1[cut1['F_TOT_EXP'].isnull()].filter(regex='INC',axis=1)
cut1.to_csv('G:/Property/Luis_C/TripleNetLease/rpieSystemFiles/inspection1.csv')

#TODO STUDY conclusion: doesn't appear to be any null values in the total expense line items, the preiovus data step may have imputed these 0s, thus should consider treating 0s as null values

#remove filings with all 0 values... very few, only 49 meet the criteria
#identify filings w/ and w/o total expense/total income: 'F_RE_TOT_INC' and 'F_TOT_EXP'
zero_vals=cut1.loc[(cut1['F_RE_TOT_INC']==0)  & (cut1['F_TOT_EXP']==0),:]
#will remove these filings... applying an inverse index selection
cut11=cut1.index.difference(zero_vals.index)
cut12=cut1.loc[cut11,:]

#substitute for 1 where F_RE_TOT_INC = 0
#calculate expense ratio
cut12.loc[cut12['F_RE_TOT_INC']==0.0,'F_RE_TOT_INC']=1
cut12['exp_ratio']=cut12['F_TOT_EXP']/cut12['F_RE_TOT_INC']
#this is challenging: don't know how to handle: can either impute a random number, like the mean value, can set these aside, develop the model, then trace back, or can do a deeper dive and discover a pattern in these filings... recall that these are multiple filers, and so there may well be more than one filing for these parcels.
cut12.loc[cut12['F_RE_TOT_INC']==0.0,'exp_ratio']=1 #this affects 1,3725 parcels

#exploratory histogram plot of the distribution of observations for expense ratio
cut12['exp_ratio'].head()

np.amax(cut12['exp_ratio'].values)
np.amin(cut12['F_RE_TOT_INC'].values)

plt.hist(cut12['exp_ratio'],bins=25)
plt.show()

#QA check that no incomes are a decimal or negative: answer: NO
cut12['F_RE_TOT_INC'][(cut12['F_RE_TOT_INC']<1) & (cut12['F_RE_TOT_INC']>0)]
cut12['F_RE_TOT_INC'][cut12['F_RE_TOT_INC']<0]



#take a look at results
cut12[['F_TOT_EXP','F_RE_TOT_INC','exp_ratio']].tail(10)

#convert to numpy.. the below two arrays are lengthy, combining for 4474 observations where expenses are higher than income... in some cases much higher... I will have to scale the data, and may even want to trim the extremes
over_1=cut12.loc[(cut12['exp_ratio']>1) & (cut12['exp_ratio']<250),'exp_ratio'].values

over_2=cut12.loc[cut12['exp_ratio']>250,'exp_ratio'].values
#what are the quantiles
over_2
#plot the expense ratio distribution
plt.close()
test=plt.hist(over_1,25,density=True)
plt.show()

#solutution: apply a min max scaler; the book recommends: standardization via z-scores, but that's not appropriate on this Poisson-like dataset

#is the distribution of expense ratio at all normal? No, it is heavily skewed.
sm.qqplot(cut12['exp_ratio'].values,line='s')

plt.close()
fig,ax = plt.subplots(1,1)
ax.set_title('KDE of expense ratio of all observations')
sns.kdeplot(cut12['exp_ratio'],ax=ax)
plt.show()
''' Feature scaling in python resources
a) http://benalexkeen.com/feature-scaling-with-scikit-learn/
b) https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
c) http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
d) https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
e) https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-1-centering-scaling-and-knn
'''
#apply a Robust Scalar
scaler= preprocessing.RobustScaler()
vals=cut12['exp_ratio'].values
max(vals)
vals_r=vals.reshape(len(vals),1)
robust_scaled_series=scaler.fit_transform(vals_r)
robust_scaled_series[0:10]
max(robust_scaled_series)
re_scaled=robust_scaled_series.reshape(len(vals))
ser=pd.Series(re_scaled)

#robust scaler made things worse... I will have to trim outliers
plt.close()
fig,ax = plt.subplots(1,1)
ax.set_title('KDE of expense ratio post scaling')
sns.kdeplot(ser,ax=ax)
plt.show()

#TODO who the hell even has expenses that high relative to income
''' Calculate and derive:
    a) a total expense line that excludes redundant items
    b) a total revenue line that excludes redundant items
    c) dummy variables for each expense category (present or not)
    d) identify and standardize the filing party
'''    

#investigate whether any expense line items have the same value... actually, not necessary as there is a TOT line item: "F_TOT_EXP" and "S_TOT_EXP"

#Investigate expenses... stats on the two TOT expense line items
expenses=cut.filter(regex='EXP')

expenses[['S_TOT_EXP','F_TOT_EXP']].head()

def whichLarger(x):
    ratio=x['S_TOT_EXP']/x['F_TOT_EXP']
    if ratio>1:
        return('S_TOT_EXP')
    else:
        return('F_TOT_EXP')
testy=expenses.apply(whichLarger,axis=1)
#F_TOT_EXP is typically the appropriate total expense selection
testy.value_counts()

#investigate INCOME
income=cut.filter(regex='INC',axis=1)

def whichLargerInc(x):
    ratio=x['S_RE_TOT_INC']/x['F_RE_TOT_INC']
    if ratio>1:
        return('S_RE_TOT_INC')
    else:
        return('F_RE_TOT_INC')
testy_inc=income.apply(whichLargerInc,axis=1)
# F_RE_TOT_INC is typically the appropriate total expense selection, with about 5% of filings reporting higher S_RE_TOT_INC
testy_inc.value_counts()
#investigate those filings where S_RE_TOT_INC > F_RE_TOT_INC... CONCLUSION: no discernable patterns in the filer relationship field for those filing greater S_RE income
cut.loc[testy_inc=='S_RE_TOT_INC','FILER_RELATN'].value_counts()

#proportion by filer relationship... super majority categorize themselves as owner affiliated... 1.5% are aligned with lessees
cut['FILER_RELATN'].value_counts()/sum(cut['FILER_RELATN'].value_counts())




#create key field with which to merge both
canti['BBL']=canti['BORO']*1000000000+canti['BLOCK']*10000+canti['LOT']
value['BBL']=value['boro']*1000000000+value['block']*10000+value['lot']