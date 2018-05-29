# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:05:36 2018

operations:
    
1) select columns to keep
2) filter for building class
3)    
    
TODO: ensure that all dataframes are aligned properly    
"""

import pymongo
import pandas as pd
import re
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  preprocessing
from sklearn import model_selection

#"mongodb://ian:secretPassword@123.45.67.89/cool_db
mongo= pymongo.MongoClient('mongodb://luis:persyy@18.221.52.113/rpie')
db=mongo.rpie
collect=db.inc_2015

df=collect.find()
df1=pd.DataFrame(list(df))
df1.dtypes

#filter the desired columns
cut=df1.filter(regex='(FILER_RELATN|TOT_INC|IE_TC|IE_BC|_INC|_EXP|BORO|BLOCK|LOT|IE_SEQ|IE_YEAR|UPDTNO|TRPL_NET_LEASE)',axis=1)
#remove all "condo blocks" fields, ie CND_BLOCK
cnd_block=re.compile('^(?!CND_BLOCK).')
cut=cut.filter(regex=cnd_block,axis=1) 

'''remove incomplete filings and those outside of the study's purview..
a) remove coops, tax-exempt, vacant land, TC 1AC, 2ABC
b) no income information... all inc and exp categories are None
c) no total expense and income   
    '''
cut['IE_BC']=cut['IE_BC'].str.decode("utf-8")

cut['BLOCK']=cut['BLOCK'].astype(int).astype(str)
cut['LOT']=cut['LOT'].astype(int).astype(str)
cut['BORO']=cut['BORO'].str.decode("utf-8")
#negative lookbehind... remove observations with BC starting w/below letters... could inform this step by extracting all NNN from Vision and keeping only those BCs
#bc=re.compile('(?<!V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z).')
bc1=re.compile('^(V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z)')
test=cut['IE_BC'].str.contains(bc1)
cut['IE_BC'].head()
cut1=cut[~test]  

encoded_bbls=cut1.apply(lambda x: str.encode('-'.join([x['BORO'],x['BLOCK'],x['LOT']])),axis=1)

hashed_bbls=encoded_bbls.apply(lambda x: hashlib.sha256(x).hexdigest())
cut1['key']=hashed_bbls
cut1.drop(['BORO','BLOCK','LOT'],inplace=True,axis=1)
# print the number of documents in a collection
lista=cut1.iloc[8:,].to_dict(orient='records')
#print db.cool_collection.count()

''' connecting to mongodb from work '''
client = pymongo.MongoClient("mongodb://luis:persyy@127.0.0.1:8080/rpie") # 
db = client.rpie
vals15=db['inc_2015']
#records insertion
for i in lista:
    vals15.insert_one(i)
    
''' CALCULATE CONTINUOUS VARIABLES '''
# 1) Calculating expense ratio... will be standardized via a robust scalar
#impute a '1' should the filer report 0 income
cut1.loc[df1['F_RE_TOT_INC']==0.0,'F_RE_TOT_INC']=1
#calculate expense ratio
cut1['exp_ratio']=cut1['F_TOT_EXP']/cut1['F_RE_TOT_INC']

#visualize expense ratio on this dataset
plt.hist(df1['exp_ratio'],bins=25)
plt.show()
#remove outliers approach
np.percentile(cut1['exp_ratio'].values,[25,50,75,95,98])
#13,110 will be established as the upper bound
ninetyeight=np.percentile(cut1['exp_ratio'].values,[98])[0]

trunc=cut1.loc[cut1['exp_ratio']<ninetyeight,:]

# 2) OWNER occupied income... ratio: owner occupied income / total income
trunc['own_occ_ratio']=trunc['OWN_OCC_INC']/trunc['F_RE_TOT_INC']
#impute 0 should value be None
trunc.loc[pd.isnull(trunc['own_occ_ratio']),'own_occ_ratio']=0

#quality check the ratio... it looks fine, with outliers in an acceptable range
plt.hist(cut1['own_occ_ratio'],bins=25)
plt.show()

''' encode catgorical variables '''
#convert datatype of the object fields to type 'category'... each category is assigned an integer value... I could use this interger value for the label encoding... "cat.codes" attribute of the new column will be of type integer

'''one-hot encoding or dummy variables '''
#Building class dummies
#take major building class for all those outside of R... Rs will be taken whole
g=trunc.groupby('IE_BC')
bc=g.groups.keys()
bc_list=[i[0] if i[0] !='R' else i for i in bc]
#create dictionary from two lists... the values of this dictionary will be used to label/substitute the BC values in the dataframe... this will facilitate one hot encoding.
bc_dict=dict(zip(bc,bc_list))
bc_unique=np.unique(np.array(bc_list))
trunc['bc_code']=trunc['IE_BC'].replace(to_replace=bc_dict)
#select the reference variable... retail: "K"
dummies=pd.get_dummies(trunc['bc_code'])
dummies.dtypes
#drop the column 'K' because K will be the reference category
dummies.drop('K',inplace=True, axis=1)
dummies.head()

#calculate dummies for EXPENSE CATEGORIES... I'm essentially doing this manually: first I identify the presence of each expense type; 5 categories: 'LEAS1_EXP', 'LEAS2_TEN_IMP_EXP', 'MANAG_EXP','INSUR_EXP', and 'catch_all_exp' (for all other remaining categories). Because each category can be present, I'll one hot encode each... the reference group will be NOT PRESENT
trunc.filter(regex=r'EXP',axis=1).dtypes
exp=['LEAS1_EXP', 'LEAS2_TEN_IMP_EXP', 'MANAG_EXP','INSUR_EXP'] 
#relabel the columns so that I can code a catchall label for all other categories not included in the list
#if there is a 0 in this column, it means there is no expense
#will need to transpose the dataframe on these columns
exp_df=trunc.filter(regex=r'EXP',axis=1).applymap(lambda x: x>0.0)
#isolate all the catch-all expense columns by label
cols_catchall=[x not in exp for x in exp_df.columns]
t=exp_df.loc[:,cols_catchall]
#if any element across the row of a BBL contains true, then the bbl has that expense... this will be converted to series and appended to the dataframe of the explicit expenses
misc_exp_bools=t.apply(lambda x: any(x), axis=1)
#first, create the dataframe of the explicit expenses
explicit_exp_df=trunc[exp].applymap(lambda x: x>0.0)
explicit_exp_df['CATCH_ALL_EXP']=misc_exp_bools
# a dataframe of dummy variables... will need to concatenate with 'trunc' dataframe
exp_cats_bools=explicit_exp_df.copy()
#prepare for dummies... function requires that the column be object or category, so I'll convert the columns by apply a a binary function
exp_cats_bools=exp_cats_bools.applymap(lambda x: 'reports' if x  else 'no')
exp_cats_bools.head()
#make dummies
exp_dummies=pd.get_dummies(exp_cats_bools)
exp_dummies.dtypes
#select the dummy field we want to keep and drop the reference field
exp_dummies_keep=exp_dummies.iloc[:,[1,3,5,7,9]]

#make dummies out of BORO variable... do this from work


#make dummies out of triple net lease
trunc['TRPL_NET_LEASE']=trunc['TRPL_NET_LEASE'].str.decode("utf-8")
dum_nnn=pd.get_dummies(trunc['TRPL_NET_LEASE'])
#select a reference variable... No... implying "1" = yes a triple net lease
dum_nnn_final=dum_nnn.loc[:,'Y']

#combine all the datframes: NNN, exp_dummies_keep (expense categories), dummies (bldg class)
dummy_agg=pd.concat([dummies,exp_dummies_keep,dum_nnn_final],axis=1)

#concatenate all the dummy variables with the continous ones: 
trunc.dtypes
#drop redundant columns: those that have been converted to dummy variables
processed_df=pd.concat([dummy_agg,trunc[['own_occ_ratio','exp_ratio']]],axis=1)

''' Split data into a test and training... use sklearn's random splitting function '''
train,test=model_selection.train_test_split(trunc,test_size=0.3)
#currently missing the dependent variable: the determination by assessors of the lease type

#STANDARDIZE all the continuous variables: own occupancy income, expense ratio 
#apply a Robust Scalar to the training data... this scaling object will be applied to the test set... but will not be refit
scaler_exp_ratio= preprocessing.RobustScaler()
vals=train['exp_ratio'].values
vals_r=vals.reshape(len(vals),1)
robust_scaled_series=scaler_exp_ratio.fit_transform(vals_r)

scaler_own_occ= preprocessing.RobustScaler()
vals=train['own_occ_ratio'].values
vals_r=vals.reshape(len(vals),1)
robust_scaled_series_own=scaler_own_occ.fit_transform(vals_r)


#apply the Robust Scalar object to the TEST data
vals_test=test['exp_ratio'].values
vals_tr=vals_test.reshape(len(vals_test),1)
ma=scaler_exp_ratio.transform(vals_tr)
re_scaledt=ma.reshape(len(vals_tr))
sert=pd.Series(re_scaledt)
#owner occupancy data
vals_test=test['own_occ_ratio'].values
vals_tr=vals_test.reshape(len(vals_test),1)
ma=scaler_own_occ.transform(vals_tr)
re_scaledt=ma.reshape(len(vals_tr))
sert=pd.Series(re_scaledt)


''' modeling: 
    a) sklearn.linear_model.LogisticRegression
    b) statsmodels.discrete.discrete_model
'''    

