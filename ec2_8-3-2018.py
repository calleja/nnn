# -*- coding: utf-8 -*-
"""
- Initial column, BC, financial filtering of best filings from sas file
- calculate CONTINUOUS variables:
a) expense ratio; diagnostics and standardization
b) owner occupied ratio
- encode CATEGORICAL variables:
a) building class
b) previous year determination (PENDING)
c) identity of filer
d) filer lease type determination
e) occurrence of physical construction (PENDING)
f) expense categories - expense types are grouped by name
- import the response variable - actual determinations made by assessors for current FY

REMAINING: 
1) pull the appropriate total income and total expense fields and divide them by sqft... pull this from newbase file
2) make determination on whether I'll include previous year determination
3) verify accuracy of newbase triple net lease field    
"""

import pymongo
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  preprocessing
from sklearn import model_selection

#importing from SAS... filer15 data is submitted on 6/16 and applies to FY 17/18... the tentative valuation associated is 01/17 and the final roll is 6/17; and so a CAMA pull around that date corresponds to "current determination" (response) and CAMA data from 6/16 is previous yr determination (what we're) using as an independent variable
path='G:/Property _CommercialModeling/SAS Data Sets/'
best_15=pd.read_sas(path+'ie_re_best_filer15.sas7bdat')
df15_c=best_15.copy()

#importing from EC2
#defaults to port 27017
client = pymongo.MongoClient("mongodb://luis:persyy@18.222.226.207:27017/rpie")    
db = client.rpie
#point to the appropriate collection
filings=db['filings']
#retrieve only 2015 filings
df_15=filings.find({'filings':{'$elemMatch':{'IE_YEAR':'2015'}}})

df_15_d=filings.aggregate([{'$match':{'filings':{'$elemMatch':{'IE_YEAR':'2015'}}}},{'$project':{'bbl':{'_id':1}, 'rpie':{'$arrayElemAt':['$filings',0]}}}])

                           
df_15=pd.DataFrame(list(df_15_d))
gh=pd.DataFrame.from_dict(df_15.iloc[:,1].tolist())
#re-merge bbl to the dataframe
df15_c=pd.concat([df_15.iloc[:,0],gh],axis=1)
df15_c[['bbl_key','_id']].head()

#filter the desired columns
cut=df15_c.filter(regex='(bbl_key|FILER_RELATN|TOT_INC|IE_TC|IE_BC|_INC|_EXP|BORO|BLOCK|LOT|IE_SEQ|IE_YEAR|UPDTNO|TRPL_NET_LEASE)',axis=1)
#remove all "condo blocks" fields, ie CND_BLOCK
cnd_block=re.compile('^(?!CND_BLOCK).')
cut=cut.filter(regex=cnd_block,axis=1)

#if from sas files
cut['IE_BC']=cut['IE_BC'].str.decode("utf-8")
cut['BLOCK']=cut['BLOCK'].astype(int).astype(str)
cut['LOT']=cut['LOT'].astype(int).astype(str)
cut['BORO']=cut['BORO'].str.decode("utf-8")
cut['bbl']=cut.apply(lambda x: int(x['BORO'])*1000000000 + int(x['BLOCK'])*10000 + int(x['LOT']),axis=1)

'''remove incomplete filings and those outside of the study's purview..
a) remove coops, tax-exempt, vacant land, TC 1AC, 2ABC
b) no income information... all inc and exp categories are None
c) no total expense and income   
'''

#BC filtering
#negative lookbehind... remove observations with BC starting w/below letters... could inform this step by extracting all NNN from Vision and keeping only those BCs
#bc=re.compile('(?<!V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z).')
bc1=re.compile('^(V|A|B|C|D|H|I|J|M|N|P|Q|T|U|W|Y|Z)')
test=cut['IE_BC'].str.contains(bc1)
cut1=cut[~test]  

''' MERGE the triple net lease determination and insert in the db '''
#compare 2016 from "previousYrDetermination.csv" to "tripleNetLease_allValued_2016.csv" to confirm agreement - not observed; so ambiguous which dataset to accept as accurate
allV=pd.read_csv('/home/lechuza/Documents/aws/tripleNetLease/tripleNetLease_allValued_2016.csv')

#newbase data... I inadvertently left out sqft of res and comm
prev_16=pd.read_csv('/home/lechuza/Documents/aws/tripleNetLease/previousYrDetermination.csv')
prev_17=pd.read_csv('/home/lechuza/Documents/aws/tripleNetLease/previousYrDetermination.csv')
prev_16=pd.read_csv('G:/Property/Luis_C/aws/fy16_determination.csv')
prev_17=pd.read_csv('G:/Property/Luis_C/aws/fy17_determination.csv')

tots=prev_16.merge(allV,left_on=['BORO','BLOCK','LOT'],right_on=['boro','block','lot'])

#do the lease types match?
tots['lease_type'].head()
tots['NETFLG'].head()
tots['NETFLG'].value_counts()
tots['lease_type'].value_counts()
test=tots.apply(lambda x: 'agree' if (x['lease_type']=='G' and x['NETFLG']=='N') | (x['lease_type']=='NNN' and x['NETFLG']=='Y') else 'disagree',axis=1)
tots.loc[test.apply(lambda x: x=='disagree'),'lease_type'].value_counts()

tots['lease_type','NETFLG']
#bottom line: we don't yet have a reputable source for lease types, so cannot yet upload anything to the db

#default to newbase for now
prev_16['bbl']=prev_16['BORO']*1000000000+prev_16['BLOCK']*1000+prev_16['LOT']
allV['bbl']=allV['boro']*1000000000+allV['block']*1000+allV['lot']



#MERGING NEWBASE to RPIE: extract total sqft and lease type fields - do this in R
#cut1 merged with that fiscal year's data (effectively with the response variable attached)
cut1['BORO']=cut1['BORO'].astype(np.int64)
cut1['BLOCK']=cut1['BLOCK'].astype(np.int64)
cut1['LOT']=cut1['LOT'].astype(np.int64)
#is there a dtype issue?
cut1[['BORO','BLOCK','LOT']].dtypes
prev_17[['BORO','BLOCK','LOT']].dtypes
cut1_w17=cut1.merge(prev_17,on=['BORO','BLOCK','LOT'])


# merging revious (FY16/17) data and storing to a diff df, although should be another field of cut1_17
cut1_16=cut1.merge(prev_16,on=['BORO','BLOCK','LOT'])
''' end the merge of NNN '''
    
''' CALCULATE CONTINUOUS VARIABLES '''
# 1) EXPENSE psft
cut1_w17['total_sqft']=cut1_w17['RESSQFT']+cut1_w17['COMSQFT']
cut1_w17['exp_psft']=cut1_w17['F_TOT_EXP']/cut1_w17['total_sqft']

# 2) INCOME psft
cut1_w17['inc_psft']=cut1_w17['F_RE_TOT_INC']/cut1_w17['total_sqft']

# 3) Calculating expense ratio... will be standardized via a robust scalar
#impute a '1' should the filer report 0 income
cut1_w17.loc[cut1_w17['F_RE_TOT_INC']==0.0,'F_RE_TOT_INC']=1
#calculate expense ratio
cut1_w17['exp_ratio']=cut1_w17['F_TOT_EXP']/cut1_w17['F_RE_TOT_INC']

''' visualize expense ratio on this dataset
plt.hist(cut1['exp_ratio'],bins=25)
plt.title('Distribution of expense ratio - untreated')
plt.show()

#remove outliers approach
np.percentile(cut1['exp_ratio'].values,[25,50,75,95,98])
#13,110 will be established as the upper bound
'''

ninetyeight=np.percentile(cut1_w17['exp_ratio'].values,[98])[0]
trunc=cut1_w17.loc[cut1_w17['exp_ratio']<ninetyeight,:]

# 4) OWNER occupied income... ratio: owner occupied income / total income
trunc['own_occ_ratio']=trunc['OWN_OCC_INC']/trunc['F_RE_TOT_INC']
#impute 0 should value be None
trunc.loc[pd.isnull(trunc['own_occ_ratio']),'own_occ_ratio']=0

#quality check the ratio... it looks fine, with outliers in an acceptable range
plt.hist(trunc['own_occ_ratio'],bins=25)
plt.show()

# square footage... calc exp and income psf

''' end calculation of continuous variables '''


''' ENCODE CATEGORICAL VARIABLES '''
''' convert datatype of the object fields to type 'category'... each category is assigned an integer value... I could use this interger value for the label encoding... "cat.codes" attribute of the new column will be of type integer
a) building class
b) previous year determination
c) identity of filer
d) their own determination
e) occurrence of physical construction    

    '''
''' identify parcels undergoing physical improvement '''
dob=pd.read_sas('G:/Property_ORGPROJ/Orgproj/AnalyticsGrp/tom/scr/adhoc_06192018/dob.sas7bdat')

''' end physical improvements queries '''

''' import categorical variable: previous year determination - which should impact and be recorded in the FY 17/18 db'''



'''  end import categorical variable: previous year determination '''


''' import previous year categorizations '''
cats=pd.read_csv('G:/Property/Luis_C/TripleNetLease/RPIE_cama_datasets/nnn_classification_201617.csv',skiprows=1, header=None,names=['pid','Boro','Block','Lot','filing','lease','tc','bc','sqft'])
cats.describe()
cats.dtypes
cats['bbl']=cats.apply(lambda x: int(x['Boro'])*1000000000 + int(x['Block'])*10000 + int(x['Lot']),axis=1)
cats['bbl'].head()
cats['lease'].value_counts()
cats.sort_values('pid').head(20)
#merge these via a left join... if a property does not havea lease type on record from previous year, mark an "na"
cut1_l=cut1.merge(cats[['bbl','lease','sqft']],on='bbl',how='inner')
''' end previous year categorizations '''


#filer type and expenses filed needs to be an interaction variable (A categorical w/a numeric)
'''per Joe Bedzula: From the guide … In a triple net lease, the tenant pays all the expenses. So if the owner is filing the RPIE, there should not be any expenses filed if the lease is truly triple net. However, DOF allows for INSURANCE & MANAMENT as expenses.
If the “lessee” is filing the RPIE, then we should see all of their operating expenses if it’s a triple net.

I think it is crucial to consider who the filer is and how they answer the lease type questions and then what expenses are filed; from there we can determine what lease type we are working with.'''

trunc.filter(regex=r'EXP',axis=1).dtypes
#these will be isolated... all others not in this list, will be rolled up
exp=['LEAS1_EXP', 'LEAS2_TEN_IMP_EXP', 'MANAG_EXP','INSUR_EXP'] 
#relabel the columns so that I can code a catchall label for all other categories not included in the list
#if there is a 0 in this column, it means there is no expense
#will need to transpose the dataframe on these columns; a df of booleans for each expense
exp_df=trunc.filter(regex=r'EXP',axis=1).applymap(lambda x: x>0.0)
#isolate all the catch-all expense columns by label
cols_catchall=[x not in exp for x in exp_df.columns]
t=exp_df.loc[:,cols_catchall]
#if any element across the row of a BBL contains true, then the bbl has that expense... this will be converted to series and appended to the dataframe of the explicit expenses
misc_exp_bools=t.apply(lambda x: any(x), axis=1)
#first, create the dataframe of the explicit expenses
explicit_exp_df=trunc[exp].applymap(lambda x: x>0.0)
explicit_exp_df['CATCH_ALL_EXP']=misc_exp_bools
explicit_exp_df.head()
# a dataframe of dummy variables... will need to concatenate with 'trunc' dataframe
exp_cats_bools=explicit_exp_df.copy()
#prepare for dummies... function requires that the column be object or category, so I'll convert the columns by apply a a binary function
exp_cats_bools=exp_cats_bools.applymap(lambda x: 'reports' if x  else 'no')
exp_cats_bools.head()

trunc_exp_cats=pd.concat([trunc,exp_cats_bools],axis=1)
#we can see below that when the catchall expenses are not reported, there are significantly more NETFLG = 'Y'
pd.crosstab(trunc_exp_cats['CATCH_ALL_EXP'],trunc_exp_cats['NETFLG']).apply(lambda r: r/r.sum(), axis=1)
#each of these variables will need to be stored on their own field bc there is a one:many relationship


trunc_exp_cats.to_csv('/home/lechuza/Documents/aws/tripleNetLease/trunc_exp_cats',index=False)

trunc_exp_cats.to_csv('G:/Property/Luis_C/TripleNetLease/trunc_exp_cats.csv',index=False)


'''STANDARDIZE all the continuous variables: own occupancy income, income psft, expense psft, expense ratio '''
#apply a Robust Scalar to the training data... this scaling object will be applied to the test set... but will not be refit
scaler_exp_ratio= preprocessing.RobustScaler()
vals=train['exp_ratio'].values
vals_r=vals.reshape(len(vals),1)
robust_scaled_series=scaler_exp_ratio.fit_transform(vals_r)

plt.hist(robust_scaled_series,bins=25)
plt.show()

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

''' Split data into a test and training... use sklearn's random splitting function '''
train,test=model_selection.train_test_split(trunc,test_size=0.3)
#currently missing the dependent variable: the determination by assessors of the lease type

''' modeling: 
    a) sklearn.linear_model.LogisticRegression
    b) statsmodels.discrete.discrete_model
'''    
 

