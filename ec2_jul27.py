"""
- Initial column, BC, financial filtering of best filings from sas file
- calculate continuous variables:
a) expense ratio; diagnostics and standardization
b) owner occupied ratio
- encode categorical variables:
a) building class
b) previous year determination (PENDING)
c) identity of filer
d) filer lease type determination
e) occurrence of physical construction (PENDING)
f) expense categories - expense types are grouped by name
- import the response variable - actual determinations made by assessors for current FY

REMAINING: 
1) pull the appropriate total income and total expense fields and divide them by sqft... 
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
prev_16.dtypes
#newbase data... I inadvertently left out sqft of res and comm
prev_16=pd.read_csv('/home/lechuza/Documents/aws/tripleNetLease/previousYrDetermination.csv')
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



#MERGING NEWBASE to RPIE
prev_16[['bbl','NETFLG']].tail(20)
len(prev_16['bbl'].unique())
cut1['BORO']=cut1['BORO'].astype(np.float64)
# only yielded 1,890... matching on bbl doesn't work well
# cut2=cut1.merge(prev_16[['bbl','NETFLG']],left_on='bbl_key',right_on='bbl')
# this is a better merge
cut2=cut1.merge(prev_16,left_on=['BORO','BLOCK','LOT'],right_on=['BORO','BLOCK','LOT'])
#cut3=cut1.merge(allV[['bbl','lease_type']],left_on='bbl_key',right_on='bbl')

''' end the merge of NNN '''
    
''' CALCULATE CONTINUOUS VARIABLES '''
# 1) Calculating expense ratio... will be standardized via a robust scalar
#impute a '1' should the filer report 0 income
cut2.loc[cut2['F_RE_TOT_INC']==0.0,'F_RE_TOT_INC']=1
#calculate expense ratio
cut2['exp_ratio']=cut2['F_TOT_EXP']/cut2['F_RE_TOT_INC']

''' visualize expense ratio on this dataset
plt.hist(cut1['exp_ratio'],bins=25)
plt.title('Distribution of expense ratio - untreated')
plt.show()

#remove outliers approach
np.percentile(cut1['exp_ratio'].values,[25,50,75,95,98])
#13,110 will be established as the upper bound
'''

ninetyeight=np.percentile(cut2['exp_ratio'].values,[98])[0]
trunc=cut2.loc[cut2['exp_ratio']<ninetyeight,:]


# 2) OWNER occupied income... ratio: owner occupied income / total income
trunc['own_occ_ratio']=trunc['OWN_OCC_INC']/trunc['F_RE_TOT_INC']
#impute 0 should value be None
trunc.loc[pd.isnull(trunc['own_occ_ratio']),'own_occ_ratio']=0

#quality check the ratio... it looks fine, with outliers in an acceptable range
plt.hist(trunc['own_occ_ratio'],bins=25)
plt.show()

# square footage... calc exp and income psf

''' end calculation of continuous variables '''


''' encode catgorical variables '''
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






''' RESPONSE VARIABLE: import lease type determinations for 2017/18 '''
leaseType_18=pd.read_csv('G:/Property/Luis_C/TripleNetLease/RPIE_cama_datasets/tripleNetLease_allValued_2018_II.csv',skiprows=1)
leaseType_18.columns=['value_src','boro','block','lot','tc','bc','lease_type']
leaseType_18['bbl_key']=leaseType_18.apply(lambda x: x['boro']*1000000000+x['block']*10000+x['lot'],axis=1)
leaseType_18.head()
#merge these to the training dataset... this is the response variable
''' end importing lease type determinations '''

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

'''concatenate all the dummy variables with the continous ones: 
   exp_cats_bools ... the 'NETFLG' column is the response variable... will conduct the study in R
'''
trunc.dtypes
#drop redundant columns: those that have been converted to dummy variables
processed_df=pd.concat([explicit_exp_df,trunc[['BORO','FILER_RELATN','IE_BC','IE_TC','F_RE_TOT_INC','F_TOT_EXP','TOT_EXP','TRPL_NET_LEASE','NETFLG','exp_ratio','own_occ_ratio','bbl','own_occ_ratio','exp_ratio']]],axis=1)
processed_df.to_csv('/home/lechuza/Documents/aws/tripleNetLease/processed_df',index=False)

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


''' modeling: 
    a) sklearn.linear_model.LogisticRegression
    b) statsmodels.discrete.discrete_model
'''    
 