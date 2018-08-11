library(tidyverse)
library(sas7bdat)
library(haven)
library(stringr)
install.packages("haven")


#readr package for reading and parsing sas data files... it is stored as a tibble
dob<-read_sas('G:/Property_ORGPROJ/Orgproj/AnalyticsGrp/tom/scr/adhoc_06192018/dob.sas7bdat')

dob.df<-as.data.frame(dob)
class(dob)
class(dob.df)

str(dob)

#& dob$PERMIT_DATE<"2016-01-01" 
completed.jobs.2015<-dob.df[(dob.df$COMPLETION_DATE<"2016-03-30" & dob.df$COMPLETION_DATE>"2015-04-01" & dob.df$PERMIT_DATE>"2014-01-31"),]

#filtering a tibble
all(is.na(dob$COMPLETION_DATE))
head(dob$COMPLETION_DATE[!is.na(dob$COMPLETION_DATE)])
completed.jobs.2015<-filter(dob,COMPLETION_DATE<"2016-03-30" & COMPLETION_DATE>"2015-04-01")

head(dob$BORO)
head(completed.jobs.2015)
completed.jobs.2015$BORO[0:4]

#create BBL codes
completed.jobs.2015$BBL<-as.numeric(completed.jobs.2015$BORO)*1000000000+as.numeric(completed.jobs.2015$BLOCK)*10000 + as.numeric(completed.jobs.2015$LOT)

#uncompleted jobs that were likely being finished in the year
dob$PERMIT_DATE>"2014-12-31" & 
  
table(dob$JOB_TYPE)
dob[,c('BORO''BLOCK','LOT',)]

head(completed.jobs.2015$BORO)
head(as.numeric(completed.jobs.2015$BORO))

head(completed.jobs.2015$BBL)

#read in newbase file
nb<-read_sas('G:/Property_ORGPROJ/Orgproj/ORG_DATA_FOLDER/orgproj/newbase/newbasb_keep_d051917s.sas7bdat')
nb2<-read_sas('G:/Property_ORGPROJ/Orgproj/ORG_DATA_FOLDER/orgproj/newbase/newbasb_keep_d052416s.sas7bdat')

#read in newbase file from EC2


cols<-names(nb)
#apply regex and search for NNN
bools2<-grepl('SQFT|SF',cols)
bools<-grepl('LEASE|BORO|BLOCK|LOT|GROSS|GRO|NET|BLO',cols)
sum(bools2)

cols[bools2]
table(nb$NETFLG)

#export BBL and netflg
tgt<-c('BORO','BLOCK','LOT','NETFLG','COMSQFT','RESSQFT','GARSQFT')
portion<-nb2[,tgt]

write.csv(portion,'G:/Property/Luis_C/aws/fy16_determination.csv',row.names=FALSE)
