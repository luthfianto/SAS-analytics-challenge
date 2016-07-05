# Current AUC 0.699465
# Last AUC 0.695045 -> LB 0.71219
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, RandomizedPCA
from sklearn.metrics import roc_auc_score
seed = 1909 

def ICD(x):
    if x.isdigit():
        x = int(x)
    elif len(x.split('.')) > 1:
        x = float(x)
        
    if     1 <= x <= 139: return 'infectious and parasitic diseases'
    elif 140 <= x <= 239: return 'neoplasms'
    elif 240 <= x <= 279: return 'endocrine, nutritional and metabolic diseases, and immunity disorders'
    elif 280 <= x <= 289: return 'blood and blood-forming organs'
    elif 290 <= x <= 319: return 'mental disorders'
    elif 320 <= x <= 359: return 'nervous system'
    elif 360 <= x <= 389: return 'sense organs'
    elif 390 <= x <= 459: return 'circulatory system'
    elif 460 <= x <= 519: return 'respiratory system'
    elif 520 <= x <= 579: return 'digestive system'
    elif 580 <= x <= 629: return 'genitourinary system'
    elif 630 <= x <= 679: return 'complications of pregnancy, childbirth, and the puerperium'
    elif 680 <= x <= 709: return 'skin and subcutaneous tissue'
    elif 710 <= x <= 739: return 'musculoskeletal system and connective tissue'
    elif 740 <= x <= 759: return 'congenital anomalies'
    elif 760 <= x <= 779: return 'certain conditions originating in the perinatal period'
    elif 780 <= x <= 799: return 'symptoms, signs, and ill-defined conditions'
    elif 800 <= x <= 999: return 'injury and poisoning'
    else:                 return x
    
###
# 'weight' # Banyak NaN
categorical_set = {
 'admission_source_id',
 'admission_type_id',
 'diabetesMed',
 'discharge_disposition_id',
 'max_glu_serum',
 'medical_specialty',
 'payer_code',
 'race',
 'diag_1',
 'diag_2',
 'diag_3'
 }
categorical=list(categorical_set)
###

train=pd.read_csv('SAStraining.csv')
test=pd.read_csv('SAStest.csv')

combined=pd.concat((train,test)).reset_index()

combined.diag_1 = combined.diag_1.apply(ICD)
combined.diag_2 = combined.diag_2.apply(ICD)
combined.diag_3 = combined.diag_3.apply(ICD)

combined[combined.columns[24:49]]=pd.DataFrame([combined[i].apply({'No':0}.get).replace(np.NaN, 1) for i in combined.columns[24:49] ]).T.astype(np.bool)

combined.gender=combined.gender.apply({'Male':1}.get).replace(np.NaN, 0).astype(np.bool)

combined.A1Cresult=combined.A1Cresult.apply({"Norm": -1, "None": 0, ">7": 1, ">8": 2}.get).replace(np.NaN, 0).astype(np.int8)
combined.A1Cresult=combined.A1Cresult.apply({"Norm": -1, "None": 0, ">7": 1, ">8": 2}.get).replace(np.NaN, 0).astype(np.int8)

combined.age=combined.age.str.split('-').str[0].str[1:3].astype(np.bool)
'weight'

onehoted=pd.concat([pd.get_dummies(combined[i]).astype(np.bool) for i in categorical], axis=1)

for i in categorical:
    del combined[i]

combined=pd.concat([combined,onehoted], axis=1)
###
object_columns=combined.loc[:, combined.dtypes == object].columns

train=combined[:8000]
test=combined[8000:]

X_train=train.drop(object_columns, axis=1).drop('readmitted', axis=1)
y_train =train.readmitted
# y_test =test.readmitted

tfidf=TfidfVectorizer(stop_words='english', ngram_range = (1,1)) # (1,2) sama aja kaya (1,1) sih kemarin
diag_1_desc_=tfidf.fit_transform(train.diag_1_desc.astype(np.str))

#svd=TruncatedSVD(90, random_state=seed)
#diag_1_desc=svd.fit_transform(diag_1_desc_)

rpca=RandomizedPCA(90, random_state=seed)
diag_1_desc=rpca.fit_transform(diag_1_desc_.toarray())

eks=np.hstack((X_train, diag_1_desc))



###
### EVALUATION
### Best w/  tf: ~0.695045 max_features=55     -> 0.71219
###               0.692145 max_features='auto'
### Best w/o tf: ~0.687561 max_features='auto'
###              ~0.682668 max_features=50    . Kesimpulan: kalo tanpa tf: auto
#
### LB   w/  tf:  0.67393 (tf cuma nambah +0.00419, kayanya ga usah)

X_fit, X_eval, y_fit, y_eval= train_test_split(eks, y_train, test_size=0.2, random_state=seed)

print 'Training Random Forest model...'
## Tune hyperparameter such as n_estimators, max_features, max_depth, min_samples_split for better cross-validation auc score
RFclassifier =RandomForestClassifier(n_estimators=649, max_features=85, max_depth=9, min_samples_split=3, random_state=seed, n_jobs=-1)
RFclassifier.fit(X_fit, y_fit)

## Predict evaluation sets
prf = RFclassifier.predict_proba(X_eval)[:,1]
print 'RandomForest: %f' % roc_auc_score(y_eval, prf)

# 0.690271
etc =ExtraTreesClassifier(n_estimators=649, max_features=85, max_depth=9, min_samples_split=4, random_state=seed, n_jobs=-1)
etc.fit(X_fit, y_fit)

## Predict evaluation sets
pred = etc.predict_proba(X_eval)[:,1]
print 'ExTrees: %f' % roc_auc_score(y_eval, pred)

from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(n_jobs=-1)
lr.fit(X_fit, y_fit)

## Predict evaluation sets
plr = lr.predict_proba(X_eval)[:,1]
print 'LogReg: %f' % roc_auc_score(y_eval, plr)

print 'Ensemble: %f' % roc_auc_score(y_eval, plr+prf+pred)

###
### SUBMISSION
###


X_test_=test.drop(object_columns, axis=1).drop('readmitted', axis=1)

#
#

tfidf=TfidfVectorizer(stop_words='english', ngram_range = (1,1)) # (1,2) sama aja kaya (1,1)
tfidf.fit(np.concatenate((train.diag_1_desc.astype(np.str),test.diag_1_desc.astype(np.str))))
test_diag_1_desc=tfidf.transform(test.diag_1_desc.astype(np.str))
train_diag_1_desc=tfidf.transform(train.diag_1_desc.astype(np.str))
ex=np.hstack((X_train, train_diag_1_desc.todense()))
#diag_2_desc=tfidf.fit_transform(test.diag_2_desc.astype(np.str))
#diag_3_desc=tfidf.fit_transform(test.diag_3_desc.astype(np.str))
X_test=np.hstack((X_test_, test_diag_1_desc.todense()))

RFclassifier = RandomForestClassifier(n_estimators=649, max_features=85, max_depth=9, min_samples_split=3, random_state=seed, n_jobs=-1)
RFclassifier.fit(ex, y_train)
yprob = RFclassifier.predict_proba(X_test)[:,1]

subs=pd.read_csv('sample_submission.csv')
subs.readmitted=yprob
subs.to_csv('subs.csv', index=False)

pickle.dump(RFclassifier, open('0.71219.p','wb'))