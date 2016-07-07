from __future__ import division
# AUC 0.703868 -> LB 0.70214 (gen 3)

# Last Best
"""
AUC 0.695045 -> LB 0.71219 (gen 1)
"""
# Last Fail
"""
- AUC 0.699465 -> LB 0.71023 (gen 1)
- AUC 0.697585 -> LB 0.68375 (gen 2)
- AUC 0.697964 -> LB 0.70243
- AUC 0.6965   -> LB 0.70367
- AUC 0.703868 -> LB 0.70214 (gen 3)
"""

import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, RandomizedPCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
seed = 1909 

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
    
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
    
    elif x[0] == 'V':
        v = int(x[1:])
        if    1 <= v <=  6: return 'Persons with potential health hazards related to communicable diseases'
        elif  7 <= v <=  9: return 'Persons with need for isolation, Other potential health hazards and Prophylactic measures'
        elif 10 <= v <= 19: return 'Persons with potential health hazards related to personal and family history'
        elif 20 <= v <= 29: return 'Persons encountering health services in Circumstances related to Reproduction and deelopment'
        elif 30 <= v <= 39: return 'Live-born infants according to type of birth'
        elif 40 <= v <= 49: return 'Persons with a condition influencing their health status'
        elif 50 <= v <= 59: return 'Persons encountering health services for specific procedures and aftercare'
        elif 60 <= v <= 69: return 'Persons encountering health services in other circumstances'
        elif 70 <= v <= 82: return 'Persons without reported diagnosis encountered during evamination and inestigation of indiiduals and populations'
        elif 83 <= v <= 84: return 'Genetics'
        elif 85: return 'Body mass indev'
        elif 86: return 'Estrogen receptor Status'
        elif 87: return 'Other Specified Personal Evposures And History Presenting Hazards To Health'
        elif 88: return 'Acquired Absence Of Other Organs And Tissue'
        elif 89: return 'Other Suspected Conditions Not Found'
        elif 90: return 'Retained Foreign Body'
        elif 91: return 'Multiple Gestation Placenta Status'

    elif x[0] == 'E':
        e = int(x[1:])
        if   800 <= e <= 807: return 'Railway accidents'
        elif 810 <= e <= 819: return 'Motor vehicle traffic accidents'
        elif 820 <= e <= 825: return 'Motor vehicle non-traffic accidents'
        elif 826 <= e <= 829: return 'Other road vehicle accidents'
        elif 830 <= e <= 832: return 'Water transport accidents'
        elif 840 <= e <= 845: return 'Air and space transport accidents'
        elif 846 <= e <= 848: return 'Vehicle accidents not elsewhere classifiable'
        elif        e == 849: return 'Place of Occurrence'
        elif 850 <= e <= 858: return 'Accidental poisoning by drugs, medicinal substances, and biologicals'
        elif 860 <= e <= 869: return 'Accidental poisoning by other solid and liquid substances, gases, and vapors'
        elif 870 <= e <= 876: return 'Misadventures to patients during surgical and medical care'
        elif 878 <= e <= 879: return 'Surgical and medical procedures as the cause of abnormal reaction of patient or later complication, without mention of misadventure at the time of procedure'
        elif 880 <= e <= 888: return 'Accidental falls'
        elif 890 <= e <= 899: return 'Accidents caused by fire and flames'
        elif 900 <= e <= 909: return 'Accidents due to natural and environmental factors'
        elif 910 <= e <= 915: return 'Accidents caused by submersion, suffocation, and foreign bodies'
        elif 916 <= e <= 928: return 'Other accidents'
        elif        e == 929: return 'Late effects of accidental injury'
        elif 930 <= e <= 949: return 'Drugs, medicinal and biological substances causing adverse effects in therapeutic use'
        elif 950 <= e <= 959: return 'Suicide and self-inflicted injury'
        elif 960 <= e <= 969: return 'Homicide and injury purposely inflicted by other persons'
        elif 970 <= e <= 978: return 'Legal intervention'
        elif        e == 979: return 'Terrorism'
        elif 980 <= e <= 989: return 'Injury undetermined whether accidentally or purposely inflicted'
        elif 990 <= e <= 999: return 'Injury resulting from operations of war'
    else:
        return x

train = pd.read_csv('SAStraining.csv')
test  = pd.read_csv('SAStest.csv')

combined = pd.concat((train,test))

combined.diag_1 = combined.diag_1.apply(ICD)
combined.diag_2 = combined.diag_2.apply(ICD)
combined.diag_3 = combined.diag_3.apply(ICD)

combined.A1Cresult = combined.A1Cresult.apply({"Norm": -1, "None": 0, ">7": 1, ">8": 2}.get).replace(np.NaN, 0).astype(np.int8)
combined.age = combined.age.str.split('-').str[0].str[1:3].astype(np.bool)

categorical = set(combined.loc[:, combined.dtypes == object].columns) - set(['diag_1_desc','diag_2_desc','diag_3_desc', 'admissionDate','patientID']); categorical

onehoted = pd.concat([pd.get_dummies(combined[i]).astype(np.bool) for i in categorical], axis=1)

tfidf = CountVectorizer(stop_words='english', ngram_range = (1,1), analyzer=stemmed_words, binary=1) # (1,2) sama aja kaya (1,1) sih kemarin
combined_diag_1_desc_ = tfidf.fit_transform(combined.diag_1_desc.astype(np.str).str.replace('unspecified','').str.replace('type',''))

# Decomposition with SVD or RandomizedPCA

#decomposer = TruncatedSVD(50, random_state=seed)
#combined_diag_1_desc = decomposer.fit_transform(combined_diag_1_desc_.toarray())
combined_diag_1_desc = combined_diag_1_desc_.toarray()
train_diag_1_desc = combined_diag_1_desc[:8000]
test_diag_1_desc  = combined_diag_1_desc[8000:]

for i in categorical:
    del combined[i]

# dates=combined.admissionDate.str.split('/', expand=True).astype(np.int16)
combined = pd.concat([combined, onehoted,
#                      dates
                      ], axis=1)
###
object_columns = combined.loc[:, combined.dtypes == object].columns

train = combined[:8000]
test  = combined[8000:]

'None'
X_train_ = train.drop(object_columns, axis=1).drop(['?','No'],1).drop('readmitted', axis=1)

X_train = np.hstack((X_train_, train_diag_1_desc))
y_train = train.readmitted


###
### EVALUATION
### 

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

# 0.69
lr =LogisticRegression(n_jobs=-1, C=0.1)
#LogisticRegression()
lr.fit(X_fit, y_fit)
plr = lr.predict_proba(X_eval)[:,1]
print 'LogReg: %f' % roc_auc_score(y_eval, plr)


###
### SUBMISSION
###

'None'
X_test_ = test.drop(object_columns, axis=1).drop(['?','No'],1).drop('readmitted', axis=1)
X_test  = np.hstack((X_test_, test_diag_1_desc))

# RFclassifier sama dengan yang atas
model =LogisticRegression(n_jobs=-1, C=0.1)
model.fit(X_train, y_train)
yprob = model.predict_proba(X_test)[:,1]

subs = pd.read_csv('sample_submission.csv')
subs.readmitted = yprob
subs.to_csv('no_svd.csv', index=False)

#pickle.dump(RFclassifier, open('0.71219.p','wb'))
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud


wordcloud = WordCloud(max_font_size=20, relative_scaling=.4).generate(" ".join(train.diag_1_desc[train.readmitted==0].astype(np.str)))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.axis("off")

plt.show()
"""