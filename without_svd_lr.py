# AUC 0.698180

# Last Best
"""
AUC 0.695045 -> LB 0.71219 (old split)
"""
# Last Fail
"""
- AUC 0.699465 -> LB 0.71023 (old split)
- AUC 0.697585 -> LB 0.68375 (new split)
- AUC 0.697964 -> LB 0.70243
- AUC 0.6965   -> LB 0.70367
"""
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, RandomizedPCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
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
 'diag_3',
# 'weight'
 }
categorical=list(categorical_set)
###

train=pd.read_csv('SAStraining.csv')
test=pd.read_csv('SAStest.csv')

combined=pd.concat((train,test))#.reset_index()

combined.diag_1 = combined.diag_1.apply(ICD)
combined.diag_2 = combined.diag_2.apply(ICD)
combined.diag_3 = combined.diag_3.apply(ICD)

#berat=combined.weight.str.split('-').str[1].str[:-1].replace(np.nan, 0).astype(np.int16)
#berat[berat<=100]=0
#combined.weight=berat
#combined[combined.columns[24:49]]=pd.DataFrame([combined[i].apply({'No':0}.get).replace(np.NaN, 1) for i in combined.columns[24:49] ]).T.astype(np.bool)
#combined[combined.columns[24:49]]=pd.DataFrame([combined[i].apply({'No':0}.get).replace(np.NaN, 1) for i in combined.columns[24:49] ]).T.astype(np.bool)
_2449=pd.concat([pd.get_dummies(combined[i]).astype(np.bool) for i in combined.columns[24:49]], axis=1)

combined.gender = combined.gender.apply({'Male':1}.get).replace(np.NaN, 0).astype(np.bool)
combined.A1Cresult = combined.A1Cresult.apply({"Norm": -1, "None": 0, ">7": 1, ">8": 2}.get).replace(np.NaN, 0).astype(np.int8)
combined.age = combined.age.str.split('-').str[0].str[1:3].astype(np.bool)

#
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = CountVectorizer(analyzer=stemmed_words)

onehoted = pd.concat([pd.get_dummies(combined[i]).astype(np.bool) for i in categorical], axis=1)

tfidf = CountVectorizer(stop_words='english', ngram_range = (1,1), analyzer=stemmed_words, binary=1) # (1,2) sama aja kaya (1,1) sih kemarin
combined_diag_1_desc_ = tfidf.fit_transform(combined.diag_1_desc.astype(np.str).str.replace('unspecified','').str.replace('type',''))

#decomposer = TruncatedSVD(50, random_state=seed)
#combined_diag_1_desc = decomposer.fit_transform(combined_diag_1_desc_.toarray())
combined_diag_1_desc=combined_diag_1_desc_.toarray()
train_diag_1_desc = combined_diag_1_desc[:8000]
test_diag_1_desc  = combined_diag_1_desc[8000:]

for i in categorical:
    del combined[i]

# dates=combined.admissionDate.str.split('/', expand=True).astype(np.int16)
combined = pd.concat([combined, onehoted, _2449,
#                      dates
                      ], axis=1)
###
object_columns = combined.loc[:, combined.dtypes == object].columns

train = combined[:8000]
test  = combined[8000:]

X_train_ = train.drop(object_columns, axis=1).drop('No',1).drop('readmitted', axis=1)

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

X_test_ = test.drop(object_columns, axis=1).drop('readmitted', axis=1)
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