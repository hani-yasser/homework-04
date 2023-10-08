#!/usr/bin/env python
# coding: utf-8

# Data from https://www.kaggle.com/blastchar/telco-customer-churn

# In[144]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[145]:


df = pd.read_csv("cars.csv")


# In[146]:


len(df)


# ## Initial data preparation

# In[147]:


df.head()


# In[148]:


df.head().T


# In[149]:


df.columns


# In[150]:


Features_Cat=['Make','Model','Year','Engine HP','Engine Cylinders','Transmission Type','Vehicle Style','highway MPG','city mpg','MSRP']


# In[151]:


data=df[Features_Cat]


# In[152]:


data.head()


# In[153]:


data.columns = data.columns.str.replace(' ', '_').str.lower()


# In[154]:


data.head()


# In[155]:


data= data.fillna(0)


# In[156]:


data.isnull().sum()


# In[157]:


data.rename(columns={'msrp': 'Price'}, inplace=True)


# In[158]:


Price_mean=data['Price'].mean()


# In[159]:


data['above_average'] = (data['Price'] > Price_mean).astype(int)


# Split the DATA

# In[160]:


from sklearn.model_selection import train_test_split


# In[161]:


df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)


# In[162]:


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


# In[163]:


len(data),len(df_train),len(df_val),len(df_test)


# In[164]:


df_train.shape


# In[165]:


y_train = df_train.above_average.values


# In[166]:


y_val = df_val.above_average.values


# In[167]:


df_train.columns


# In[168]:


df_train = df_train.drop(columns=['Price','above_average'])


# In[169]:


y_val = df_val.above_average.values


# ## Question 1

# In[170]:


numerical=['year','engine_hp','highway_mpg','city_mpg']
categorical = ['make', 'model','transmission_type','vehicle_style']


# In[171]:


auc = roc_auc_score(y_train, df_train['year'])


# In[172]:


auc


# In[173]:


for c in numerical:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print('%9s, %.3f' % (c, auc))


# ## Question 2

# In[174]:


columns = categorical + numerical

train_dicts = df_train[columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[columns].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]
     

#@ INSPECTING ROC AUC SCORE:
roc_auc_score(y_val, y_pred)


# ## Question 3

# In[175]:


def confusion_matrix_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)
    
    return df_scores


# In[176]:


df_scores = confusion_matrix_dataframe(y_val, y_pred)
df_scores[::10]
     


# In[177]:


df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)


# In[178]:


df_scores[::10]


# ### Different Method to justify results

# In[179]:


threshold = 0.5
y_pred_int = (y_pred >= threshold).astype(int)
y_pred_int


# In[180]:


print(classification_report(y_val,y_pred_int))


# In[181]:


print(confusion_matrix(y_val,y_pred_int))


# In[182]:


plt.plot(df_scores.threshold, df_scores.p, label='precision')
plt.plot(df_scores.threshold, df_scores.r, label='recall')

plt.legend()
plt.show()


# ## Question 4

# In[183]:


df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)


# In[184]:


df_scores[::10]


# In[185]:


plt.plot(df_scores.threshold, df_scores.f1)
plt.xticks(np.linspace(0, 1, 11))
plt.show()


# In[186]:


df_train_full


# ## Question 5 

# In[187]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[190]:


scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]

    y_train = df_train.above_average
    y_val = df_val.above_average

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# ## Question 6

# In[194]:


kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 0.5, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.above_average
        y_val = df_val.above_average

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

