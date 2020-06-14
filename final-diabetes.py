#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:


data = pd.read_csv('diabates_clean.csv')


# In[3]:


data.columns


# In[4]:


data = data.drop(['Unnamed: 0'], axis= 1)


# In[5]:


data.head()


# In[6]:


data.corr()


# In[7]:


array = data.values

X = array[:,0:8]

y = array[:,8] # dv
test_size = 0.30
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
print('Partitioning Done!')


# In[8]:


from sklearn.linear_model import LogisticRegression
logreg_clf = LogisticRegression(max_iter =1000)
logreg_clf.fit(X_train, y_train)
prediction = logreg_clf.predict(X_test)
outcome = y_test
print(accuracy_score(outcome,prediction))


# In[9]:


#Using Adaboost Classifier
tuned_parameters = [{'n_estimators': [100, 200, 500]}]
logreg_clf = LogisticRegression(max_iter=1000)
ada_clf = AdaBoostClassifier(logreg_clf)
clf = GridSearchCV(ada_clf,tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(X_train, y_train )
print(clf.best_score_)


# In[10]:


print(clf.best_params_)


# In[11]:

#validation for better parameters
tuned_parameters = [{'max_samples': [5,7,10],'n_estimators': [10,20,100],'max_features': [0.2,0.6,1.0]}]
logreg_clf = LogisticRegression(max_iter=1000)
bag_clf = BaggingClassifier(logreg_clf)
clf = GridSearchCV(bag_clf,tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(X_train,y_train)
print(clf.best_score_)
print(clf.best_params_)


# In[25]:


#Bagging Classifier
logreg_clf = LogisticRegression(max_iter=1000)
bag_clf = BaggingClassifier(logreg_clf, n_estimators = 100, max_features = 0.6, max_samples = 10 )
bag_clf.fit( X_train, y_train )
bag_test_pred = pd.DataFrame( { 'actual': y_test,'predicted': bag_clf.predict( X_test ) } )
predict_proba_df = pd.DataFrame( bag_clf.predict_proba( X_test ) )
predict_proba_df.head()


# In[26]:


bag_test_pred = bag_test_pred.reset_index()
bag_test_pred['chd_0'] = predict_proba_df.iloc[:,0:1]
bag_test_pred['chd_1'] = predict_proba_df.iloc[:,1:2]
bag_test_pred[0:10]


# In[27]:

#plotting and comparing 
sns.distplot( bag_test_pred[bag_test_pred.actual == 1]["chd_1"], kde=False, color = 'b' )
sns.distplot( bag_test_pred[bag_test_pred.actual == 0]["chd_1"], kde=False, color = 'g' )


# In[28]:

#dumping as pilckle
import pickle


# In[29]:


file='diabetes-preediction-model.pkl'
pickle.dump(bag_clf,open(file,'wb'))


# In[ ]:




