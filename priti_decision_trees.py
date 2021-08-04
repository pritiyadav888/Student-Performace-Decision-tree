#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:19:39 2021

@author: pyadav
"""

# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import scipy.stats as stats
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import FitFailedWarning


#load
data_priti = pd.read_csv('/Users/ayadav/Downloads/student/student-por.csv', sep = ';')

#initial investigations

data_priti.head()
print(data_priti.info())
print(data_priti.columns)
print(data_priti.isnull().sum())
print(data_priti.describe())

# Get list of categorical variables
s = (data_priti.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
data_priti.select_dtypes(include=['object'])

target = 'G3'
sns.distplot(data_priti[target])
# plt.savefig('target.pdf')
plt.show()

#Visualization
"""
sns.catplot(x="school",y="G3", kind="box",data=data_priti)
sns.catplot(x="sex",y="G3",kind="box",data=data_priti)
sns.catplot(x="age",y="G3",kind="bar",data=data_priti)
sns.catplot(x="address",y="G3", kind="box",data=data_priti)
sns.catplot(x="famsize",y="G3", kind="box",data=data_priti)
sns.catplot(x="Pstatus",y="G3", kind="box",data=data_priti)
sns.catplot(x="Medu",y="G3", kind="box",data=data_priti)
sns.catplot(x="Fedu",y="G3", kind="box",data=data_priti)
sns.catplot(x="Mjob",y="G3", kind="box",data=data_priti)
sns.catplot(x="Fjob",y="G3", kind="box",data=data_priti)
sns.catplot(x="reason",y="G3", kind="box",data=data_priti)
sns.catplot(x="guardian",y="G3", kind="box",data=data_priti)
sns.catplot(x="traveltime",y="G3", kind="box",data=data_priti)
sns.catplot(x="studytime",y="G3", kind="box",data=data_priti)
sns.catplot(x="studytime",y="G3", kind="box",data=data_priti)
sns.catplot(x="failures",y="G3", kind="box",data=data_priti)
sns.catplot(x="schoolsup",y="G3", kind="box",data=data_priti)
sns.catplot(x="famsup",y="G3", kind="box",data=data_priti)
sns.catplot(x="paid",y="G3", kind="box",data=data_priti)
sns.catplot(x="activities",y="G3", kind="box",data=data_priti)
sns.catplot(x="nursery",y="G3", kind="box",data=data_priti)
sns.catplot(x="higher",y="G3", kind="box",data=data_priti)
sns.catplot(x="internet",y="G3", kind="box",data=data_priti)
sns.catplot(x="romantic",y="G3", kind="box",data=data_priti)
sns.catplot(x="famrel",y="G3", kind="box",data=data_priti)
sns.catplot(x="freetime",y="G3", kind="box",data=data_priti)
sns.catplot(x="Dalc",y="G3", kind="box",data=data_priti)
sns.catplot(x="goout",y="G3", kind="box",data=data_priti)
sns.catplot(x="Walc",y="G3", kind="box",data=data_priti)
sns.catplot(x="health",y="G3", kind="box",data=data_priti)

"""
#Pre-process and preparethe datafor machine learning

data_priti[['school','G3']].groupby(data_priti['school'],as_index=True).mean()
data_priti[['school','G3']].groupby(data_priti['school'],as_index=True).std()
data_priti['school'].value_counts()

#data_priti['pass_priti'] = np.select([data_priti.G1 >= 35 , data_priti.G2 >= 35, data_priti.G3 >= 35], 1, 0)
total = data_priti['G1']+ data_priti['G2']+ data_priti['G3']
data_priti['pass_priti'] = np.where((total >= 35),1,0)
data_priti =data_priti.drop(['G1','G2','G3'], axis = 1)
print(data_priti.columns)

features_priti= data_priti.loc[:, data_priti.columns != 'pass_priti']
#print(features_priti.columns)
target_variable_priti = data_priti['pass_priti']
#print(target_variable_priti.columns)

#imbalances
features_priti.count()
target_variable_priti.count()
print(target_variable_priti.value_counts(normalize= True))
features_priti.apply(pd.Series.value_counts)
for (columnName, columnData) in features_priti.iteritems(): 
   print('\nColunm Name : ', columnName)
   print('Column Contents \n: ', columnData.value_counts(normalize= True))
   
#seperating features Categorical & Numerical
cat_features_priti = (features_priti.select_dtypes(exclude=[np.number]))
numeric_features_priti = (features_priti.select_dtypes(include=[np.number]))
print('Categorical features are -\n',cat_features_priti)
#print('Numeric features are -\n',numeric_features_priti)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#df.select_dtypes(include=['float64'])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#cat_features_priti + numeric_features_priti

transformer_priti = ColumnTransformer(transformers=[('cat_features_priti', OneHotEncoder(), [0, 1])])


#transformer_priti = ColumnTransformer([("transformer",'cat_features_priti', OneHotEncoder(), [0, 1])])

clf_priti = DecisionTreeClassifier(criterion='entropy',  max_depth=5, random_state = 74)


from sklearn.pipeline import Pipeline
pipeline_priti = Pipeline([('One_hot_cat',transformer_priti),
  ('decision_clf',clf_priti)])


from sklearn.model_selection import train_test_split

X_train_priti, X_test_priti, y_train_priti, y_test_priti = train_test_split(cat_features_priti, target_variable_priti, test_size = .20, random_state = 40)

pipeline_priti.fit(X_train_priti, y_train_priti)
y_predict = pipeline_priti.predict(X_test_priti)
#y_pred = pipeline_priti.predict(GP, F, U, GT3,)
importances = clf_priti.feature_importances_
print(importances)

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=74)

print(crossvalidation)

# enumerate splits
cnt = 1
for train, test in crossvalidation.split(data_priti):
    print(f'Fold:{cnt}, Train set: {len(train)}, Test set:{len(test)}')
    cnt += 1
    
def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')
    
    
score = cross_val_score(pipeline_priti, cat_features_priti,target_variable_priti, cv= crossvalidation, scoring="neg_mean_squared_error")
print(f'Scores for each fold: {score}')
print(f'Scores for each fold in positive: {abs(score)}')
# summarize the model performance
print('MAE: %.3f (%.3f)' % ((score.mean()), score.std()))

rmse(score.mean())   


tree.plot_tree(clf_priti)
from sklearn.metrics import roc_auc_score

score_decision = pipeline_priti.score(X_test_priti, y_test_priti)
print(f'Model accuracy is:{score_decision*100} %')
print(clf_priti.get_params())
baseline_auc = roc_auc_score(y_test_priti, y_predict)

from sklearn.model_selection import RandomizedSearchCV
parameters={'decision_clf__min_samples_split' : range(10,300,20),'decision_clf__max_depth': range(1,30,2),'decision_clf__min_samples_leaf':range(1,15,3)}



#Fine tune model

fine_tune = RandomizedSearchCV(estimator = pipeline_priti, cv = 5, param_distributions = parameters, 
                                scoring='accuracy', n_iter = 7,verbose=3, refit = True)
start = time()
n_iter = 7
fine_tune.fit(X_train_priti, y_train_priti)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter))
report(fine_tune.cv_results_)

print ('Best Parameters for RandomizedSearchCV : ', fine_tune.best_params_, ' \n')


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
print(f'\nCalculation For Model',print_score(pipeline_priti, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=True))
print(f'\nCalculation For Model',print_score(pipeline_priti, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=False) ) 


print(f'\nCalculation For RandomizedSearchCv',print_score(fine_tune, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=True))
print(f'\nCalculation For RandomizedSearchCv',print_score(fine_tune, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=False) )       
      

#fine tune model with grid search
from sklearn.model_selection import GridSearchCV
C = [1.0,1.5,2.0,2.5]
param_grid={'decision_clf__min_samples_split' : range(10,300,20),'decision_clf__max_depth': range(1,30,2),'decision_clf__min_samples_leaf':range(1,15,3)}
grid_search_cv = GridSearchCV(estimator = pipeline_priti, cv = 5, param_grid= param_grid, 
                                scoring='accuracy', verbose=3, refit = True)
start = time()

grid_search_cv.fit(X_train_priti, y_train_priti)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_cv.cv_results_['params'])))
report(grid_search_cv.cv_results_)

print(grid_search_cv.best_score_)
##issue

print(f'\nCalculation For GridSearchCv',print_score(grid_search_cv, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=True))
print(f'\nCalculation For GridSearchCv',print_score(grid_search_cv, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=False) )       
      


from sklearn.ensemble import RandomForestClassifier

rf_clf_priti = RandomForestClassifier(criterion='entropy',  max_depth=5, random_state = 42)


from sklearn.pipeline import Pipeline
pipeline_random = Pipeline([('One_hot_cat',transformer_priti),
  ('decision_clf',rf_clf_priti)])

pipeline_random.fit(X_train_priti, y_train_priti)


print_score(pipeline_random, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=True)
print_score(pipeline_random, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=False)        


#rom sklearn.metrics import classification_report
#randomized_grid_predictions = fine_tune.predict(X_test_priti)
#print(fine_tune.cv_results_)
#print(fine_tune.best_params_)
#print(fine_tune.best_score_)

#graphviz
import pydotplus
import graphviz
import collections
dot_data=tree.export_graphviz(clf_priti,
                                out_file=None,
                                filled=True,
                                rounded=True)
text_representation = tree.export_text(clf_priti)
print(text_representation)

graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')


import joblib
#import pickle
joblib.dump(fine_tune.best_params_,"decision_classifier_model.pkl")

save_pipeline = joblib.dump(pipeline_priti,open("decision_classifier_model.pkl", 'wb'))


#After selecting few columns as Features
students = data_priti[['age','studytime','failures','absences']]
students.head()
students.corr()
X_train_priti, X_test_priti, y_train_priti, y_test_priti = train_test_split(students,target_variable_priti, test_size = .20, random_state = 74)

clf_priti = DecisionTreeClassifier(criterion='entropy',  max_depth=5, random_state = 74)
clf_priti.fit(X_train_priti, y_train_priti)
y_predict = clf_priti.predict(X_test_priti)
print_score(clf_priti, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=True)
print_score(clf_priti, X_train_priti, y_train_priti, X_test_priti, y_test_priti, train=False)
print(clf_priti.feature_importances_)

