#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:40:47 2017

@author: standarduser
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

data_train = pd.read_csv("train.csv")
data_test  = pd.read_csv("test.csv")


#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)

#sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
#              palette={"male":"blue", "female":"green"},
#              markers=["*","o"], linestyles=["-","--"])

# Preprocessing of Feature Data 

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins   = (-1,0,5,12,18,25,35,60,120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult',
                   'Adult', 'Senior']
    age = pd.cut(df.Age, bins, labels = group_names)
    df.Age = age
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x:x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1,0,8,15,31,1000)
    group_names = ['Unknown', '1st_rate', '2nd_rate', '3rd_rate', '4th_rate']
    fare  = pd.cut(df.Fare, bins, labels = group_names)
    df.Fare = fare
    return df

def name_formate(df):
    df['Last_name'] = df.Name.apply(lambda x:x.split(' ')[0])
    df['Gender'] = df.Name.apply(lambda x:x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)
    
def cleaning_data(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = name_formate(df)
    df = drop_features(df)
    return df

data_train = cleaning_data(data_train)
data_test = cleaning_data(data_test)
#print data_train.head()

#sns.barplot(x='Age', y='Survived', hue="Sex", data = data_train)
#sns.barplot(x='Cabin', y='Survived', hue="Sex", data = data_train)
#sns.barplot(x='Fare', y='Survived', hue="Sex", data = data_train)


# Converting String data to numeric data , So that become easy for machine to read

from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Last_name', 'Gender']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

data_train, data_test = encode_features(data_train, data_test)
#print data_train.head()

# Machine Learnig Part starts here ....
# Spliting data into training and testing further training feature and label

X = data_train.drop(['Survived', 'PassengerId'] ,1)
y = data_train['Survived']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=23)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

clf  = RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

scoring_patt = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scoring_patt)
grid_obj = grid_obj.fit(X_train, y_train)
clf = grid_obj.best_estimator_

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print accuracy_score(y_test, prediction)

from sklearn.cross_validation import KFold

def runKTimes(clf):
    kf = KFold(891, n_folds=10)
    turn = 0
    accuracy = []
    for train_index, test_index in kf:
        turn += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        res = accuracy_score(y_test,predict)
        accuracy.append(res)
        print "fold :{0}  accuracy {1}".format(turn, res)
    mean = np.mean(accuracy)
    print "Mean Accuracy : {}".format(mean)

runKTimes(clf)

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId' : ids, 'Survived' : predictions})

output.to_csv('titanic-predictions.csv', index = False)