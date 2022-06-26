import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
# Just in case
import matplotlib.pyplot as plt
import seaborn as sns

# Disable pandas dataframe copy-ish warning
pd.options.mode.chained_assignment = None
# For reproducibility purpose
seed = 42
# SMOTE k neighbors
k = 3


def SVMF(X_train_in, y_train_in, X_test_in, y_test_in):
    svm_model = svm.SVC()
    svm_model.fit(X_train_in, np.ravel(y_train_in))
    predict = svm_model.predict(X_test_in)
    print(classification_report(y_test_in, predict))


def RF(X_train_in, y_train_in, X_test_in, y_test_in):
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train_in, np.ravel(y_train_in))  # ravel for reshape
    rf_prediction = rf_model.predict(X_test_in)
    print(classification_report(y_test_in, rf_prediction))


df = pd.read_csv('machine_failure_clean.csv', encoding='utf-8', engine='python')
print('## Original dataset ##')
print('Size {}'.format(df.shape[0]))
print(df['Machine failure'].value_counts(normalize=True))

X = df.loc[:, df.columns != 'Machine failure']
# Encode Type to numeric
X['Type'] = pd.Categorical(df['Type']).codes
y = df.loc[:, df.columns == 'Machine failure']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('## Synthetic Dataset ###')

# Random Over Sampler
ros = RandomOverSampler(random_state=seed)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
print('Random Over-sampling')
print('Size {}'.format(X_train_ros.shape[0]))
print(y_train_ros['Machine failure'].value_counts(normalize=True))

# SMOTE
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_sm, y_sm = sm.fit_resample(X_train, y_train)
print('Size {}'.format(X_sm.shape[0]))
print(y_sm['Machine failure'].value_counts(normalize=True))

# ADASYN
ada = ADASYN(random_state=seed)
x_ada, y_ada = ada.fit_resample(X_train, y_train)
print('ADASYN')
print('Size {}'.format(x_ada.shape[0]))
print(y_ada['Machine failure'].value_counts(normalize=True))

print('## RANDOM FOREST ##')
print('Baseline RF')
RF(X_train, y_train, X_test, y_test)
print('RF Random Over-sampling')
RF(X_train_ros, y_train_ros, X_test, y_test)
print('RF SMOTE')
RF(X_sm, y_sm, X_test, y_test)
print('RF ADASYN')
RF(x_ada, y_ada, X_test, y_test)

print('## SVM ##')
print('Baseline SVM')
SVMF(X_train, y_train, X_test, y_test)
print('SVM Random Over-sampling')
SVMF(X_train_ros, y_train_ros, X_test, y_test)
print('SVM SMOTE')
SVMF(X_sm, y_sm, X_test, y_test)
print('SVM ADASYN')
SVMF(x_ada, y_ada, X_test, y_test)
