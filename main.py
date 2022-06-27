import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
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
    svm_model.fit(X_train_in, np.ravel(y_train_in))  # ravel for reshape
    predict = svm_model.predict(X_test_in)
    print(classification_report(y_test_in, predict))


def RF(X_train_in, y_train_in, X_test_in, y_test_in):
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train_in, np.ravel(y_train_in))  # ravel for reshape
    rf_prediction = rf_model.predict(X_test_in)
    print(classification_report(y_test_in, rf_prediction))


df = pd.read_csv('machine_failure_clean.csv', encoding='utf-8', engine='python')
class_names = ['Normal Condition', 'Defective condition']
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


def evaluate_model(fold, data_x, data_y):
    k_fold = KFold(fold, shuffle=True, random_state=seed)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for thisTrain, thisTest in k_fold.split(data_x):
        # print('DEBUG {}'.format(thisTrain))
        train_x, train_y, test_x, test_y = data_x.iloc[thisTrain], data_y[thisTrain], data_x.iloc[thisTest], data_y[thisTest]

        # Fit the classifier
        #classifier = svm.SVC().fit(train_x, train_y)
        classifier = RandomForestClassifier().fit(train_x, train_y)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()


predicted_target, actual_target = evaluate_model(2, X_sm, np.ravel(y_sm))
plot_confusion_matrix(predicted_target, actual_target)

## Cross-validation
# print('## RANDOM FOREST ##')
# print('Baseline RF')
# RF(X_train, y_train, X_test, y_test)
# print('RF Random Over-sampling')
# RF(X_train_ros, y_train_ros, X_test, y_test)
# print('RF SMOTE')
# RF(X_sm, y_sm, X_test, y_test)
# print('RF ADASYN')
# RF(x_ada, y_ada, X_test, y_test)
#
# print('## SVM ##')
# print('Baseline SVM')
# SVMF(X_train, y_train, X_test, y_test)
# print('SVM Random Over-sampling')
# SVMF(X_train_ros, y_train_ros, X_test, y_test)
# print('SVM SMOTE')
# SVMF(X_sm, y_sm, X_test, y_test)
# print('SVM ADASYN')
# SVMF(x_ada, y_ada, X_test, y_test)