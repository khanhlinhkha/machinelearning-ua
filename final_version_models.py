import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV

"""
Load the data that has been preprocessed
"""

file = 'Preprocessed_data.xlsx'
df_copy_3=pd.read_excel(file, index_col=0)
converted=pd.read_excel(file, sheet_name='converted', index_col=0)
indices_train=pd.read_excel(file, sheet_name='indices train', index_col=0)
indices_val=pd.read_excel(file, sheet_name='indices val', index_col=0)
indices_test=pd.read_excel(file, sheet_name='indices test', index_col=0)

indices_train_list = indices_train[0].tolist()
indices_test_list = indices_test[0].tolist()
indices_val_list = indices_val[0].tolist()

# Create train, validation, and test data

X_train = df_copy_3.iloc[indices_train_list]
X_test = df_copy_3.iloc[indices_test_list]
X_val = df_copy_3.iloc[indices_val_list]

Y_train = converted.iloc[indices_train_list]
Y_test = converted.iloc[indices_test_list]
Y_val = converted.iloc[indices_val_list]

X_train_total = pd.concat([X_train, X_val])  # Concatenate two data frames
Y_train_total = pd.concat([Y_train, Y_val])  # Concatenate two data frames

"""
Lift function
"""
def liftatp(scores, Y_test, P):
    ratio = np.size(np.argwhere(np.ravel(Y_test) == 1)) / np.size(Y_test)

    sorted_indices = np.argsort(-scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = np.ravel(Y_test)[sorted_indices]

    topN = np.floor(np.size(Y_test) * P)
    topN_labels = sorted_labels[:int(topN)]
    topN_numberpos = np.size(np.argwhere(topN_labels == 1))

    lift = (topN_numberpos / topN) / ratio

    return lift

"""
KNN
"""
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier
knn = KNeighborsClassifier()

# Define the parameter grid to search
param_grid = {
    'n_neighbors': [100,200,300],  # tried [100,500,1000,5000,10000] --> Best Parameters:  {'metric': 'euclidean', 'n_neighbors': 100, 'weights': 'distance'} --> then tried [100,200,300]
    'weights': ['uniform', 'distance'],  # Different weighting schemes
    'metric': ['euclidean', 'manhattan']  # Different distance metrics
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_) # Best Parameters:  {'metric': 'euclidean', 'n_neighbors': 100, 'weights': 'distance'}

# KNN with optimal number of neighbors and weight set to inverse distance
clf_knn = KNeighborsClassifier(n_neighbors = 100, weights = 'distance', metric = 'euclidean')
clf_knn.fit(X_train, Y_train)

# make predictions on validation set
y_val_labels_knn = clf_knn.predict(X_val)
y_val_scores_knn = clf_knn.predict_proba(X_val)[:,1]

# calculate the accuracy on validation data
accuracy_val_knn = accuracy_score(Y_val, y_val_labels_knn)
print("Validation Accuracy with Best Model: ", accuracy_val_knn)

# AUC value on validation data
AUC_val_knn = roc_auc_score(Y_val, y_val_scores_knn)
print("Validation AUC with Best Model: ", AUC_val_knn)

#make predictions on test set
y_test_labels_knn = clf_knn.predict(X_test)
y_test_scores_knn = clf_knn.predict_proba(X_test)[:,1]

# calculate the accuracy on validation data
accuracy_test_knn = accuracy_score(Y_test, y_test_labels_knn)
print("Test Accuracy with Best Model: ",accuracy_test_knn)

# AUC value on test data
AUC_test_knn = roc_auc_score(Y_test, y_test_scores_knn)
print("Test AUC with Best Model: ",AUC_test_knn)

# make confusion matrix for test set
cf_knn = confusion_matrix(Y_test, y_test_labels_knn)
print("Confusion matrix of test set:\n{}".format(cf_knn))

# draw ROC curve
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_test_scores_knn)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on test data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC KNN test = %0.4f' %roc_auc) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()

"""
Decision tree
"""
# train decision tree on training data
from sklearn import tree

# loop for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

clf_dt = tree.DecisionTreeClassifier()

min_samples_leaf_range = [50,100,200] # tried [100,300,500,700,1000,5000,10000] --> 100 --> then tried [50,100,200] --> 200
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15,20],
    'min_samples_split': [2, 5, 10,20,50],
    'min_samples_leaf': min_samples_leaf_range
}

gridsearch = GridSearchCV(clf_dt, param_grid, cv = 5)
gridsearch.fit(X_train , Y_train)

# select best model
print ("Best model is: " + str(gridsearch.best_params_))
# Best model is: {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 200, 'min_samples_split': 10}

# Given the optimal leaf size, rebuild/retrain tree.
clf_dt = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=200, max_depth = None,  min_samples_split = 10)
clf_dt.fit(X_train, Y_train)

# make predictions on validation set
y_val_labels_dt = clf_dt.predict(X_val)
y_val_scores_dt = clf_dt.predict_proba(X_val)[:,1]

# calculate the accuracy on validation data
accuracy_val_dt = accuracy_score(Y_val, y_val_labels_dt)
print("Validation Accuracy with Best Model: ",accuracy_val_dt)

# AUC value on validation data
AUC_val_dt = roc_auc_score(Y_val, y_val_scores_dt)
print("Validation AUC with Best Model: ",AUC_val_dt)

# make predictions on test set
y_test_labels_dt = clf_dt.predict(X_test)
y_test_scores_dt = clf_dt.predict_proba(X_test)[:,1]

# calculate the accuracy on test data
accuracy_test_dt = accuracy_score(Y_test, y_test_labels_dt)
print("Test Accuracy with Best Model: ",accuracy_test_dt)

# AUC value on test data
AUC_test_dt = roc_auc_score(Y_test, y_test_scores_dt)
print("Test AUC with Best Model: ",AUC_test_dt)

# make confusion matrix for test set
cf_dt = confusion_matrix(Y_test, y_test_labels_dt)
print("Confusion matrix of test set:\n{}".format(cf_dt))

# ROC curve
# calculate false positive and true positive rate
fpr, tpr, threshold = roc_curve (Y_test, y_test_scores_dt)
roc_auc = metrics.auc(fpr, tpr)

# plot fpr and tpr
plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label = 'AUC DT test = %0.4f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--') # plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') # y- label
plt.xlabel('False Positive Rate') # x- label
plt.show()

# get tree as pdf:
from sklearn.tree import export_graphviz

dot_data = export_graphviz(
    clf_dt,
    out_file=None,
    feature_names=X_train.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)

with open("decision_tree.dot", "w") as dot_file:
    dot_file.write(dot_data)

import subprocess
subprocess.call(['dot', '-Tpdf', 'decision_tree.dot', '-o', 'decision_tree.pdf'])

"""
Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rfc = RandomForestClassifier()

# Tuning
n_estimators = [100,500,1000] # tried [100,500,1000] --> 500
param_grid = {'n_estimators': n_estimators}

gridsearch = GridSearchCV(rfc, param_grid, cv = 5)
gridsearch.fit(X_train , Y_train)

# select best model
print("Best model is: " + str(gridsearch.best_params_)) #  {'n_estimators': 500}

# Given the optimal estimators, rebuild/retrain model.
rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(X_train, Y_train)

# make predictions on validation set
y_val_labels_rf = rfc.predict(X_val)
y_val_scores_rf = rfc.predict_proba(X_val)[:,1]

# calculate the accuracy on validation data
accuracy_val_rf = accuracy_score(Y_val, y_val_labels_rf)
print("Validation Accuracy with Best Model: ",accuracy_val_rf)

# AUC value on validation data
AUC_val_rf = roc_auc_score(Y_val, y_val_scores_rf)
print("Validation AUC with Best Model: ",AUC_val_rf)

# make predictions on test set
y_test_labels_rf = rfc.predict(X_test)
y_test_scores_rf = rfc.predict_proba(X_test)[:,1]

# calculate the accuracy on test data
accuracy_test_rf = accuracy_score(Y_test, y_test_labels_rf)
print("Test Accuracy with Best Model: ",accuracy_test_rf)

# AUC value on test data
AUC_test_rf = roc_auc_score(Y_test, y_test_scores_rf)
print("Test AUC with Best Model: ",AUC_test_rf)

# make confusion matrix for test set
cf_rf = confusion_matrix(Y_test, y_test_labels_rf)
print("Confusion matrix of test set:\n{}".format(cf_rf))

# ROC curve
# calculate false positive and true positive rate
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_rf)
roc_auc = metrics.auc(fpr, tpr)

# plot fpr and tpr
plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label = 'AUC RF test = %0.4f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--') # plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') # y- label
plt.xlabel('False Positive Rate') # x- label
plt.show()

"""
Nonlinear SVM model
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create an SVM model
svm_model = SVC()

param_grid = {
    'gamma': [2**(-20), 2**(-15),2**(-10),2**(-5),2**(0),2**(5)],
    'C': [2**(-20), 2**(-15),2**(-10),2**(-5),2**(0),2**(5)],
    'kernel': ['rbf']}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# select best model
print("Best model is: " + str( grid_search.best_estimator_ ))

#Retrain model with best C-value
best_SVM_nonlin = SVC(C = 1.0, gamma = 0.03125, kernel = 'rbf') #input best parameters here
best_SVM_nonlin.fit(X_train, np.ravel(Y_train))

# make predictions on validation set
y_val_labels_SVM_nonlin = best_SVM_nonlin.predict(X_val)
y_val_scores_SVM_nonlin = best_SVM_nonlin.decision_function(X_val)

# calculate the accuracy on validation data
accuracy_val_SVM_nonlin = accuracy_score(Y_val, y_val_labels_SVM_nonlin)
print("Validation Accuracy with Best Model: ",accuracy_val_SVM_nonlin)

# AUC value on validation data
AUC_val_SVM_nonlin = roc_auc_score(Y_val, y_val_scores_SVM_nonlin)
print("Validation AUC with Best Model: ",AUC_val_SVM_nonlin)

# Compute the scores for the test instances
y_test_scores_SVM_nonlin = best_SVM_nonlin.decision_function(X_test)
y_test_labels_SVM_nonlin = best_SVM_nonlin.predict(X_test)

# Accuracy on test data
accuracy_test_SVM_nonlin = accuracy_score(Y_test, y_test_labels_SVM_nonlin)
print("Test Accuracy with Best Model: ",accuracy_test_SVM_nonlin)

# AUC value on test data
AUC_test_SVM_nonlin = roc_auc_score(Y_test, y_test_scores_SVM_nonlin)
print("Test AUC with Best Model: ",AUC_test_SVM_nonlin)

# make confusion matrix for test set
cf_SVM_nonlin = confusion_matrix(Y_test, y_test_labels_SVM_nonlin)
print("Confusion matrix of test set:\n{}".format(cf_SVM_nonlin))

# ROC curve
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_test_scores_SVM_nonlin)
roc_auc_SVM_nonlin = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on test data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC nonlin SVM test = %0.4f' %roc_auc_SVM_nonlin) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()

"""
Logistic Regression
"""
from sklearn.linear_model import LogisticRegression

# values for the hyperparameter/ regularization parameter C
Power=range(-12,10)
C=[]
for power in Power:
    C.append(2**power)

# loop for hyperparameter tuning
# AUC values on the validation set are saved in the auc_val list
auc_val=[] # list initialization
for C_value in C:
    Model=LogisticRegression(C=C_value, solver='liblinear')
    Model.fit(X_train, np.ravel(Y_train))
    probs_val=Model.predict_proba(X_val)[:,1]
    auc_val.append(roc_auc_score(Y_val, probs_val))

print("Hyperparameter tuning ended")
index_maximal_auc=np.argmax(auc_val)
C_optimal=C[index_maximal_auc]

print("Optimal value for parameter C is %f" %C_optimal) # 1.0

# Retrain model with best C-value
LR = LogisticRegression(C = 1.0, solver='liblinear')
LR.fit(X_train, np.ravel(Y_train))

# make predictions on validation set
y_val_labels_lr = LR.predict(X_val)
y_val_scores_lr = LR.predict_proba(X_val)[:,1]

# calculate the accuracy on validation data
accuracy_val_lr = accuracy_score(Y_val, y_val_labels_lr)
print("Validation Accuracy with Best Model: ",accuracy_val_lr)

# AUC value on validation data
AUC_val_lr = roc_auc_score(Y_val, y_val_scores_lr)
print("Validation AUC with Best Model: ",AUC_val_lr)

# Compute the scores for the test instances
y_test_scores_lr = LR.predict_proba(X_test)[:,1]
y_test_labels_lr = LR.predict(X_test)

# Accuracy on test data
accuracy_test_lr = accuracy_score(Y_test, y_test_labels_lr)
print("Test Accuracy with Best Model: ",accuracy_test_lr)

# AUC value on test data
AUC_test_lr = roc_auc_score(Y_test, y_test_scores_lr)
print("Test AUC with Best Model: ",AUC_test_lr)

# make confusion matrix for test set
cf_lr = confusion_matrix(Y_test, y_test_labels_lr)
print("Confusion matrix of test set:\n{}".format(cf_lr))

# ROC curve
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_test_scores_lr)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on test data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC linear regression test = %0.4f' %roc_auc) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()

"""
ADABOOST
"""

from sklearn.ensemble import AdaBoostClassifier

# Fit model
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_ada = adaboost_model.predict(X_val)
y_val_scores_ada = adaboost_model.predict_proba(X_val)[:, 1]
accuracy_val_ada = accuracy_score(Y_val, y_val_labels_ada)
print("Validation Accuracy: ", accuracy_val_ada)

# AUC value on validation data
AUC_val_ada = roc_auc_score(Y_val, y_val_scores_ada)
print("Validation AUC: ", AUC_val_ada)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
gridsearch = GridSearchCV(adaboost_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# Best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'learning_rate': 0.2, 'n_estimators': 200}

# Retrain AdaBoost model with the best model parameters
best_params = {'learning_rate': 0.2, 'n_estimators': 200}
best_adaboost_model = AdaBoostClassifier(**best_params)
best_adaboost_model.fit(X_train, Y_train)

# Predictions on the validation set
y_val_labels_ada = best_adaboost_model.predict(X_val)
y_val_scores_ada = best_adaboost_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_ada = accuracy_score(Y_val, y_val_labels_ada)
print("Validation Accuracy with Best Model: ", accuracy_val_ada)

# AUC value on validation data
AUC_val_ada = roc_auc_score(Y_val, y_val_scores_ada)
print("Validation AUC with Best Model: ", AUC_val_ada)

# Predictions on the test set
y_test_labels_ada = best_adaboost_model.predict(X_test)
y_test_scores_ada = best_adaboost_model.predict_proba(X_test)[:, 1]

# Accuracy on test data
accuracy_test_ada = accuracy_score(Y_test, y_test_labels_ada)
print("Test Accuracy with Best Model: ", accuracy_test_ada)

# AUC value on test data
AUC_test_ada = roc_auc_score(Y_test, y_test_scores_ada)
print("Test AUC with Best Model: ", AUC_test_ada)

# Confusion matrix for the test set
cf_ada = confusion_matrix(Y_test, y_test_labels_ada)
print("Confusion matrix of test set:\n", cf_ada)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_ada)
roc_auc_ada = AUC_test_ada

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC AdaBoost test = %0.4f' % roc_auc_ada)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""
CATBOOST
"""

from catboost import CatBoostClassifier

# Fit model
catboost_model = CatBoostClassifier()
catboost_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_cat = catboost_model.predict(X_val)
y_val_scores_cat = catboost_model.predict_proba(X_val)[:, 1]
accuracy_val_cat = accuracy_score(Y_val, y_val_labels_cat)
print("Validation Accuracy: ", accuracy_val_cat)

# AUC value on validation data
AUC_val_cat = roc_auc_score(Y_val, y_val_scores_cat)
print("Validation AUC: ", AUC_val_cat)

# Hyperparameter Tuning
param_grid = {
    'depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
gridsearch = GridSearchCV(catboost_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# Best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'depth': 5, 'learning_rate': 0.1, 'n_estimators': 200}

# Retrain CatBoost model with best parameters
best_params = {'depth': 5, 'learning_rate': 0.1, 'n_estimators': 200}
best_catboost_model = CatBoostClassifier(**best_params)
best_catboost_model.fit(X_train, Y_train)

# Predictions on the validation set with the best model
y_val_labels_cat = best_catboost_model.predict(X_val)
y_val_scores_cat = best_catboost_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_cat = accuracy_score(Y_val, y_val_labels_cat)
print("Validation Accuracy with Best Model: ", accuracy_val_cat)

# AUC value on validation data
AUC_val = roc_auc_score(Y_val, y_val_scores_cat)
print("Validation AUC with Best Model: ", AUC_val)

# Predictions on the test set
y_test_labels_cat = best_catboost_model.predict(X_test)
y_test_scores_cat = best_catboost_model.predict_proba(X_test)[:, 1]

# Accuracy on test data
accuracy_test = accuracy_score(Y_test, y_test_labels_cat)
print("Test Accuracy with Best Model: ", accuracy_test)

# AUC value on test data
AUC_test_cat = roc_auc_score(Y_test, y_test_scores_cat)
print("Test AUC with Best Model: ", AUC_test_cat)

# Confusion matrix for the test set
cf_cat = confusion_matrix(Y_test, y_test_labels_cat)
print("Confusion matrix of test set:\n", cf_cat)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_cat)
roc_auc_cat = AUC_test_cat

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC CatBoost test = %0.4f' % roc_auc_cat)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""
XGBOOST
"""

import xgboost as xgb

# Fit model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
xgb_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_xgb = xgb_model.predict(X_val)
y_val_scores_xgb = xgb_model.predict_proba(X_val)[:, 1]
accuracy_val_xgb = accuracy_score(Y_val, y_val_labels_xgb)
print("Validation Accuracy: ", accuracy_val_xgb)

# AUC value on validation data
AUC_val_xgb = roc_auc_score(Y_val, y_val_scores_xgb)
print("Validation AUC: ", AUC_val_xgb)

# Tune hyperparameters with GridSearchCV

param_grid = {
    'max_depth': [1, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 800]
}
gridsearch = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

# Retrain XGBoost model with best parameters
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, **best_params)
best_xgb_model.fit(X_train, np.ravel(Y_train))

# Predictions on the validation set
y_val_labels_xgb = best_xgb_model.predict(X_val)
y_val_scores_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_xgb = accuracy_score(Y_val, y_val_labels_xgb)
print("Validation Accuracy with Best Model: ", accuracy_val_xgb)

# AUC value on validation data
AUC_val_xgb = roc_auc_score(Y_val, y_val_scores_xgb)
print("Validation AUC with Best Model: ", AUC_val_xgb)

# Predictions on the test set
y_test_labels_xgb = best_xgb_model.predict(X_test)
y_test_scores_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

# Accuracy on test data
accuracy_test_xgb = accuracy_score(Y_test, y_test_labels_xgb)
print("Test Accuracy with Best Model: ", accuracy_test_xgb)

# AUC value on test data
AUC_test_xgb = roc_auc_score(Y_test, y_test_scores_xgb)
print("Test AUC with Best Model: ", AUC_test_xgb)

# Confusion matrix for the test set
cf_xgb = confusion_matrix(Y_test, y_test_labels_xgb)
print("Confusion matrix of test set:\n", cf_xgb)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_xgb)
roc_auc_xgb = AUC_test_xgb

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC XGBoost test = %0.4f' % roc_auc_xgb)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Feature importance: SHAP
import shap

explainer = shap.Explainer(best_xgb_model)
shap_values = explainer.shap_values(X_val)

# Summary plot
shap.summary_plot(shap_values, X_val)

# Plot the SHAP bar plot
shap.summary_plot(shap_values, X_val, plot_type="bar")
plt.show()

# Learning Curve

# Accuracy:
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_xgb_model, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Accuracy")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.grid()
plt.show()

#ROC AUC

train_sizes, train_scores, test_scores = learning_curve(
    best_xgb_model, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='roc_auc', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training AUC ROC")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation AUC ROC")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.xlabel('Training examples')
plt.ylabel('AUC ROC')
plt.title('Learning Curve: AUC ROC')
plt.legend(loc="best")
plt.grid()
plt.show()

"""
Deployment
"""

"""
Lift curve for kNN model
"""
lift_curve_kNN=[]
for i in range(100):
    lift_curve_kNN.append(liftatp(y_test_scores_knn, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_kNN, 'b')


"""
Lift curve for Decision tree model
"""
lift_curve_DT=[]
for i in range(100):
    lift_curve_DT.append(liftatp(y_test_scores_dt, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_DT, 'r')

"""
Lift curve for RandomForest model
"""
lift_curve_RandomForest=[]
for i in range(100):
    lift_curve_RandomForest.append(liftatp(y_test_scores_rf, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_RandomForest, 'green')

"""
Lift curve for nonlinear SVM model
"""
lift_curve_SVM_nonlinear=[]
for i in range(100):
    lift_curve_SVM_nonlinear.append(liftatp(y_test_scores_SVM_nonlin, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_SVM_nonlinear, 'yellow')

"""
Lift curve for Logistic Regression model
"""
lift_curve_LR=[]
for i in range(100):
    lift_curve_LR.append(liftatp(y_test_scores_lr, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_LR, 'purple')

"""
Lift curve for AdaBoost model
"""
lift_curve_ada=[]
for i in range(100):
    lift_curve_ada.append(liftatp(y_test_scores_ada, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_ada, 'cyan')

"""
Lift curve for CatBoost model
"""
lift_curve_cat=[]
for i in range(100):
    lift_curve_cat.append(liftatp(y_test_scores_cat, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_cat, 'magenta')

"""
Lift curve for XGBOOST model
"""
lift_curve_xgb=[]
for i in range(100):
    lift_curve_xgb.append(liftatp(y_test_scores_xgb, Y_test, (i+1)/100))

# Select the following two lines and press "F9"
plt.title('Lift curve') #title
plt.plot(np.arange(100), lift_curve_xgb, 'mediumseagreen')

plt.title('Lift curve') #title

plt.plot(np.arange(100), lift_curve_DT, 'r', label='Decision Tree') #label shown in the legend
plt.plot(np.arange(100), lift_curve_SVM_nonlinear, 'y', label='SVM Nonlinear') #label shown in the legend
plt.plot(np.arange(100), lift_curve_LR, 'purple', label='Logistic Regression') #label shown in the legend
plt.plot(np.arange(100), lift_curve_kNN, 'b', label='kNN') #label shown in the legend
plt.plot(np.arange(100), lift_curve_RandomForest, 'black', label='Random Forest')
plt.plot(np.arange(100), lift_curve_ada, 'g', label='ADA') #label shown in the legend
plt.plot(np.arange(100), lift_curve_cat, 'magenta', label='CAT') #label shown in the legend
plt.plot(np.arange(100), lift_curve_xgb, 'cyan', label='XGB') #label shown in the legend

plt.axhline(y=1, color='orange', linestyle='-', label='Random baseline') #random baseline
plt.xlim([0, 100])
plt.ylim([0, np.max(lift_curve_LR)+1])
plt.legend(loc = 'upper right') #plot a legend in the upper right corner
plt.ylabel('Lift') #ylabel
plt.xlabel('Percentage of test instances targeted (decreasing by score)') #xlabel
plt.show()

# ROC curve for test set svm
fpr_svm, tpr_svm, threshold_svm = metrics.roc_curve(Y_test, y_test_scores_SVM_nonlin)
roc_auc_SVM_nonlin = metrics.auc(fpr_svm, tpr_svm)

# ROC curve for test set rf
fpr_rf, tpr_rf, threshold_rf = roc_curve(Y_test, y_test_scores_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)

# ROC curve for test set lr
fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(Y_test, y_test_scores_lr)
roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)

# ROC curve for test set
fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(Y_test, y_test_scores_xgb)
roc_auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)

# ROC curve for test set cat
fpr_cat, tpr_cat, threshold_cat = roc_curve(Y_test, y_test_scores_cat)
roc_auc_cat = metrics.auc(fpr_cat, tpr_cat)

# ROC curve for test set ada
fpr_ada, tpr_ada, threshold_ada = roc_curve(Y_test, y_test_scores_ada)
roc_auc_ada = metrics.auc(fpr_ada, tpr_ada)

# Calculate ROC curve and AUC for KNN
fpr_knn, tpr_knn, threshold_knn = roc_curve(Y_test, y_test_scores_knn)
roc_auc_knn = metrics.auc(fpr_knn, tpr_knn)

# Calculate ROC curve and AUC for Decision Tree
fpr_dt, tpr_dt, threshold_dt = roc_curve(Y_test, y_test_scores_dt)
roc_auc_dt = metrics.auc(fpr_dt, tpr_dt)

# Plot ROC curve and AUC for non-linear SVM
plt.plot(fpr_svm, tpr_svm, 'b', label='Non-linear SVM (AUC = %0.4f)' % roc_auc_SVM_nonlin)

# Plot ROC curve and AUC for Random Forest
plt.plot(fpr_rf, tpr_rf, 'g', label='Random Forest (AUC = %0.4f)' % roc_auc_rf)

# Plot ROC curve and AUC for Logistic Regression
plt.plot(fpr_lr, tpr_lr, 'r', label='Logistic Regression (AUC = %0.4f)' % roc_auc_lr)

# Plot ROC curve and AUC for XGBoost
plt.plot(fpr_xgb, tpr_xgb, 'm', label='XGBoost (AUC = %0.4f)' % roc_auc_xgb)

# Plot ROC curve and AUC for CatBoost
plt.plot(fpr_cat, tpr_cat, 'c', label='CatBoost (AUC = %0.4f)' % roc_auc_cat)

# Plot ROC curve and AUC for AdaBoost
plt.plot(fpr_ada, tpr_ada, 'y', label='AdaBoost (AUC = %0.4f)' % roc_auc_ada)

# Plot ROC curve and AUC for KNN
plt.plot(fpr_knn, tpr_knn, 'k', label='KNN (AUC = %0.4f)' % roc_auc_knn)

# Plot ROC curve and AUC for Decision Tree
plt.plot(fpr_dt, tpr_dt, 'orange', label='Decision Tree (AUC = %0.4f)' % roc_auc_dt)

# Plotting the diagonal line indicating a random model
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



