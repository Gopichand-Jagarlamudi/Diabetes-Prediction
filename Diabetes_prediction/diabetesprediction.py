# Import Libraries

#from pycaret.classification import *
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

# Inspect the data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
dataset = read_csv('diabetes.csv')

# Print First 10 and last 10 rows
print(dataset.head(10))
print(dataset.tail(10), "\n")

# All features are numeric and there are 0 null's
print(dataset.describe(), "\n")
print(dataset.info(), "\n")

# Lot of 0 values for SkinThickness and Insulin, some for glucose, blood pressure, and BMI as well
p = dataset.hist(figsize = (20,20))
plt.show()

dataset2 = dataset.iloc[:, :-1]
print("# of Rows, # of Columns: ",dataset2.shape)
print("\nColumn Name           # of 0 Values\n")
print((dataset2[:] == 0).sum())

# Transform 0's into NaN's (excluding pregnancies)
diabetes_data_copy = dataset.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())
print(diabetes_data_copy.info(), "\n")

# Fill-up missing values with KNNImputer
KNN_imputer = KNNImputer(n_neighbors=5)
output = KNN_imputer.fit_transform(diabetes_data_copy)

dataset = pd.DataFrame(output, columns=KNN_imputer.get_feature_names_out())  # Transfrom numpy-array 2 pandas-dataframe
print(dataset.describe(), "\n")
print(dataset.info())

p = dataset.hist(figsize = (20,20))
plt.show()

# Outcome is now balanced
print(dataset["Outcome"].value_counts())

feats_X = dataset.iloc[:, [0,1,2,3,4,5,6,7]]
trgt_y = pd.DataFrame(dataset.iloc[:, 8])

# Feature Engineering -> Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Relevant/Good: features(Independent/Input) --high +/-ve correlation--> target(Dependent/Output)
# Irrelevant/Bad: features(Independent/Input) --low near-0 correlation--> target(Dependent/Output)

# 'SelectKBest' Feature-Selection strategy
for t in range(trgt_y.shape[1]):
  KBest = SelectKBest(score_func=f_classif, k= 8)  # Classification(chi2|f_classif|mutual_info_classif); Regression(r_regression|f_regression|mutual_info_regression)
  output = KBest.fit_transform(feats_X, trgt_y.iloc[:,t])  # Fits to data & transform/reduce it to the selected features
  top_feats_X_1 = pd.DataFrame(output, columns=KBest.get_feature_names_out())  # Transfrom numpy-array 2 pandas-dataframe
  print(top_feats_X_1.head(), "\n") #1st 5 rows
  print(top_feats_X_1.tail()) #last 5 rows

sns.heatmap(top_feats_X_1.corr(), annot=True, linewidths=0.5, fmt='.2f', cmap='BrBG')
plt.show()

# Training & Test Sets: train_test_split
X = top_feats_X_1.loc[:, :]
X = feats_X.loc[:, :]
y = trgt_y  # Using "Outcome" to ensure data(Train|Test) contain fair samples of this feature

print(X.describe())
print(y.describe())
print(X.shape)
print(y.shape, "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train:", X_train.shape, ",\t", "y_train:", y_train.shape)
print("X_test:", X_test.shape, ",\t", "y_test:", y_test.shape)

# Data resampling
from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

# Testing SMOTEENN
smoteenn = SMOTEENN(sampling_strategy='not majority')
print("Oversampling minority class... ", end='')
X_train, y_train = smoteenn.fit_resample(X_train, y_train)
print("Done!")

"""# Testing SMOTE Oversampler
smote = SMOTE(sampling_strategy='not majority', k_neighbors=5)
print("Oversampling minority class... ", end='')
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Done!")"""


"""# Testing SVMSMOTE Oversampler
svm_smote = SVMSMOTE(sampling_strategy='not majority', k_neighbors=5)
print("Oversampling minority class... ", end='')
X_train, y_train = svm_smote.fit_resample(X_train, y_train)
print("Done!")"""

# Features' Standardization AND Samples' Normalization ("skewness/unsymmetric" dataset)
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, Normalizer

# STANDARDIZATION (Column-wise): Non-Linear Transformation for 'feats_X'
# 'QuantileTransformer()' transforms features (columns) with 'skewed/congested' or 'highly-spread' data into a uniform(0..1)/normal(-1..0..+1) distr.
quantile_trans_standzatn = QuantileTransformer(output_distribution='normal')
quantile_trans_standzatn = quantile_trans_standzatn.fit(X_train)  # Fit ONLY on data(Train) to avoid 'data-leak' & 'bias' from data(Test)
standized_X_train = quantile_trans_standzatn.transform(X_train)
standized_X_test = quantile_trans_standzatn.transform(X_test)

# STANDARDIZATION (Column-wise): Linear Transformation for 'classifier_y'
# Requires NO transformation for 'trgt_y' cos it already contains values(Binary)
standized_y_train = y_train
standized_y_test = y_test
# Histrogram shows dataset WITHOUT 'skewness' (data is uniformly distributed)
# pd.DataFrame(standized_X_train, columns=X_train.columns).hist(bins=50, figsize=(14, 13))
# plt.show()

"""# compare_models: Trains/evaluates performance of all estimators available in the model library using cross validation.
clf1 = setup(data = dataset,
             target = 'Outcome',
             preprocess = False,
             verbose= False)

top5 = compare_models(sort='AUC',
                      n_select = 5,
                      exclude=['lightgbm','dummy','svm','ridge','qda']
                     )
"""
# Select ML/DL algorithm AND Tune/Re-tune hyperparameters
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

ml_algorithm = input("Please enter a classifier: ") # KNN, GradBoost, RandFrst, ExtraTree, DeciTree, etc

if (ml_algorithm == "lr"):
  algorithm = MultiOutputClassifier(LogisticRegression(solver='lbfgs'))  # For classification only
elif (ml_algorithm == "lda"):
  algorithm = MultiOutputClassifier(LinearDiscriminantAnalysis())
elif (ml_algorithm == "nb"):
  algorithm = MultiOutputClassifier(GaussianNB())
elif (ml_algorithm == "ada"):
  algorithm = MultiOutputClassifier(AdaBoostClassifier())
elif (ml_algorithm == "gbc"):
  algorithm = MultiOutputClassifier(GradientBoostingClassifier())
elif (ml_algorithm == "xgboost"):
  algorithm = MultiOutputClassifier(XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'))
elif (ml_algorithm == "dt"):
  algorithm = MultiOutputClassifier(DecisionTreeClassifier())
elif (ml_algorithm == "rf"):
  algorithm = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
elif (ml_algorithm == "et"):
  algorithm = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=100))
elif (ml_algorithm == "knn"):
  algorithm = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=6, weights='distance'))
else:
    print("Invalid classifier\nExiting...")
    exit()

# TRAINING: K-Fold Cross-Validation. Fit algorithm(ML) to data(Training).
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef

cv_preds = cross_val_predict(algorithm, standized_X_train, standized_y_train, cv=5, method='predict')
print("Shape of Output/Predictions: ", cv_preds.shape, "\n")
reversed_y_train = pd.DataFrame(cv_preds)
reversed_y_train = np.rint(reversed_y_train).astype(np.int32)

# Evaluate model's performance. Overfitting(Train ERR << Test ERR); Underfitting(Train ERR >> Test ERR).
raw_y_train = y_train.values[:,:]
acc = accuracy_score(raw_y_train, reversed_y_train, normalize=True)  # normalize=True(fraction of correctly classified samples); normalize=False(no. of correctly classified samples)
pr_rc_fs_sp = precision_recall_fscore_support(raw_y_train, reversed_y_train, average='weighted')  # average='weighted'(compute metrics per label, and find their avg. weighted by support (no. of true instances per label))
mcc = matthews_corrcoef(raw_y_train, reversed_y_train)
algorithm.fit(standized_X_train, standized_y_train)
pred_scr = algorithm.predict_proba(standized_X_train)[0]
roc = roc_auc_score(raw_y_train, pred_scr[:,1], average='weighted', multi_class='ovo')  # multi_class='ovo'(meaning One-vs-one. Computes avg. AUC of all possible pairwise combinations of classes)

# Training Metrics
print("Training Metrics: ", "\n----------------")
print("ACCURACY: ", acc)
print("PRECISION: ", pr_rc_fs_sp[0])
print("RECALL: ", pr_rc_fs_sp[1])
print("F1-SCORE: ", pr_rc_fs_sp[2])
print("AREA under ROC: ", roc)
print("MCC: ", mcc)

# TESTING/GENERALIZATION: Make Predictions for 'y-component' wrt. data(Test)
model = algorithm
model.fit(standized_X_train, standized_y_train)
pred_y_test = model.predict(standized_X_test)
reversed_y_test = pd.DataFrame(pred_y_test)
reversed_y_test = np.rint(reversed_y_test).astype(np.int32)

# Evaluate model's performance. Overfitting(Train ERR << Test ERR); Underfitting(Train ERR >> Test ERR).
raw_y_test = y_test.values[:,:]
acc = accuracy_score(raw_y_test, reversed_y_test, normalize=True)  # normalize=True(fraction of correctly classified samples); normalize=False(no. of correctly classified samples)
pr_rc_fs_sp = precision_recall_fscore_support(raw_y_test, reversed_y_test, average='weighted')  # average='weighted'(compute metrics per label, and find their avg. weighted by support (no. of true instances per label))
mcc = matthews_corrcoef(raw_y_test, reversed_y_test)
pred_scr = model.predict_proba(standized_X_test)[0]
roc = roc_auc_score(raw_y_test, pred_scr[:,1], average='weighted', multi_class='ovo')  # multi_class='ovo'(meaning One-vs-one. Computes avg. AUC of all possible pairwise combinations of classes)

# Testing Metrics
print("\nTesting Metrics: ", "\n----------------")
print("ACCURACY: ", acc)
print("PRECISION: ", pr_rc_fs_sp[0])
print("RECALL: ", pr_rc_fs_sp[1])
print("F1-SCORE: ", pr_rc_fs_sp[2])
print("AREA under ROC: ", roc)
print("MCC: ", mcc)

cnf_matrix = metrics.confusion_matrix(y_test, pred_y_test)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()



scores = cross_val_score(algorithm, standized_X_train, standized_y_train, cv=3)
print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (ml_algorithm, 100*scores.mean(), 100*scores.std() * 2))

