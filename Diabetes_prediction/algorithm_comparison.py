import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    dataset = read_csv('diabetes.csv')
    diabetes_data_copy = dataset.copy(deep=True)
    diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
    
    # Fill missing values
    KNN_imputer = KNNImputer(n_neighbors=5)
    output = KNN_imputer.fit_transform(diabetes_data_copy)
    dataset = pd.DataFrame(output, columns=KNN_imputer.get_feature_names_out())
    return dataset

def get_models():
    return {
        'RandFrst': RandomForestClassifier(n_estimators=100),
        'ExTree': ExtraTreesClassifier(n_estimators=100),
        'GradBoost': GradientBoostingClassifier(n_estimators=100)
    }

def get_oversamplers():
    return {
        'No sampling': None,
        'SMOTE': SMOTE(sampling_strategy='not majority', k_neighbors=5),
        'SVMSMOTE': SVMSMOTE(sampling_strategy='not majority', k_neighbors=5),
        'SMOTEENN': SMOTEENN(sampling_strategy='not majority')
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = metrics.accuracy_score(y_train, train_pred)
    test_acc = metrics.accuracy_score(y_test, test_pred)
    
    train_f1 = metrics.f1_score(y_train, train_pred, average='weighted')
    test_f1 = metrics.f1_score(y_test, test_pred, average='weighted')
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1
    }

def compare_algorithms():
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Prepare features and target
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    quantile_trans = QuantileTransformer(output_distribution='normal')
    X_train = quantile_trans.fit_transform(X_train)
    X_test = quantile_trans.transform(X_test)
    
    # Initialize results
    results = {
        'RandFrst': {'No sampling': {}, 'SMOTE': {}, 'SVMSMOTE': {}, 'SMOTEENN': {}},
        'ExTree': {'No sampling': {}, 'SMOTE': {}, 'SVMSMOTE': {}, 'SMOTEENN': {}},
        'GradBoost': {'No sampling': {}, 'SMOTE': {}, 'SVMSMOTE': {}, 'SMOTEENN': {}}
    }
    
    # Get models and oversamplers
    models = get_models()
    oversamplers = get_oversamplers()
    
    print("\nRunning comparisons...")
    # Run comparisons
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        for sampler_name, sampler in oversamplers.items():
            print(f"  Using {sampler_name}...")
            
            # Apply oversampling if needed
            if sampler is not None:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # Evaluate model
            metrics = evaluate_model(model, X_train_resampled, X_test, y_train_resampled, y_test)
            results[model_name][sampler_name] = metrics
    
    # Print results in a clear format
    print("\nResults Summary:")
    print("\nAccuracy Scores (Train/Test):")
    print("Model\t\tNo sampling\t\tSMOTE\t\t\tSVMSMOTE\t\tSMOTEENN")
    print("-" * 100)
    for model_name in models.keys():
        row = f"{model_name}\t\t"
        for sampler_name in oversamplers.keys():
            train_acc = results[model_name][sampler_name]['train_acc']
            test_acc = results[model_name][sampler_name]['test_acc']
            row += f"{train_acc:.3f}/{test_acc:.3f}\t\t"
        print(row)
    
    print("\nF1 Scores (Train/Test):")
    print("Model\t\tNo sampling\t\tSMOTE\t\t\tSVMSMOTE\t\tSMOTEENN")
    print("-" * 100)
    for model_name in models.keys():
        row = f"{model_name}\t\t"
        for sampler_name in oversamplers.keys():
            train_f1 = results[model_name][sampler_name]['train_f1']
            test_f1 = results[model_name][sampler_name]['test_f1']
            row += f"{train_f1:.3f}/{test_f1:.3f}\t\t"
        print(row)
    
    return results

if __name__ == "__main__":
    results = compare_algorithms() 