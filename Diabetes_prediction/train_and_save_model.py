import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and preprocess data
def load_and_preprocess_data():
    dataset = read_csv('diabetes.csv')
    diabetes_data_copy = dataset.copy(deep=True)
    diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
    
    # Fill missing values
    KNN_imputer = KNNImputer(n_neighbors=5)
    output = KNN_imputer.fit_transform(diabetes_data_copy)
    dataset = pd.DataFrame(output, columns=KNN_imputer.get_feature_names_out())
    
    return dataset

# Train and save model
def train_and_save_model():
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
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('diabetes_model_nb.sav', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model() 