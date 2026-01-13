# Diabetes Risk Prediction System

This project implements a machine learning system for predicting diabetes risk based on various health measurements. It includes a FastAPI backend and a user-friendly web frontend.

## Project Structure
```
diabetes-prediction/
├── deployement_FastAPI/
│   ├── main.py           # FastAPI backend
│   └── input.py          # Sample API client
├── frontend/
│   └── index.html        # Web interface
├── diabetes.csv          # Dataset
├── diabetes3.py          # Original ML implementation
├── algorithm_comparison.py # Algorithm comparison tool
├── train_and_save_model.py # Model training script
└── requirements.txt      # Python dependencies
```

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- A modern web browser

## Installation

1. Clone or download the project repository

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Workflow and Steps

### 1. Data Preprocessing
- Loading the PIMA Indians dataset
- Handling missing values (replacing 0s with NaN)
- Using KNNImputer for imputation
- Data cleaning and validation

### 2. Data Transformation
- Standardizing features using QuantileTransformer
- Normalizing data distributions
- Handling skewed data
- Feature scaling

### 3. Feature Engineering
- Feature selection using SelectKBest
- Correlation analysis
- Feature importance assessment
- Dimensionality reduction

### 4. Model Training
- Splitting data into training and testing sets
- Implementing cross-validation
- Training multiple models:
  - Random Forest
  - Extra Trees
  - Gradient Boosting
- Hyperparameter tuning

### 5. Model Evaluation
- Calculating accuracy scores
- Computing F1 scores
- Generating confusion matrices
- ROC curve analysis
- Cross-validation scores

### 6. Testing and Validation
- Testing on unseen data
- Performance comparison
- Overfitting/underfitting analysis
- Model generalization assessment

## Running the Project

### 1. Train and Save the Model
First, train the model and save it:
```bash
python train_and_save_model.py
```
This will create `diabetes_model_nb.sav` in your project directory.

### 2. Start the Backend Server
Open a new terminal and run:
```bash
cd deployement_FastAPI
python -m uvicorn main:app --reload --port 8002
```
The API will be available at http://127.0.0.1:8002

### 3. Run the Frontend
You can run the frontend in two ways:

#### Option 1: Direct File Opening
- Navigate to the `frontend` folder
- Double-click `index.html` to open it in your browser

#### Option 2: Using Python's HTTP Server
```bash
cd frontend
python -m http.server 8000
```
Then open http://localhost:8000 in your browser

## Using the Application

1. Open the web interface in your browser
2. Fill in the patient's health measurements:
   - Number of Pregnancies
   - Glucose Level (mg/dL)
   - Blood Pressure (mm Hg)
   - Skin Thickness (mm)
   - Insulin Level (mu U/ml)
   - BMI (kg/m²)
   - Diabetes Pedigree Function
   - Age (years)
3. Click "Predict Risk"
4. View the prediction result

## API Endpoints

### 1. Prediction Endpoint
- URL: `http://127.0.0.1:8002/diabetes_prediction`
- Method: POST
- Input Format:
```json
{
    "Pregnancies": 8,
    "Glucose": 183,
    "BloodPressure": 64,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 23.3,
    "DiabetesPedigreeFunction": 0.672,
    "Age": 32
}
```

### 2. Welcome Endpoint
- URL: `http://127.0.0.1:8002/`
- Method: GET
- Returns a welcome message

## Additional Tools

### Algorithm Comparison
To compare different machine learning algorithms and oversampling techniques:
```bash
python algorithm_comparison.py
```
This will show a comparison table of different algorithms and their performance metrics.

## Troubleshooting

1. If you get a "uvicorn not found" error:
   - Make sure you've activated your virtual environment
   - Try installing uvicorn separately: `pip install uvicorn`

2. If the frontend can't connect to the backend:
   - Ensure the backend server is running
   - Check that you're using the correct port (8002)
   - Verify CORS settings in main.py

3. If you get a "model not found" error:
   - Make sure you've run train_and_save_model.py first
   - Check that diabetes_model_nb.sav exists in the correct location

## Notes
- The model is trained on the PIMA Indians dataset
- The system uses Random Forest Classifier as the default model
- The frontend includes input validation and error handling

