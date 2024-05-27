# Telco Customer Churn Prediction

This repository contains code for predicting customer churn in a telecommunications company using the CatBoost classifier. The project includes data preprocessing, model training, evaluation, and deployment using Streamlit and FastAPI.

## Table of Contents

1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Deployment](#deployment)
6. [Usage](#usage)

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt
```

## Data Preprocessing

The data preprocessing steps include handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.

### Steps:

1. **Load Data**:
   ```python
   telecom_df = pd.read_csv(data_path)
   ```

2. **Handle Missing Values**:
   ```python
   telecom_df["TotalCharges"] = pd.to_numeric(telecom_df["TotalCharges"], errors='coerce')
   telecom_df["TotalCharges"].fillna(telecom_df["tenure"] * telecom_df["MonthlyCharges"], inplace=True)
   ```

3. **Encode Categorical Variables**:
   ```python
   telecom_df['SeniorCitizen'] = telecom_df['SeniorCitizen'].astype(object)
   columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
   for column in columns_to_replace:
       telecom_df[column] = telecom_df[column].replace('No internet service', 'No')
   telecom_df['Churn'] = telecom_df['Churn'].replace({'No': 0, 'Yes': 1})
   ```

4. **Split Data**:
   ```python
   strasplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   train_index, test_index = next(strasplit.split(telecom_df, telecom_df['Churn']))
   strat_train = telecom_df.loc[train_index]
   strat_test = telecom_df.loc[test_index]
   x_train = strat_train.drop("Churn", axis=1)
   y_train = strat_train["Churn"].copy()
   x_test = strat_test.drop("Churn", axis=1)
   y_test = strat_test["Churn"].copy()
   ```

## Model Training

The model training involves initializing the CatBoost classifier and fitting it to the training data.

### Steps:

1. **Identify Categorical Features**:
   ```python
   categorical_columns = telecom_df.select_dtypes(include=['object']).columns.to_list()
   ```

2. **Initialize and Train the Model**:
   ```python
   catboost_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
   catboost_model.fit(x_train, y_train, cat_features=categorical_columns, eval_set=(x_test, y_test))
   ```

## Evaluation

Evaluate the model using various metrics such as accuracy, recall, precision, and ROC-AUC.

### Steps:

1. **Predict and Calculate Metrics**:
   ```python
   y_pred = catboost_model.predict(x_test)
   accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]
   ```

2. **Store Results**:
   ```python
   model_name = ['Catboost_model']
   result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_name)
   print(result)
   ```

3. **Save the Model**:
   ```python
   model_path = os.path.join(model_dir, "catboost_model.cbm")
   catboost_model.save_model(model_path)
   ```

## Deployment

The deployment includes creating a Streamlit app for user interaction and a FastAPI service for model inference.

### Streamlit App

1. **Load Model and Data**:
   ```python
   model = load_model()
   data = load_data()
   ```

2. **User Interface for Churn Prediction**:
   ```python
   st.title("Telco Customer Churn Project")
   ```

3. **SHAP Values for Model Explanation**:
   ```python
   plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train)
   ```

### FastAPI Service

1. **Define API Endpoint**:
   ```python
   app = FastAPI(title="Churn prediction", version="1.0")
   ```

2. **Churn Prediction Endpoint**:
   ```python
   @app.post('/predict/')
   def predict_churn(data: dict):
       churn_probability = get_churn_probability(data, model)
       return {'Churn probability': churn_probability}
   ```

3. **Run the Service**:
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 5000
   ```

## Usage

To use the churn prediction model, run the Streamlit app for an interactive interface or send POST requests to the FastAPI service with customer data to get churn probability predictions.

### Streamlit App

Run the Streamlit app:
```bash
streamlit run app.py
```

### FastAPI Service

Start the FastAPI service:
```bash
uvicorn api:app --host 127.0.0.1 --port 5000
```

Send a POST request to the `/predict/` endpoint with customer data to get the churn probability.

```json
{
  "customerID": "6464-UIAEA",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

This README provides a comprehensive guide to understanding, installing, and using the Telco Customer Churn Prediction project. Follow the steps to preprocess data, train the model, evaluate its performance, and deploy it using Streamlit and FastAPI.
