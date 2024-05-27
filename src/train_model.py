#import dependencies
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import f1_score,recall_score,classification_report,accuracy_score,precision_score,roc_curve,auc,confusion_matrix,roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
#import catboodst for classifier model
from catboost import CatBoostClassifier,Pool

data_path = "C:/Users/hp/OneDrive/Documents/Telcom service/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

telecom_df = pd.read_csv(data_path)

#convert "tptalcharges" to numeric and flll the NaN values with the numerical values
telecom_df["TotalCharges"] = pd.to_numeric(telecom_df["TotalCharges"],errors='coerce')
telecom_df["TotalCharges"].fillna(telecom_df["tenure"]*telecom_df["MonthlyCharges"],inplace=True)

#convert seniorcitizens to an object data type
telecom_df['SeniorCitizen'] = telecom_df['SeniorCitizen'].astype(object)

#Replace "No phone service" and "No internet services" with "No"
telecom_df['MultipleLines'] = telecom_df['MultipleLines'].replace('No phone service', 'No')
columns_to_replace = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for column in columns_to_replace:
    telecom_df[column] = telecom_df[column].replace('No internet service','No')

#convert churn to 0 and 1
telecom_df['Churn'] = telecom_df['Churn'].replace({'No':0,'Yes':1})

#stratifiedshufflesplit
strasplit = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_index,test_index = next(strasplit.split(telecom_df,telecom_df['Churn']))

#create train and test set
strat_train = telecom_df.loc[train_index]
strat_test = telecom_df.loc[test_index]

x_train = strat_train.drop("Churn",axis=1)
y_train = strat_train["Churn"].copy()

x_test = strat_test.drop("Churn",axis=1)
y_test = strat_test["Churn"].copy()

#Implement catboost classifier
#Identify categorical(binary in this situation since we have 0 and 1 only) classes
categorical_columns = telecom_df.select_dtypes(include=['object']).columns.to_list()

#initialize the model
catboost_model = CatBoostClassifier(verbose=False,random_state=0,scale_pos_weight=3)
catboost_model.fit(x_train,y_train,cat_features=categorical_columns,eval_set=(x_test,y_test))

#predict the set
y_pred = catboost_model.predict(x_test)

#calculate evaluation metrics
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

#create a df to store results
model_name = ['Catboost_model']
result = pd.DataFrame({'Accuracy':accuracy,'Recall':recall,'Roc_Auc':roc_auc,'Precision':precision},index=model_name)

# Print results
print(result)

# Save the model in the 'model' directory
model_dir = "C:/Users/hp/OneDrive/Documents/ML_Project/Telcom service/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "catboost_model.cbm")
catboost_model.save_model(model_path)