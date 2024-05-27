import uvicorn
import pandas as pd
from fastapi import FastAPI
from catboost import CatBoostClassifier

#path to the model
MODEL_PATH = "C:/Users/hp/OneDrive/Documents/Telcom service/model/catboost_model.cbm"

#load modelfunction
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

#function of the churn probability
def get_churn_probability(data,model):
    #convert data to dataframe 
    dataframe = pd.DataFrame.from_dict(data,orient='index').T
    #make prediction
    churn_probability = model.predict_proba(dataframe)[0][1]
    return churn_probability

#load model
model = load_model()

#create API for churn probability
app = FastAPI(title="Churn prediction",version="1.0")

@app.get('/')
def index():
    return {'message':'Churn predicition API'}

#define API endpoint
@app.post('/predict/')
def predict_churn(data:dict):
    #get the prediction
    churn_probability = get_churn_probability(data,model)
    return {'Churn probability': churn_probability}

if __name__ == '__main__':
    uvicorn.run("api:app",host='127.0.0.1',port=5000)