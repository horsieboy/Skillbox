import dill
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
model = joblib.load('data/model.pkl')


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    predict: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    with open('data/ohe.pkl', 'rb') as file:
        ohe = dill.load(file)
    ohe_columns = ohe.transform(df)
    df_prepared = pd.DataFrame(ohe_columns, columns=list(ohe.get_feature_names_out()))
    y = model['model'].predict(df_prepared)
    return {'predict': y}
