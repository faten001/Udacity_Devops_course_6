# Put the code for your API here.
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from starter.ml.data import process_data
import pandas as pd

# use pythoon type hints to make fastapi creates automatic documentation.

#{"age":30,"workclass":"Private", "fnlgt":83311, "education":"Masters", "education_num":14,
#"marital_status":"Never-married", "occupation":"Tech-support", "relationship":"Not-in-family", 
#"race":"White", "sex":"Female","capital_gain":0, "capital_loss":0, "hours_per_week":100, "native_country":"United-States"}
class Person_info(BaseModel):
    age: int
    workclass: str # you can ;use Literal['value1','value2']
    fnlgt: int
    education: str
    education_num:int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex:str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


app = FastAPI()





def inference(sample):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    # load model and encoders.
    encoder = joblib.load('model/one_h_encoder.sav')
    lb = joblib.load('model/label_coding.sav')
    model = joblib.load('model/random_forest.sav')
    # convert the input to dict
    sample = {"age":sample.age,"workclass":sample.workclass, "fnlgt":sample.fnlgt, "education":sample.education, "education_num":sample.education_num,
           "marital_status":sample.marital_status, "occupation":sample.occupation, "relationship":sample.relationship, 
          "race":sample.race, "sex":sample.sex,"capital_gain":sample.capital_gain, "capital_loss":sample.capital_loss, 
        "hours_per_week":sample.hours_per_week, "native_country":sample.native_country}
    # process the input for inference
    processed_data,_,_,_ = process_data(pd.DataFrame([sample]),categorical_features=cat_features,training=False,encoder=encoder,lb=lb)
    
    # predict
    preds = model.predict(processed_data)
    # convert the numerical label to categorical value.
    return lb.classes_[preds[0]]

# post on different path and does model inference
@app.post("/inference")
async def create_item(sample: Person_info):
    
    pred = inference(sample)
    
    return {'input':sample,'prediction':str(pred)}

# get must be on the root domain and give greeting (Done)
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": "failed",
            "error": "Incorrect input",
            "details": f"column {exc.errors()[0]['loc']} is {exc.errors()[0]['type']}",
        },
    )
@app.get("/")
async def greetings():
    return {"Greeting message": "Hello there"}

# include a screen shot of the docs that shows the example and name it example.png

