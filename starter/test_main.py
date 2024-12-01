from fastapi.testclient import TestClient
from main import app
import requests
import json
client = TestClient(app)

# test the greeting function.
def test_get():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"Greeting message": "Hello there"}

def test_post():
    
    data = {"age":40,"workclass":"Private", "fnlgt":83311, "education":"Masters", "education_num":14,"marital_status":"Never-married", "occupation":"Tech-support", "relationship":"Not-in-family", "race":"White", "sex":"Female",
            "capital_gain":1000, "capital_loss":0, "hours_per_week":200, "native_country":"United-States"}
    
    x = requests.post('http://127.0.0.1:8000/inference',json=data)
    prediction = json.loads(x.content.decode('utf-8'))['prediction']
    # check the status code
    assert x.status_code == 200
    # check the prediction is one of the two classes.
    assert (prediction in ['<=50K','>50K'])

# age input is missing.
def test_post_wrong_input():
    
    data = {"workclass":"Private", "fnlgt":83311, "education":"Masters", "education_num":14,"marital_status":"Never-married", "occupation":"Tech-support", "relationship":"Not-in-family", "race":"White", "sex":"Female",
            "capital_gain":1000, "capital_loss":0, "hours_per_week":200, "native_country":"United-States"}
    
    x = requests.post('http://127.0.0.1:8000/inference',json=data)

    
    # check the status code
    assert x.status_code == 422
    # check the error msg.
    assert json.loads(x.content.decode('utf-8')) == {"status":"failed","error":"Incorrect input","details":"column ('body', 'age') is missing"}