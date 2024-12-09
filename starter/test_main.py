from fastapi.testclient import TestClient
from main import app
import requests
import json
client = TestClient(app)


# test the greeting function.
def test_get():
    # local deployment
    # r = client.get('/')
    # render deployment
    r = requests.get(url="https://udacity-devops-course-6.onrender.com/")
    assert r.status_code == 200
    assert r.json() == {"Greeting message": "Hello there"}


def test_post_lower_than_threshold():
    data = {"age": 40, "workclass": "Private",
            "fnlgt": 83311, "education": "Masters",
            "education_num": 14, "marital_status": "Never-married",
            "occupation": "Tech-support", "relationship": "Not-in-family",
            "race": "White", "sex": "Female",
            "capital_gain": 1000, "capital_loss": 0,
            "hours_per_week": 50, "native_country": "United-States"}
    # local deployment
    # x = requests.post('http://127.0.0.1:8000/inference', json=data)
    # deployment on render
    x = requests.post('https://udacity-devops-course-6.onrender.com/inference',
                      json=data)
    prediction = json.loads(x.content.decode('utf-8'))['prediction']
    # check the status code
    assert x.status_code == 200
    # check the prediction is one of the two classes.
    assert (prediction == '<=50K')


def test_post_higher_than_threshold():
    data = {"age": 30, "workclass": "Private",
            "fnlgt": 280464, "education": "Masters",
            "education_num": 14, "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial", "relationship": "Husband",
            "race": "White", "sex": "Male",
            "capital_gain": 10000, "capital_loss": 0,
            "hours_per_week": 20, "native_country": "United-States"}
    # local deployment
    # x = requests.post('http://127.0.0.1:8000/inference', json=data)
    # deployment on render
    x = requests.post('https://udacity-devops-course-6.onrender.com/inference',
                      json=data)
    prediction = json.loads(x.content.decode('utf-8'))['prediction']
    # check the status code
    assert x.status_code == 200
    # check the prediction is one of the two classes.
    assert (prediction == '>50K')


# age input is missing.
def test_post_wrong_input():
    data = {"workclass": "Private", "fnlgt": 83311, "education": "Masters",
            "education_num": 14, "marital_status": "Never-married",
            "occupation": "Tech-support", "relationship": "Not-in-family",
            "race": "White", "sex": "Female",
            "capital_gain": 1000, "capital_loss": 0,
            "hours_per_week": 50, "native_country": "United-States"}
    # local deployment
    # x = requests.post('http://127.0.0.1:8000/inference', json=data)
    # render deployment
    x = requests.post('https://udacity-devops-course-6.onrender.com/inference',
                      json=data)
    # check the status code
    assert x.status_code == 422
    # check the error msg.
    # local deployment
    # msg= {"status": "failed", "error": "Incorrect input",
    #          "details": "column ('body', 'age') is missing"}
    msg = {"status": "failed", "error": "Incorrect input",
           "details": "column ('body', 'age') is value_error.missing"}
    assert (json.loads(x.content.decode('utf-8')) == msg)
