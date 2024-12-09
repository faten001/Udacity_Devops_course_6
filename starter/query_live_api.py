
import requests

# the sample we want to infer
data = {"age": 40, "workclass": "Private", "fnlgt": 83311,
        "education": "Masters",
        "education_num": 14, "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family", "race": "White", "sex": "Female",
        "capital_gain": 1000, "capital_loss": 0, "hours_per_week": 50,
        "native_country": "United-States"}

# sending request to live api.
request = requests.post(
    url='https://udacity-devops-course-6.onrender.com/inference',
    json=data,
    headers={"Content-Type": "application/json"})
print(f'Status code : {request.status_code}')
print(request.json())
