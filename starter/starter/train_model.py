# Script to train machine learning model.

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib


# Add code to load in the data.
data = pd.read_csv('data/census.csv')
# the code to remove white spaces in categorical columns.
# num_cols = data._get_numeric_data().columns
# categorical_col =  list(set(data.columns) - set(num_cols))
# for c in categorical_col:
#     data[c] = data[c].str.replace(r'\s+', '', regex=True)
# data.columns = [col.replace('-','_') for col in data.columns]
# data.to_csv('starter/data/census.csv',index=False)

train, test = train_test_split(data, test_size=0.20)

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


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and reference on the model.
model = train_model(X_train, y_train)
# print(X_train.shape)
pred_train = inference(model, X_train)

# evaluate the model
precision, recall, fbeta = compute_model_metrics(y_train, pred_train)
# print training scores
print(f"Training Scores\nprecision: {precision} \
      recall: {recall}\nfbeta: {fbeta}")

# save the model and the encoders
joblib.dump(model, 'model/random_forest.sav')
joblib.dump(encoder, 'model/one_h_encoder.sav')
joblib.dump(lb, 'model/label_coding.sav')

# load the model and encoders for inference on test data
rf = joblib.load('model/random_forest.sav')
encoder = joblib.load('model/one_h_encoder.sav')
lb = joblib.load('model/label_coding.sav')
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# evaluate testset
pred = inference(rf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)
# print test scores
print(f"\nTesting Scores\nprecision: {precision} \
      recall: {recall}\nfbeta: {fbeta}")
