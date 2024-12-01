from starter.ml.data import process_data
from starter.ml.model import train_model
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.read_csv('data/census.csv')
    # talk a small subset of the data for testing.
    df = df[:10].copy()
    
    return df

@pytest.fixture
def categorical_columns(data):
    num_cols = data._get_numeric_data().columns
    categorical_col =  list(set(data.columns) - set(num_cols))
    categorical_col.remove('salary')
    
    return categorical_col
# make sure the function return none when in test mode and the encoder where not passed. 
def test_no_encoder(data,categorical_columns): 
    X, y, encoder, lb = process_data(data,categorical_columns,training=False)
    assert X == None

# make sure the target is not in the training data.
def test_target_removed(data,categorical_columns):
    X, y, encoder, lb = process_data(data,categorical_columns,training=True,label='salary')
    X = np.transpose(X)
    # check there no duplicates columns of the target in X
    target_leak = False
    for row in X:
        if np.array_equal(row, y):
            target_leak = True

    assert target_leak == False


# ensure the model after training is not none.
def test_train_model(data,categorical_columns):
    X, y, encoder, lb = process_data(data,categorical_columns,training=True,label='salary')
    model = train_model(X,y)
    assert model != None