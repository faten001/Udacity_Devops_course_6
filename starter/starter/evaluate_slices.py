import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference, compute_model_metrics
pd.set_option('display.float_format', '{:.3f}'.format)

# evaluate on different slices of a categorical feature
# print the scores
# save the results in a text file.


def test_feature_slice(feature):
    # load the model and encoders
    results_df = pd.DataFrame(columns=['slice', 'precision', 'recall',
                                       'fbeta', 'size'])
    model = joblib.load('starter/model/random_forest.sav')
    data = pd.read_csv('starter/data/census.csv')
    encoder = joblib.load('starter/model/one_h_encoder.sav')
    lb = joblib.load('starter/model/label_coding.sav')

    # evalate on different value of the feature.
    print(f'Scores on different slices of {feature}\n')
    for value in data[feature].unique():
        data_slice = data[data[feature] == value].copy()
        X, y = prepare_data(data_slice, encoder, lb)
        # inference and calculate the scores
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        # save the results to dataframe
        results_df.loc[len(results_df)] = [value, precision, recall, fbeta,
                                           len(data_slice)]
    print(results_df)
    # save the results into a text file
    with open('starter/slice_output.txt', 'w') as file:
        file.write(results_df.to_string(header=True, index=False))


def prepare_data(data, encoder, lb):
    num_cols = data._get_numeric_data().columns
    categorical_col = list(set(data.columns) - set(num_cols))
    categorical_col.remove('salary')
    X, y, _, _ = process_data(data, categorical_col, 'salary',
                              False, encoder, lb)
    return X, y


test_feature_slice('education')
