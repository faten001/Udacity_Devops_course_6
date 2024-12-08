# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model date: Dec-2024
- Model version: initial version v1.0
- Model type: Random forest classifier
- Author: Faten Naif 
- Training algorithm: The algirthm uses bagging technique to creates a set of decision trees from random subsets of the training set. To make the final prediction it uses the majoirty vote of these different trees, which reduces overfitting.
- parameters: 
    - criterion=gini
    - max_depth=3
    - random_state=42
- fairness constraints: The dataset exhibits certain biases, including imbalanced class distributions and underrepresented groups. For example, the "Holand-Netherlands" under native countries contains only a single sample in addition to spelling mistake. Also, there is a potential data entry error, as one sample reports an unusually high value of 94 working hours per week, which may not be realistic.
- License: Apache-2.0
## Intended Use
- Primary intended uses: experimental and educational use as it was intended for learning to make ML inferences through an API.
- Primary intended users: Students who are learning ML and APIs.
- Out-of-scope use cases: This model should not be used for decision-making or to represent real-world scenarios, as the dataset contains inherent biases that may affect the reliability and fairness of the results.

## Training Data 
 - Trained on 80% of the income census data downloaded from https://archive.ics.uci.edu/dataset/20/census+income
 - data preprocessing:
    - Remove leading and trailing spaces in categorical columns.
    - Hot encoding the categorical columns.
    - Encode the target to numerical values.
## Evaluation Data
Details about the datasets used for evaluation
Datasets: 20% of the income census used for evaluation.
preprocessing: the same preprocessing as training set.
## Metrics
- Metrics: Precision, Recall and fbeta (with parameter beta = 1)
- Training scores:
    - Precision = 0.99
    - Recall = 0.12
    - fbeta = 0.22
- Validation scores:
    - Precision = 0.99
    - Recall = 0.14
    - fbeta = 0.24
## Ethical Considerations
- Potential bias in outputs for underrepresented groups.
## Caveats and Recommendations
- The training data contains inherent biases that should be carefully examined and          addressed, especially if they are not reflective of natural patterns in the data.
- Additional data cleaning and preprocessing are required to ensure the quality and integrity of the dataset.
- Class distributions should be balanced, and performance should be evaluated for each class individually to assess the model's quality and fairness across all classes.
