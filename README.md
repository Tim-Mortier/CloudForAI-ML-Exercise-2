# CloudForAI-ML-Exercise-2

This project implements a machine learning pipeline to predict survival on the Titanic dataset. The solution is built using a Random Forest Classifier, focusing on clean, reproducible, and modular code for data preprocessing, model training, and evaluation.

# Installation

1. Clone the repository:

    ```
    git clone CloudForAI-ML-Exercise-2
    cd CloudForAI-ML-Exercise-2
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

# Usage
Running the Code

Ensure that the Titanic dataset CSV file is in the same directory as the script, and named Titanic-Dataset.csv.

1. Execute the script:

    ```
    python main.py
    ```

    The output will display:
    - Preprocessing status.
    - Evaluation metrics (accuracy, recall, and precision).

## Code Organization

The project is divided into the following modular functions:

- `load_data(file_path)`: Loads the Titanic dataset from a CSV file.

- `fill_missing_values(df)`: Handles missing values for `Age`, `Cabin`, and `Embarked`.

- `encode_features(df)`: Encodes categorical variables using `LabelEncoder` and `OneHotEncoder`.

- `preprocess_data_pipeline(file_path)`: Combines data loading, missing value handling, and feature encoding.

- `train_model_pipeline(X, y)`: Splits data into training and test sets and trains a Random Forest model.

- `evaluate_model_pipeline(model, X_test, y_test)`: Evaluates the model using accuracy, recall, and precision metrics. 

## Requirements


Install the required libraries via requirements.txt:

```
pip install -r requirements.txt
```
# Example Output

Random Forest Evaluation:

```
{'accuracy': 0.85, 'recall': 0.75, 'precision': 0.82}
```
