import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score

def load_data(file_path):
    """
    Load the Titanic dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def fill_missing_values(df):
    """
    Fill missing values in the dataset.

    Args:
        df (pd.DataFrame): Dataset with potential missing values.

    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df.dropna(subset=['Embarked'], inplace=True)
    return df

def encode_features(df):
    """
    Encode categorical features in the dataset.

    Args:
        df (pd.DataFrame): Dataset with categorical features.

    Returns:
        pd.DataFrame: Dataset with encoded features.
    """
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    embarked_encoded = one_hot_encoder.fit_transform(df[['Embarked']])
    embarked_df = pd.DataFrame(embarked_encoded, columns=one_hot_encoder.get_feature_names_out(['Embarked']))
    df = pd.concat([df, embarked_df], axis=1)
    df.drop('Embarked', axis=1, inplace=True)
    return df

def preprocess_data(file_path):
    """
    Load and preprocess the Titanic dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame, pd.Series: Preprocessed feature dataframe (X) and target series (y).
    """
    df = load_data(file_path)
    df = fill_missing_values(df)
    df = encode_features(df)
    df.dropna(inplace=True)
    X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
    y = df['Survived']
    return X, y

def train_model(X, y): 
    """
    Train a Random Forest model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        model: Trained Random Forest model object.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and visualize results.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.

    Returns:
        dict: Evaluation metrics (accuracy, recall, precision).
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

if __name__ == "__main__":
    file_path = "Titanic-Dataset.csv"
    X, y = preprocess_data(file_path)
    random_forest_model, X_test, y_test = train_model(X, y)
    rf_metrics = evaluate_model(random_forest_model, X_test, y_test)
    print(rf_metrics)
