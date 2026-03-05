import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    # Remove missing values
    df = df.dropna()

    # Remove ID column
    df.drop('customerID', axis=1, inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({
        'Yes': 1,
        'No': 0
    })

    # One-hot encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    return df