import numpy as np
import pandas as pd

def preprocess_titanic(train, validate, test):
    # Define a function that applies the encoding
    def encode(df):
        # Encoding the 'sex' column
        df_encoded = df.copy()
        df_encoded['sex_male'] = np.where(df_encoded['sex'] == 'male', 1, 0)
        df_encoded['sex_female'] = np.where(df_encoded['sex'] == 'male', 0, 1)

        # Encoding the 'embark_town' column
        embark_town_dummies = pd.get_dummies(df_encoded['embark_town'], drop_first=True).astype(int)
        df_encoded[['is_queenstown', 'is_southampton']] = embark_town_dummies

        # Drop the original categorical columns and unnecesary
        df_encoded = df_encoded.drop(columns=['sex', 'embark_town'])
        df_encoded = df_encoded.drop(columns='passenger_id')
        return df_encoded

    # Apply the encoding function to each dataset and store in new variables
    encoded_train = encode(train)
    encoded_validate = encode(validate)
    encoded_test = encode(test)

    return encoded_train, encoded_validate, encoded_test

def preprocess_telco(train, validate, test):
    ''' Encode the categorical variables in the train, validate, and test datasets.

    This function performs the following encoding steps:
    1. Encodes the 'gender' column into a new binary column 'is_male', where 'Male' is 1 and 'Female' is 0.
    2. Encodes the 'partner' column into a new binary column 'alone', where 'Yes' is mapped to 0 (not alone) and 'No' is mapped to 1 (alone).
    3. Creates dummy variables for other specified categorical columns such as 'contract_type' and 'internet_service_type', dropping the first category to avoid multicollinearity.

    Parameters:
    - train (DataFrame): The training dataset.
    - validate (DataFrame): The validation dataset.
    - test (DataFrame): The test dataset.

    The function modifies these datasets in place by adding new encoded columns and returns them.

    Returns:
    - DataFrame: The modified train dataset with encoded variables.
    - DataFrame: The modified validate dataset with encoded variables.
    - DataFrame: The modified test dataset with encoded variables.
    '''
    # dropping the columns changed
    train = train.drop(columns=['gender', 'partner', 'contract_type', 'internet_service_type'])
    validate = validate.drop(columns=['gender', 'partner', 'contract_type', 'internet_service_type'])
    test = test.drop(columns=['gender', 'partner', 'contract_type', 'internet_service_type'])
    # Define encoding for 'gender' using .replace()
    gender_encode = {'Male': 1, 'Female': 0}
    train['is_male'] = train['gender'].replace(gender_encode)
    validate['is_male'] = validate['gender'].replace(gender_encode)
    test['is_male'] = test['gender'].replace(gender_encode)

    # Define encoding for 'partner' using .map()
    partner_encode = {'Yes': 0, 'No': 1}
    train['alone'] = train['partner'].map(partner_encode)
    validate['alone'] = validate['partner'].map(partner_encode)
    test['alone'] = test['partner'].map(partner_encode)

    # Create dummy variables for other categorical columns
    # Adjust these based on your specific categorical columns
    for column in ['contract_type', 'internet_service_type']:
        train_dummies = pd.get_dummies(train[column], drop_first=True)
        validate_dummies = pd.get_dummies(validate[column], drop_first=True)
        test_dummies = pd.get_dummies(test[column], drop_first=True)

        train = pd.concat([train, train_dummies], axis=1)
        validate = pd.concat([validate, validate_dummies], axis=1)
        test = pd.concat([test, test_dummies], axis=1)

    return encoded_train, encoded_validate, encoded_test



