from sklearn.model_selection import train_test_split


def prep_iris(df):
    """
    this function takes in the iris dataframe and returns a cleaned version
    """
    # Replace periods with underscores in the entire DataFrame
    df = df.replace('.', '_')

    # Drop the 'measurement_id' and 'species_id' columns
    df = df.drop(columns=['measurement_id', 'species_id'])

    # Rename 'species_name' column to 'species'
    df = df.rename(columns={'species_name': 'species'})

    return df


def prep_telco(df):
    """
    this  function takes in the telco database and cleans it
    deletes
        -payment type id
        -internet service id
        -contract type id
        -customer id
    turns total charges into a float by deleting empty spaces
    """
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id', ])

    df = df[df.total_charges != ' ']

    df.total_charges.astype('float64')

    return df


def clean_titanic(df):
    """
    students - write docstring
    """
    # drop unncessary columns
    df = df.drop(columns=['embarked', 'age', 'deck', 'class'])

    # made this a string so its categorical
    df.pclass = df.pclass.astype(object)

    # filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')

    return df


def splitting_data(df, col):
    '''
    this function splits my data focusing on my target variable
    parameters
        df= datafram
        col= column
    returns
        train
        validate
        test
    '''

    # first split
    train, validate_test = train_test_split(df,
                                            train_size=0.6,
                                            random_state=123,
                                            stratify=df[col]
                                            )

    # second split
    validate, test = train_test_split(validate_test,
                                      train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col]

                                      )
    return train, validate, test
