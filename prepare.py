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

    """
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'Unnamed: 0', ])

    df = df[df.total_charges != ' ']

    df.total_charges.astype('float64')

    return df


def splitting_data(df):
    train, validate_test = train_test_split(df,  # send in initial df
                                            train_size=0.60,
                                            # size of the train df, and the test size will default to 1-train_size
                                            random_state=123,  # set any number here for consistency
                                            stratify=df.survived  # need to stratify on target variable
                                            )

    validate, test = train_test_split(validate_test,  # this is the df that we are splitting now
                                      test_size=0.50,  # set test or train size to 50%
                                      random_state=123,  # gotta send in a random seed
                                      stratify=validate_test.survived  # still got to stratify
                                      )
    return train, validate, test
