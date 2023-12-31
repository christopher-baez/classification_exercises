import env
import os
import pandas as pd
import numpy as np
from pydataset import data

def check_file_exists(filename, query, url):
    if os.path.exists(filename):
        print('this file exists, reading csv')
        df = pd.read_csv(filename)
    else:
        print('this file doesnt exist, read from sql, and export to csv')
        df = pd.read_sql(query, url)
        df.to_csv(filename)

    return df


def get_telco_data():
    ''' function to get data from SQL  and turning it to a dataframe'''
    url = env.get_db_url('telco_churn')
    query = '''
    select *
    from customers
        left join contract_types
            using (contract_type_id)
        left join internet_service_types
            using (internet_service_type_id)
        left join payment_types
            using (payment_type_id)
    '''

    filename = 'telco_churn.csv'


    # call the check_file_exists fuction
    df = check_file_exists(filename, query, url)

    df.set_index(df.columns[0], inplace=True)
    return df


# %%
def get_iris_data():
    url = env.get_db_url('iris_db')
    query = '''
    select * 
    from measurements
        join species
            using (species_id)
    '''

    filename = 'iris.csv'

    # call the check_file_exists fuction
    df = check_file_exists(filename, query, url)

    df.set_index(df.columns[0], inplace=True)
    return df


def get_titanic_data():
    url = env.get_db_url('titanic_db')
    query = 'select * from passengers'

    filename = 'titanic.csv'

    # call the check_file_exists fuction
    df = check_file_exists(filename, query, url)

    df.set_index(df.columns[0], inplace=True)
    return df