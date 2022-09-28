###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging

from datetime import datetime

import pandas as pd
import os
import sqlite3
from sqlite3 import Error

from constants import *
from utils import *


#def data_preprocessing_pipeline():

def data_preprocessing pipeline():
    # Define function to load the csv file to the database
    def load_data_into_db(DATA_DIRECTORY,DB_PATH,DB_FILE_NAME,INPUT_FILE_NAME):
        df=pd.read_csv(DATA_DIRECTORY + INPUT_FILE_NAME )
        df['total_leads_droppped'].fillna(0, inplace=True)
        df['referred_lead'].fillna(0, inplace=True)

        conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
        print("Connection Successful",conn)
        df.to_sql('leadscoring_inference',con=conn,if_exists='replace',index=False)
        df_read=pd.read_sql('select * from leadscoring_inference', conn)
        return(df_read.head())

    # Define function to map cities to their respective tiers
    def map_city_tier(DB_PATH,city_tier_mapping,DB_FILE_NAME):
        conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)

        df = pd.read_sql("select * from leadscoring_inference", con=conn)
        df_lead_scoring=df
        df_lead_scoring["city_tier"] = df_lead_scoring["city_mapped"].map(city_tier_mapping)
        df_lead_scoring["city_tier"] = df_lead_scoring["city_tier"].fillna(3.0)
        df=df_lead_scoring

        print("Connection Successful",conn)
        df.to_sql('inference_city_tier_mapped',con=conn,if_exists='replace',index=False)

    # Define function to map insignificant categorial variables to "others"
    def map_categorical_vars(DB_PATH,DB_FILE_NAME):
        conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)

        df_lead_scoring= pd.read_sql("select * from inference_city_tier_mapped",con=conn)

        #df_lead_scoring=pd.read_csv('C:\\Users\\44775\\Desktop\\Assignment\\01_data_pipeline\\scripts\\data\\leadscoring.csv')
         # cumulative percetage is calculated for all the three columns - first_platform_c, first_utm_medium_c, first_utm_source_c
        for column in ['first_platform_c', 'first_utm_medium_c', 'first_utm_source_c']:
            df_cat_freq = df_lead_scoring[column].value_counts()
            df_cat_freq = pd.DataFrame({'column':df_cat_freq.index, 'value':df_cat_freq.values})
            df_cat_freq['perc'] = df_cat_freq['value'].cumsum()/df_cat_freq['value'].sum()


        # all the levels below 90 percentage are assgined to a single level called others
        new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(list_platform)] # get rows for levels which are not present in      list_platform
        new_df.loc['first_platform_c'] = "others" # replace the value of these levels to others
        old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(list_platform)] # get rows for levels which are present in list_platform
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe

        # all the levels below 90 percentage are assgined to a single level called others
        new_df = df[~df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are not present in list_medium
        new_df.loc['first_utm_medium_c'] = "others" # replace the value of these levels to others
        old_df = df[df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are present in list_medium
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe

        # all the levels below 90 percentage are assgined to a single level called others
        new_df = df[~df['first_utm_source_c'].isin(list_source)] # get rows for levels which are not present in list_source
        new_df.loc['first_utm_source_c'] = "others" # replace the value of these levels to others
        old_df = df[df['first_utm_source_c'].isin(list_source)] # get rows for levels which are present in list_source
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe

        print("Connection Successful",conn)
        df.to_sql('inference_categorical_variables_mapped',conn,if_exists='replace',index=False)
        df_read=pd.read_sql('select * from inference_categorical_variables_mapped', conn)
        return(df_read.head())


    # Define function that maps interaction columns into 4 types of interactions
    def interactions_mapping(DB_PATH,interaction_mapping_file,INDEX_COLUMNS,DB_FILE_NAME):

        # Read dataframe from the database
        conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
        df = pd.read_sql("select * from inference_categorical_variables_mapped", con=conn)


        # unpivot the interaction columns and put the values in rows
        df_unpivot = pd.melt(df, id_vars=INDEX_COLUMNS, var_name='interaction_type', value_name='interaction_value')
        # handle the nulls in the interaction value column
        df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
        # map interaction type column with the mapping file to get interaction mapping
        interaction_mapping_file=pd.DataFrame(interaction_mapping_file)
        df = pd.merge(df_unpivot,interaction_mapping_file, on='interaction_type', how='left')
        #dropping the interaction type column as it is not needed
        df = df.drop(['interaction_type'], axis=1)
         # pivoting the interaction mapping column values to individual columns in the dataset
        df_pivot = df.pivot_table(
            values='interaction_value', index=INDEX_COLUMNS, columns='interaction_mapping', aggfunc='sum')
        df_pivot = df_pivot.reset_index()
        #df_pivot.drop(['created_date'],axis=1,inplace=True)


        # generate the profile report of final dataset

        #profile = ProfileReport(df_pivot, title="Cleaned Data Summary")
        #profile.to_notebook_iframe()
        #profile.to_file('profile_report/cleaned_data_report.html')
        # the file is written in the data folder
        print("Connection Successful",conn)
        df_pivot.to_sql('inference_cleaned_data',conn,if_exists='replace',index=False)
        return(df_pivot.head())

    # Define function to validate model's input schema
    def model_input_schema_check(model_input_schema):
        conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
        df= pd.read_sql("select * inference_cleaned_data",con=conn)

        if (df.columns==model_input_schema).all():
            print("Connection Successful",conn)
            df.to_sql('inference_cleaned_data',conn,if_exists='replace',index=False)
            return('Raw datas schema is in line with the schema present in schema.py')
        else:
            return('Raw datas schema is NOT in line with the schema present in schema.py')
    
    #Calling the function
    load_data_into_db(DATA_DIRECTORY,DB_PATH,DB_FILE_NAME,INPUT_FILE_NAME)
    map_city_tier(DB_PATH,city_tier_mapping,DB_FILE_NAME)
    map_categorical_vars(DB_PATH,DB_FILE_NAME)
    interactions_mapping(DB_PATH,interaction_mapping_file,INDEX_COLUMNS,DB_FILE_NAME)




###############################################################################
# Define the function to train the model
# ##############################################################################



def encode_features(DB_PATH,DB_FILE_NAME):
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df=pd.read_sql('Select * from inference_cleaned_data',conn)
    df['city_tier']=df['city_tier'].astype('object')
    index_names = df[df['total_leads_dropped'] == 'others' ].index

    # drop these row indexes
    # from dataFrame
    df.drop(index_names, inplace = True)
    df=pd.get_dummies(df, columns =FEATURES_TO_ENCODE ,drop_first=True)

    #Connect the database

    print("Connection Successful",conn)
    df.to_sql('features',con=conn,if_exists='replace',index=False)

    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction():
    pass
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''

###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check():
    pass
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.


    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################


def input_features_check():
    pass
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
