# experiment, model name and stage to load the model from mlflow model registry
#MODEL_NAME = 
#STAGE = 
#EXPERIMENT = 

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES =['first_platform_c_Level10', 'first_platform_c_Level11','first_platform_c_Level12', 'first_platform_c_Level13',
'first_platform_c_Level14', 'first_platform_c_Level15','first_platform_c_Level17', 'first_platform_c_Level18',
'first_platform_c_Level19', 'first_platform_c_Level2','first_platform_c_Level20', 'first_platform_c_Level21',
'first_platform_c_Level22', 'first_platform_c_Level26', 'city_tier_2.0','first_platform_c_Level3', 'city_tier_3.0', 'first_platform_c_Level1',
'first_platform_c_Level28', 'first_platform_c_Level29', 'first_platform_c_Level27',
'first_platform_c_Level4', 'first_platform_c_Level25','first_platform_c_Level24', 'first_platform_c_Level40',
'first_platform_c_Level39', 'first_platform_c_Level38','first_platform_c_Level32', 'first_platform_c_Level23',
'first_platform_c_Level34', 'first_platform_c_Level43','first_platform_c_Level42', 'first_platform_c_Level36',
'first_platform_c_Level16', 'first_platform_c_Level33','first_platform_c_Level41', 'first_platform_c_Level35',
'first_platform_c_Level37', 'first_platform_c_Level30','first_platform_c_Level31']

# list of features that need to be one-hot encoded
# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['city_tier', 'first_platform_c', 'first_utm_medium_c','first_utm_source_c'] 

import pandas as pd


DATA_DIRECTORY = '/home/dags/Lead_scoring_inference_pipeline/data'
INPUT_FILE_NAME= '/leadscoring_inference.csv'

INDEX_COLUMNS = ['first_platform_c','first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
       'referred_lead', 'city_tier']

#############################################################################################################



import pandas as pd
DB_PATH ='/home/dags/Lead_scoring_inference_pipeline'

interaction_mapping_file=pd.read_csv('/home/dags/Lead_scoring_data_pipeline/mapping/interaction_mapping.csv')
DB_FILE_NAME='/lead_scoring_data_cleaning.db' 

TRACKING_URL ="Http://0.0.0.0:6007/" 
 
ml_flow_path ='/home/dags/Lead_scoring_training_pipeline/mlruns/0/fa10837487c34e4ab7c4a48c45bde971/artifacts/models'

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES =['first_platform_c_Level10', 'first_platform_c_Level11','first_platform_c_Level12', 'first_platform_c_Level13',
'first_platform_c_Level14', 'first_platform_c_Level15','first_platform_c_Level17', 'first_platform_c_Level18',
'first_platform_c_Level19', 'first_platform_c_Level2','first_platform_c_Level20', 'first_platform_c_Level21',
'first_platform_c_Level22', 'first_platform_c_Level26', 'city_tier_2.0','first_platform_c_Level3', 'city_tier_3.0', 'first_platform_c_Level1',
'first_platform_c_Level28', 'first_platform_c_Level29', 'first_platform_c_Level27',
'first_platform_c_Level4', 'first_platform_c_Level25','first_platform_c_Level24', 'first_platform_c_Level40',
'first_platform_c_Level39', 'first_platform_c_Level38','first_platform_c_Level32', 'first_platform_c_Level23',
'first_platform_c_Level34', 'first_platform_c_Level43','first_platform_c_Level42', 'first_platform_c_Level36',
'first_platform_c_Level16', 'first_platform_c_Level33','first_platform_c_Level41', 'first_platform_c_Level35',
'first_platform_c_Level37', 'first_platform_c_Level30','first_platform_c_Level31']

# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['city_tier', 'first_platform_c', 'first_utm_medium_c','first_utm_source_c']

MODEL_CONFIG = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'device':'gpu',
        'importance_type': 'split' ,
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'num_leaves': 31,
        'objective': None,
        'random_state': 42,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'silent': 'warn',
        'subsample': 1.0,
        'subsample_for_bin': 200000 ,
        'subsample_freq': 0
        }

city_tier_mapping = {'bengaluru': 1,
     'chennai': 1,
     'hyderabad': 1,
     'kolkata': 1,
     'mumbai': 1,
     'ncr': 1,
     'pune': 1,
     'agra': 2,
     'ahmedabad': 2,
     'aligarh': 2,
     'anand': 2,
     'bhopal': 2,
     'coimbatore': 2,
     'gandhinagar': 2,
     'gwalior': 2,
     'indore': 2,
     'jabalpur': 2,
     'jaipur': 2,
     'kanpur': 2,
     'kochi': 2,
     'lucknow': 2,
     'ludhiana': 2,
     'madurai': 2,
     'meerut': 2,
     'nagpur': 2,
     'nashik': 2,
     'patna': 2,
     'rajkot': 2,
     'solapur': 2,
     'surat': 2,
     'tiruchirapalli': 2,
     'vadodara': 2,
     'vijayawada': 2,
     'vishakapatnam': 2,
     'allahabad': 3,
     'amravati': 3,
     'amritsar': 3,
     'aurangabad': 3,
     'bhavnagar': 3,
     'bhilai': 3,
     'bhubaneswar': 3,
     'chandigarh': 3,
     'dehradun': 3,
     'dhanbad': 3,
     'etawah': 3,
     'faridabad': 3,
     'gorakhpur': 3,
     'guntur': 3,
     'guwahati': 3,
     'hajipur': 3,
     'hubli-dharwad': 3,
     'jammu': 3,
     'jamnagar': 3,
     'jamshedpur': 3,
     'jodhpur': 3,
     'kannur': 3,
     'kollam': 3,
     'kozhikode': 3,
     'kurnool': 3,
     'mangalore': 3,
     'mysore': 3,
     'nellore': 3,
     'patiala': 3,
     'pondicherry': 3,
     'raipur': 3,
     'ranchi': 3,
     'roorkee': 3,
     'salem': 3,
     'sangli': 3,
     'srinagar': 3,
     'thiruvananthapuram': 3,
     'thrissur': 3,
     'tirunelveli': 3,
     'varanasi': 3,
     'vellore': 3,
     'warangal': 3}
# levels are stored in list for further processing

list_platform=['Level0', 'Level3', 'Level7', 'Level1', 'Level2', 'Level8']

list_medium=['Level0', 'Level2', 'Level6', 'Level3', 'Level4', 'Level9', 'Level11', 'Level5', 'Level8', 'Level20', 'Level13', 'Level30', 'Level33', 'Level16', 'Level10', 'Level15', 'Level26', 'Level43']

list_source=['Level2', 'Level0', 'Level7', 'Level4', 'Level6', 'Level16', 'Level5', 'Level14']


raw_data_schema = ['created_date', 'city_mapped', 'first_platform_c',
           'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
           'referred_lead', '1_on_1_industry_mentorship', 'call_us_button_clicked',
           'career_assistance', 'career_coach', 'career_impact', 'careers',
           'chat_clicked', 'companies', 'download_button_clicked',
           'download_syllabus', 'emi_partner_click', 'emi_plans_clicked',
           'fee_component_click', 'hiring_partners',
           'homepage_upgrad_support_number_clicked',
           'industry_projects_case_studies', 'live_chat_button_clicked',
           'payment_amount_toggle_mover', 'placement_support',
           'placement_support_banner_tab_clicked', 'program_structure',
           'programme_curriculum', 'programme_faculty',
           'request_callback_on_instant_customer_support_cta_clicked',
           'shorts_entry_click', 'social_referral_click',
           'specialisation_tab_clicked', 'specializations', 'specilization_click',
           'syllabus', 'syllabus_expand', 'syllabus_submodule_expand',
           'tab_career_assistance', 'tab_job_opportunities', 'tab_student_support',
           'view_programs_page', 'whatsapp_chat_click']

model_input_schema = ['first_platform_c','first_utm_medium_c','first_utm_source_c','total_leads_dropped','referred_lead', 
                    'city_tier']

