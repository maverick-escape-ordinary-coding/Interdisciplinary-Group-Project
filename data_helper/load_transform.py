"""
Template to Load, Transform and Predict
"""

import pandas as pd
import psycopg2
import configparser
import urllib.request
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def get_config(config_file):
    config = configparser.ConfigParser()
    
    #jupyter notebook doesn't handle custom command line args very well, so verify if the
    #input ends in .ini. If not, default to app.ini

    if (config_file == None) or (not str(config_file).endswith(".ini")):
        config_file = 'app.ini'
    # verfy config location and basic structure
    try:
        config.read_file(open(config_file))
    except:
        raise Exception("{} not found".format(config_file))
    return config
    
# returns X, y and the original full dataframe (including unfiltered X, y) with the calculated columns added
def load_transform(config_file):
    config = get_config(config_file)
    try:
        db_params = config['DB']
    except:
        raise Exception("Config must contain a [DB] section")
    
    try:
        app_params = config['APP']
    except:
        raise Exception("Config must contain a [APP] section")
    
    # load data from SQL
    con = psycopg2.connect(host=db_params['host'], database=db_params['db'], user = db_params['user'], password = db_params['password'] )
    sql = "SELECT * from %s" % (db_params['table'])
    client_df = pd.read_sql_query(sql, con)

    client_df['_total_benefits'] = 0
    if ('benefits_column_list' in app_params):
        benefit_colmuns = app_params['benefits_column_list'].split(',')
        benefit_colmuns = [s.strip() for s in benefit_colmuns]
        for benefit_column in benefit_colmuns:
            #hax!, db contains stringy floats
            try:
                client_df[benefit_column].apply(float)
                client_df['_total_benefits'] += client_df[benefit_column]
            except:
                print('Float conversion failed')

   
    # adjusts the salary by calling the cost of living api
    # will fail if a city is not found in the the cost of living data
    def adjust_salary_by_location(reference, location, salary):
        location_parsed=location.split(' ')
        if len(location_parsed) > 0:
            location = location_parsed[0]
        if app_params['include_rent'].lower() == 'true':
            type = 'rent'
        else:
            type = 'consumer'

        response = urllib.request.urlopen(app_params['col_uri'].format(reference, location, type)).read().decode('utf-8')
        try:
            diff = json.loads(response)['% Diff']
        except:
            print('Error response: {}'.format(response))
            #raise Exception('Error response: {}'.format(response))      
            return(salary)  
        return (salary * (1+(diff/100)))

    #make cost of living adjustment to salary if needed 
    if ('use_cost_of_living' in app_params and app_params['use_cost_of_living'].lower() == 'true'):
        client_df['_adj_actual_salary_column'] = client_df.apply(
            lambda row : adjust_salary_by_location(app_params['location_reference'].lower(),
            row[app_params['location_column']], 
            row[app_params['actual_salary_column']]), axis = 1)        
        client_df['_total_annual_pkg'] = client_df['_adj_actual_salary_column'] + client_df['_total_benefits'] 
    else:
        client_df['_total_annual_pkg'] = client_df[app_params['actual_salary_column']] + client_df['_total_benefits'] 
    
    #adj for part time values if ness
    if ('weekly_hours_column' in app_params):
        client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg'] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])
    else: 
        client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg']

    # treat outliers and scale total package if needed
    if ('outlier_percentiles' in app_params):
        values = [s.strip() for s in app_params['outlier_percentiles'].split(',')]
        low, high = client_df['_total_annual_pkg_fte'].quantile( [float(values[0]), float(values[1])] )
        client_df = client_df.query("_total_annual_pkg_fte >= {} & _total_annual_pkg_fte <= {}".format(low, high))
        #reset index, make it linear again for ease of joining below
        client_df.reset_index(drop=True, inplace=True)


    scaler = MinMaxScaler(feature_range=(0, 1))
    # note this scales directly on the original dataframe, use only for feature selection.
    # set to false when predicting.
    if ('scale_target' in app_params and app_params['scale_target'].lower() == 'true'):
        client_df['_total_annual_pkg_fte'] = scaler.fit_transform(client_df['_total_annual_pkg_fte'].values.reshape(-1, 1))

    # oh, cat
    df_oh_cat_features = df_label_cat_features = df_num_features = None

    # produce a dataframe that contains our one hot encoded features
    if ('oh_cat_feature_list' in app_params):
        oh_cat_features = app_params['oh_cat_feature_list'].split(',')
        oh_cat_features = [s.strip() for s in oh_cat_features]
        df_oh_cat_features = pd.get_dummies(data=client_df[oh_cat_features], drop_first=True)
        if ('scale_cat_features' in app_params and app_params['scale_cat_features'].lower() == 'true'):
            for column in df_oh_cat_features.columns:
                df_oh_cat_features[column] = scaler.fit_transform(df_oh_cat_features[column].values.reshape(-1, 1)) 

    # produce a dataframe that contains our label encoded features
    if ('label_cat_feature_list' in app_params):
        label_cat_features = app_params['label_cat_feature_list'].split(',')
        label_cat_features = [s.strip() for s in label_cat_features]
        le = LabelEncoder()
        for feature in label_cat_features:
            if (type(df_label_cat_features) != pd.DataFrame):
                df_label_cat_features = pd.DataFrame()
            le.fit(client_df[feature])
            df_label_cat_features[feature] = le.transform(client_df[feature])
            if ('scale_label_features' in app_params and app_params['scale_label_features'].lower() == 'true'):
                df_label_cat_features[feature] = scaler.fit_transform(df_label_cat_features[feature].values.reshape(-1, 1)) 

    # produce a dataframe that contains our numerical features
    if ('num_feature_list' in app_params):
        num_features = app_params['num_feature_list'].split(',')
        num_features = [s.strip() for s in num_features]
        df_num_features = client_df[num_features]
        if ('scale_num_features' in app_params and app_params['scale_num_features'].lower() == 'true'):
            for feature in num_features:
                df_num_features[feature] = scaler.fit_transform(df_num_features[feature].values.reshape(-1, 1))

    # create X from our three feature dataframes if they have been initialised
    X = pd.DataFrame()
    for features in (df_oh_cat_features, df_num_features, df_label_cat_features): #
        if type(features) == pd.DataFrame:
            if (X.empty):
                X = features
            else:    
                X = X.join(features)
                
    # create a copy of X, y in case they are further filtered below
    X_full = X.copy()
    y_full = client_df['_total_annual_pkg_fte']
    
    drop = False
    if ('train_split_column' in app_params):
        if (app_params['train_split_column'] not in X):
            drop = True
            X[app_params['train_split_column']] = client_df[app_params['train_split_column']]
        X = X.query("{} == '{}'".format(app_params['train_split_column'], app_params['train_split_value']))
        y = client_df.query("{} == '{}'".format(app_params['train_split_column'], app_params['train_split_value']))['total_annual_pkg_fte']
        if drop:
            X.drop(app_params['train_split_column'], axis = 1, inplace=True )
    else:
        y = y_full    

    # X, the feature df, may be filtered by the config to exclude rows
    # y, the known truths matching X
    # X_full, unfiltered features
    # y_full, unfiltered known truths matching X_Full
    # client_df, the full client dataframe with the calculated columns added,
    # unfiltered but with outliers removed
    return X, y, X_full, y_full, client_df