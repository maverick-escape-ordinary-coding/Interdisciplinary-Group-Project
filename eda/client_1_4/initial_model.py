import pandas as pd
import numpy as np
import psycopg2
import configparser

from sklearn.preprocessing import MinMaxScaler

config = configparser.ConfigParser()
config.read('app.ini')
db_params = config['DB']
app_params = config['APP']

#load data from SQL
con = psycopg2.connect(host='localhost', database=db_params['db'], user = db_params['user'], password = db_params['password'] )
sql = "SELECT * from %s" % (db_params['table'])
client_df = pd.read_sql_query(sql, con)
print(client_df.columns)
client_df['annual_salary_fte'] = client_df[app_params['actual_salary_column']] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])

client_df['total_annual_pkg'] = client_df[app_params['actual_salary_column']]

if ('benefits_column_list' in app_params):
    benefit_colmuns = app_params['benefits_column_list'].split(',')
    benefit_colmuns = [s.strip() for s in benefit_colmuns]
    for benefit_column in benefit_colmuns:
        client_df['total_annual_pkg'] += client_df[benefit_column]

client_df['total_annual_pkg_fte'] = client_df['total_annual_pkg'] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])

features = app_params['feature_list'].split(',')
features = [s.strip() for s in features]

for feature in features:
    average_val = client_df.groupby(feature, as_index=False)['total_annual_pkg_fte'].mean()
    
    average_val_dict = pd.Series(average_val.total_annual_pkg_fte.values, index=average_val[feature]).to_dict()
    client_df.insert(loc=client_df.columns.get_loc(feature) + 1, column='norm_salary_%s' % feature,
                       value=[np.nan for i in range(client_df.shape[0])])
    client_df["norm_salary_%s" % feature] = client_df[feature].apply(lambda x: average_val_dict.get(x))

norm_values = []
for col in client_df.columns:
    if col[:4] == 'norm':
        norm_values.append(col)

average_salary = client_df[norm_values].mean(axis=1)
client_df.insert(loc=client_df.columns.get_loc('total_annual_pkg_fte') + 1, column='quasi_predicted_pkg',
                   value=average_salary)

raw_difference = client_df['total_annual_pkg_fte'] - client_df['quasi_predicted_pkg']
client_df.insert(loc=client_df.columns.get_loc('quasi_predicted_pkg') + 1, column='raw_pkg_difference',
                   value=raw_difference)

scaler = MinMaxScaler(feature_range=(-1,1))
SD = scaler.fit_transform(client_df.raw_pkg_difference.values.reshape(-1, 1))
client_df.insert(loc=client_df.columns.get_loc('raw_pkg_difference') + 1, column='SD',
                   value=SD.flatten())

a = min(client_df.raw_pkg_difference, key=abs)
print(a)
for value in client_df.raw_pkg_difference:
    if value == a:
        print(value)
        print(client_df.index[client_df['raw_pkg_difference'] == value].tolist())

pay_label = []
numeric_pay_label = []
for value in client_df.SD:
    if value < -0.6:
        pay_label.append('underpaid')
        numeric_pay_label.append(-1)
    elif value > -0.45:
        pay_label.append('overpaid')
        numeric_pay_label.append(1)
    else:
        pay_label.append('fairly_paid')
        numeric_pay_label.append(0)
client_df.insert(loc=client_df.columns.get_loc('raw_pkg_difference') + 2, column='pay_label',
                   value=pay_label)
client_df.insert(loc=client_df.columns.get_loc('raw_pkg_difference') + 3, column='numeric_pay_label',
                   value=numeric_pay_label)

output_columns = features.copy()
output_columns.extend(['total_annual_pkg_fte', 'quasi_predicted_pkg','raw_pkg_difference','pay_label'])
print(features)

print(client_df[output_columns])