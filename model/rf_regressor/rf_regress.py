#%% setup / create Xy
import pandas as pd
import numpy as np
import psycopg2
import configparser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from fs_generic import feature_selection
from len_service_change import service_float
import sys

sys.path.append("../../data_helper")
from load_transform import load_transform, get_config

def plot_feature_importances(model, data):
    n_features = len(data.columns)
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

#takes the first arg on the command line as the config filename
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        config_file_name = sys.argv[1]
    else:
        config_file_name = None

X, y, X_full, y_full, client_df = load_transform(config_file_name)
app_params = get_config(config_file_name)['APP']


if ('weekly_hours_column' in app_params):
    client_df['_annual_salary_fte'] = client_df[app_params['actual_salary_column']] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])
else:
    client_df['_annual_salary_fte'] = client_df[app_params['actual_salary_column']]

client_df['_total_annual_pkg'] = client_df[app_params['actual_salary_column']]

if ('benefits_column_list' in app_params):
    benefit_colmuns = app_params['benefits_column_list'].split(',')
    benefit_colmuns = [s.strip() for s in benefit_colmuns]
    for benefit_column in benefit_colmuns:
        #hax!, db contains stringy floats
        try:
            client_df[benefit_column].apply(float)
            client_df['_total_annual_pkg'] += client_df[benefit_column]
        except:
            print('Float conversion failed')

if ('weekly_hours_column' in app_params):
    client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg'] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])
else: 
    client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg']

#treat outliers and scale total package if needed
if ('outlier_percentiles' in app_params):
    values = [s.strip() for s in app_params['outlier_percentiles'].split(',')]
    low, high = client_df['_total_annual_pkg_fte'].quantile( [float(values[0]), float(values[1])] )
    client_df = client_df.query("_total_annual_pkg_fte >= {} & _total_annual_pkg_fte <= {}".format(low, high))
    #reset index, make it linear again for ease of joining below
    client_df = client_df.reset_index()

scaler = MinMaxScaler(feature_range=(0, 1))
#note this scales directly on the original dataframe, use only for feature selection.
#set to false when predicting.
if ('scale_target' in app_params and app_params['scale_target'].lower() == 'true'):
    client_df['_total_annual_pkg_fte'] = scaler.fit_transform(client_df['_total_annual_pkg_fte'].values.reshape(-1, 1))

#oh, cat
df_oh_cat_features = df_label_cat_features = df_num_features = None

#produce a dataframe that contains our one hot encoded features
if ('oh_cat_feature_list' in app_params):
    oh_cat_features = app_params['oh_cat_feature_list'].split(',')
    oh_cat_features = [s.strip() for s in oh_cat_features]
    df_oh_cat_features = pd.get_dummies(data=client_df[oh_cat_features], drop_first=True)
    if ('scale_cat_features' in app_params and app_params['scale_cat_features'].lower() == 'true'):
        for column in df_oh_cat_features.columns:
            df_oh_cat_features[column] = scaler.fit_transform(df_oh_cat_features[column].values.reshape(-1, 1)) 

#produce a dataframe that contains our label encoded features
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

#produce a dataframe that contains our numerical features
if ('num_feature_list' in app_params):
    num_features = app_params['num_feature_list'].split(',')
    num_features = [s.strip() for s in num_features]
    df_num_features = client_df[num_features]
    # just added this condition to client_a dataset to change format of lenght of service from 'x years y months' to float(x.y/12)
    if ('client_a' in app_params):
        df_num_features = service_float(df_num_features)
    if ('scale_num_features' in app_params and app_params['scale_num_features'].lower() == 'true'):
        for feature in num_features:
            df_num_features[feature] = scaler.fit_transform(df_num_features[feature].values.reshape(-1, 1))

#create X from our three feature dataframes if they have been initialised
X = pd.DataFrame()
for features in (df_oh_cat_features, df_num_features, df_label_cat_features): #
    if type(features) == pd.DataFrame:
        if (X.empty):
            X = features
        else:    
            X = X.join(features)


# limiting amount of features by feature selection
sel_feats = feature_selection(config_file_name)
X = X[X.columns.intersection(sel_feats)]
#create a copy of X for use further down to avoid a modified version
X_full = X.copy()

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
    y = client_df['_total_annual_pkg_fte']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#%% model
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("r2 for test RF {}".format(rf.score(X_test, y_test)))
print("r2 for train RF {}".format(rf.score(X_train, y_train)))

#reset y just in case it was filtered above
y_full = client_df['_total_annual_pkg_fte']
# y_full = y
y_pred = rf.predict(X_full)

print("r2 for full RF {}".format(rf.score(X_full, y_full)))
client_df['_predicted_pkg'] = y_pred

plot_feature_importances(rf,X_full)

#%% update overpaid/underpaid 
c_under = (client_df['_predicted_pkg'] - client_df['_total_annual_pkg_fte']) > client_df['_total_annual_pkg_fte'] * float(app_params['pay_threshold'])
c_over = (client_df['_predicted_pkg'] - client_df['_total_annual_pkg_fte']) <  -client_df['_total_annual_pkg_fte'] * float(app_params['pay_threshold'])
values = ['under', 'over', 'fair']
conditions = [c_under, c_over, True]

client_df['_pay_status'] = np.select(conditions, values)

#%% plot results
female_over = client_df.query("gender == 'Female' & _pay_status == 'over'").shape[0]
female_under = client_df.query("gender == 'Female' & _pay_status == 'under'").shape[0]
female_fair = client_df.query("gender == 'Female' & _pay_status == 'fair'").shape[0]
male_over = client_df.query("gender == 'Male' & _pay_status == 'over'").shape[0]
male_under = client_df.query("gender == 'Male' & _pay_status == 'under'").shape[0]
male_fair = client_df.query("gender == 'Male' & _pay_status == 'fair'").shape[0]
male = [male_under, male_fair, male_over]
female = [female_under, female_fair, female_over]

print(female)
print(male)


labels = ['Under', 'Fair', 'Over']
width = 0.35

fig, ax = plt.subplots()
ax.bar(labels, male, width, label='Men')
ax.bar(labels, female, width, bottom = male, label='Women')

ax.set_ylabel('n')
ax.set_title('Pay Status By Gender')

ax.legend()
plt.savefig('Pay_status_rf.png', bbox_inches='tight')

plt.show()

#%% pay stats
male = client_df.query("gender == 'Male'")
female = client_df.query("gender == 'Female'")
mean_male_real = male['_total_annual_pkg_fte'].mean()
mean_female_real = female['_total_annual_pkg_fte'].mean()

mean_male_predict = male['_predicted_pkg'].mean()
mean_female_predict = female['_predicted_pkg'].mean()

print("Actual: For every £1 paid to a male, a female is paid £{:.2f}".format(mean_female_real/mean_male_real))
print("Predicted: For every £1 paid to a male, a female is paid £{:.2f}".format(mean_female_predict/mean_male_predict))

total_p = client_df['_total_annual_pkg_fte'].sum()
total_a = client_df['_predicted_pkg'].sum()

print('Total Annual Salaries Actual: {:.2f} Predicted: {:.2f} Diff: {:.2f}'.format(total_a, total_p, total_a - total_p))

